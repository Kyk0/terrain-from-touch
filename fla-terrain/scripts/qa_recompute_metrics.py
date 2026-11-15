#!/usr/bin/env python
"""B14: independent re-derivation of the headline D7 numbers from raw episodes + saved predictions.

  python scripts/qa_recompute_metrics.py [--data data/v1] [--pred a/results/v1] [--metrics results/v1/metrics.json]

Does NOT import fla_terrain.models (evaluate/report/dataset). Taken from A's dataset.py only to decode the
prediction files: episode order = sorted(glob('<data>/**/episode_*.npz')), windows = starts 0, 10, 20, ... with
length 50 (start + 50 <= T), windows whose last frame t > fall_time dropped, concatenated over episodes; pred `idx` are
indices into that concatenated list; split = terrain_params.range_split. Everything else (labels at the last frame,
contact, touchdowns, metric definitions from DECISIONS.md D7) is computed here from the raw episode files.
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np

W, S = 50, 10
TD_AIR = 4          # own touchdown detector: contact_gt rising edge after >= 4 frames (80 ms) without contact
NAMES = ["hard_grip", "hard_slip", "soft", "rough", "slope", "edge"]


def read(p):
    j = json.load(open(p[:-4] + ".json"))
    z = np.load(p)
    t = z["t"]
    T = len(t)
    st = np.arange(0, T - W + 1, S) if T >= W else np.zeros(0, int)
    last = st + W - 1
    ft = j.get("fall_time")
    if ft is not None:
        last = last[t[last] <= ft]
    tp = j["terrain_params"]
    ep = {"split": tp.get("range_split"), "mode": j["mode"], "cls": tp.get("class"), "id": j["episode_id"],
          "regions": [r["label"] for r in tp.get("regions", [])], "n": len(last)}
    win = {"t_end": t[last], "label": z["label_terrain"][last].astype(int), "mu": z["prop_mu"][last],
           "contact": z["contact_gt_L"][last] | z["contact_gt_R"][last]}
    strip = None
    if str(tp.get("class", "")).startswith("strip") and j["mode"] == "walk":
        strip = {"meta_td": [(tt, rs) for tt, _, rs in j.get("touchdowns", [])], "t": t, "cL": z["contact_gt_L"].astype(bool), "cR": z["contact_gt_R"].astype(bool),
                 "lab": z["label_terrain"].astype(int)}
    return ep, win, strip


def touchdowns(c):
    out, air = [], TD_AIR
    for i, v in enumerate(c):
        if v and air >= TD_AIR:
            out.append(i)
        air = 0 if v else air + 1
    return out


def f1(y, p, k):
    tp = np.sum((y == k) & (p == k))
    fp = np.sum((y != k) & (p == k))
    fn = np.sum((y == k) & (p != k))
    return 2 * tp / max(2 * tp + fp + fn, 1)


def strip_latency(eps, wins, strips, offs, pred_terrain_by_window, mode_end="first"):
    """mode_end: 'first' / 'second' = which of the 2 consecutive windows gives t (own t0 from contact_gt + frame
    label); 'meta' = t0 and steps from the recorder's touchdown log (per-foot region contact, physics rate)."""
    res = []
    for e, ep in enumerate(eps):
        s = strips[e]
        if s is None or ep["split"] != "test":
            continue
        new = ep["regions"][1]
        tds = sorted([(i, f) for f in ("cL", "cR") for i in touchdowns(s[f])])
        # touchdown on the new surface: the frame label (region touched) is the new class within 2 frames
        on_new = [i for i, _ in tds if np.any(s["lab"][i:i + 3] == new)]
        meta_new = [tt for tt, rs in s["meta_td"] if 1 in rs]
        if mode_end == "meta":
            if not meta_new:
                res.append(None)
                continue
            t0 = min(meta_new)
        else:
            if not on_new:
                res.append(None)
                continue
            t0 = s["t"][on_new[0]]
        te = wins[e]["t_end"]
        pr = pred_terrain_by_window[offs[e]:offs[e] + ep["n"]]
        cand = np.nonzero(te >= t0 - 1e-6)[0]
        det = None
        for k in cand[:-1]:
            if pr[k] == new and pr[k + 1] == new:
                det = te[k + 1] if mode_end == "second" else te[k]
                break
        miss = det is None or det - t0 > 5.0
        t_det = t0 + 5.0 if miss else det
        if mode_end == "meta":
            res.append((ep["cls"], t_det - t0, sum(1 for x in meta_new if t0 - 1e-6 <= x <= t_det + 1e-6), miss))
            continue
        all_td = [s["t"][i] for i, _ in tds]
        steps = sum(1 for x in all_td if t0 - 1e-6 <= x <= t_det + 1e-6 and np.any(s["lab"][int(round(x * 50)) - 1:int(round(x * 50)) + 2] == new))
        res.append((ep["cls"], t_det - t0, steps, miss))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/path/to/fla-terrain/data/v1")
    ap.add_argument("--pred", default="/path/to/fla-terrain/a/results/v1")
    ap.add_argument("--metrics", default="results/v1/metrics.json")
    ap.add_argument("--sets", default="P+N,P+N+S")
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.data, "**", "episode_*.npz"), recursive=True))
    with Pool(32) as pool:
        R = pool.map(read, paths, chunksize=16)
    eps = [r[0] for r in R]
    wins = [r[1] for r in R]
    strips = [r[2] for r in R]
    offs = np.cumsum([0] + [e["n"] for e in eps])
    ep_of = np.repeat(np.arange(len(eps)), [e["n"] for e in eps])
    cat = {k: np.concatenate([w[k] for w in wins]) for k in wins[0]}
    split = np.array([e["split"] for e in eps])[ep_of]
    mode = np.array([e["mode"] for e in eps])[ep_of]
    cls = np.array([str(e["cls"]) for e in eps])[ep_of]
    M = json.load(open(a.metrics))
    rows = []
    print(f"[B14] {len(eps)} episodes, {len(ep_of)} windows; test {np.sum(split == 'test')} (metrics.json windows "
          f"{M['windows']})")
    for s in a.sets.split(","):
        for seed in (0, 1, 2):
            P = np.load(os.path.join(a.pred, f"pred_{s}_s{seed}.npz"))
            idx = P["idx"]
            assert np.all(split[idx] == "test") and len(idx) == np.sum(split == "test"), "window mapping mismatch"
            pt = np.full(len(ep_of), -9)
            pt[idx] = P["terrain"]
            pmu = np.full(len(ep_of), np.nan)
            pmu[idx] = P["mu"]
            sel = np.zeros(len(ep_of), bool)
            sel[idx] = True
            walk = sel & (mode == "walk")
            y = cat["label"]
            m1 = walk & (y >= 0)
            macro = np.mean([f1(y[m1], pt[m1], k) for k in range(6)])
            g = walk & np.isin(cls, ["hard_grip", "hard_slip"]) & np.isin(y, [0, 1])
            g1 = np.mean([f1(y[g], pt[g], k) for k in (0, 1)])
            m3 = walk & cat["contact"] & np.isfinite(cat["mu"])
            mae = np.mean(np.abs(pmu[m3] - cat["mu"][m3]))
            m3b = m3 & (y >= 0)
            mae_b = np.mean(np.abs(pmu[m3b] - cat["mu"][m3b]))
            lat = {m: strip_latency(eps, wins, strips, offs, pt, m) for m in ("first", "second", "meta")}
            A = M["runs"][s][str(seed)]
            mine = {"terrain_f1": macro, "g1_grip_slip_f1": g1, "mu_mae": mae}
            theirs = {"terrain_f1": A["headline_walk"]["terrain_f1"], "g1_grip_slip_f1": A["g1_grip_slip_f1"],
                      "mu_mae": A["headline_walk"]["mu_mae"]}
            for mode_end, L in lat.items():
                ok = [x for x in L if x is not None]
                mine[f"strip_steps_{mode_end}"] = float(np.median([x[2] for x in ok]))
                mine[f"strip_s_{mode_end}"] = float(np.median([x[1] for x in ok]))
                mine[f"strip_n_{mode_end}"] = (len(L), sum(x is None for x in L), float(np.mean([x[3] for x in ok])))
            theirs["strip_steps"] = A["strip"]["median_latency_steps"]
            theirs["strip_s"] = A["strip"]["median_latency_s"]
            theirs["strip_n"] = (A["strip"]["n"], A["strip"]["never_touched"], A["strip"]["miss_rate"])
            rows.append((s, seed, mine, theirs, mae_b, int(m1.sum()), int(g.sum()), int(m3.sum())))
    print(f"{'set':6s} sd | {'macroF1 mine/A/diff':>22s} | {'G1-1 F1 mine/A/diff':>22s} | {'muMAE mine/A/diff':>22s} | "
          f"{'strip steps mine(1st,2nd)/A':>26s} | {'strip s mine(1st,2nd)/A':>24s}")
    for s, seed, mi, th, mae_b, n1, n2, n3 in rows:
        d = lambda k: abs(mi[k] - th[k])  # noqa: E731
        print(f"{s:6s} {seed}  | {mi['terrain_f1']:.4f} {th['terrain_f1']:.4f} {d('terrain_f1'):.4f} | "
              f"{mi['g1_grip_slip_f1']:.4f} {th['g1_grip_slip_f1']:.4f} {d('g1_grip_slip_f1'):.4f} | "
              f"{mi['mu_mae']:.4f} {th['mu_mae']:.4f} {d('mu_mae'):.4f} | "
              f"{mi['strip_steps_first']:.1f},{mi['strip_steps_second']:.1f} / {th['strip_steps']:.1f} | "
              f"{mi['strip_s_first']:.2f},{mi['strip_s_second']:.2f} / {th['strip_s']:.2f} | meta-t0: "
              f"{mi['strip_steps_meta']:.1f} st {mi['strip_s_meta']:.3f} s miss {mi['strip_n_meta'][2]:.3f} (A miss {th['strip_n'][2]:.3f})")
    print("counts (mine): macro windows %d, G1-1 windows %d (A %d), muMAE windows %d; strips (n, never touched, miss "
          "rate) mine %s / A %s" % (rows[0][5], rows[0][6], M["runs"]["P+N"]["0"]["g1_grip_slip_n"], rows[0][7],
                                     rows[0][2]["strip_n_first"], rows[0][3]["strip_n"]))
    print("muMAE excluding label -1 windows (mine): " + ", ".join(f"{s} s{seed} {mb:.4f}" for s, seed, _, _, mb, *_ in rows))
    # gate quantities from my numbers
    def mean_of(s, k):
        return np.mean([mi[k] for ss, _, mi, *_ in rows if ss == s])
    print(f"G1 from my numbers: crit1 dF1 = {100 * (mean_of('P+N+S', 'g1_grip_slip_f1') - mean_of('P+N', 'g1_grip_slip_f1')):+.2f} "
          f"points (need >= +5); crit2 d steps = {mean_of('P+N', 'strip_steps_first') - mean_of('P+N+S', 'strip_steps_first'):+.2f} "
          f"(need >= 1.0); crit3 MAE ratio = {mean_of('P+N+S', 'mu_mae') / mean_of('P+N', 'mu_mae'):.3f} (need <= 0.80)")
    json.dump([{"set": s, "seed": seed, "mine": {k: (list(v) if isinstance(v, tuple) else v) for k, v in mi.items()},
                "A": {k: (list(v) if isinstance(v, tuple) else v) for k, v in th.items()}} for s, seed, mi, th, *_ in rows],
              open("results/v1/qa_recompute_B14.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
