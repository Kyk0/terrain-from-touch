#!/usr/bin/env python
"""B17 item 1: independent recomputation of the v2 test metrics (M1, M2, M2_partial, M3, M5 short/long miss) from raw
episodes of data/v2 and the stored test predictions (models/v2_src3/main/<view>_s0/test_pred.npz).

Does NOT import fla_terrain.models (evaluate, v2eval, dataset_v2). Taken from A's dataset_v2.py only to decode `idx`:
episode order = sorted(glob('<data>/**/episode_*.npz')), windows start 0, 10, ... length 50 (start + 50 <= T), windows
whose last frame t > fall_time dropped, concatenated; idx = positions in that list; split = terrain_params.range_split.
Metric definitions from DECISIONS D7 / D14 (my reading):
  M1   terrain macro-F1 over the classes present, mode walk, test windows with label >= 0
  M2   support-state macro-F1 over (airborne, partial, full) per foot, mean of L and R, walk test windows;
       M2_partial = F1 of the partial class, mean of L and R
  M3   friction MAE, walk test windows with label >= 0 and >= 1 foot with GT support > 0 (contact), finite mu
  M5   edge episodes (class edge, mode walk): every stance (run of GT support > 0 frames) of each foot whose GT support
       is partial within its first 5 frames (0.1 s); t_p = first partial frame; duration = stance end - t_p; detected if
       a window end in [t_p, stance end] predicts partial for that foot; miss rate split at duration 0.3 s.
  python scripts/qa_v2_recompute.py [--views ...] [--seed 0]
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np

W, S = 50, 10
DATA = "/path/to/fla-terrain/data/v2"
MODELS = "/path/to/fla-terrain/models/v2_src3/main"
JOBS = "/path/to/fla-terrain/a/results/v2/src3/jobs/main"
VIEWS = ["P", "P+Nreal+Sreal", "P+Nreal+Sblk", "P+Nreal", "P+Nblk+Sblk", "P+Nblk", "Nreal+Sblk", "Nreal"]


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
    ep = {"split": tp.get("range_split"), "mode": j["mode"], "cls": str(tp.get("class")), "n": len(last)}
    sL, sR = z["label_support_L"].astype(int), z["label_support_R"].astype(int)
    win = {"label": z["label_terrain"][last].astype(int), "mu": z["prop_mu"][last].astype(float),
           "sL": sL[last], "sR": sR[last], "t_end": t[last]}
    full = None
    if ep["cls"] == "edge" and ep["mode"] == "walk" and ep["split"] == "test":
        full = {"t": t, "sL": sL, "sR": sR}
    return ep, win, full


def f1(y, p, k):
    tp = np.sum((y == k) & (p == k)); fp = np.sum((y != k) & (p == k)); fn = np.sum((y == k) & (p != k))
    return 2 * tp / max(2 * tp + fp + fn, 1)


def m5(eps, fulls, offs, psup):
    ev = {"short": [], "long": []}
    for e, ep in enumerate(eps):
        f = fulls[e]
        if f is None:
            continue
        te = WIN_T[offs[e]:offs[e] + ep["n"]]
        for side, key in ((0, "sL"), (1, "sR")):
            sup = f[key]
            ps = psup[side][offs[e]:offs[e] + ep["n"]]
            on = sup > 0
            k = 0
            while k < len(on):
                if not on[k]:
                    k += 1
                    continue
                j = k
                while j < len(on) and on[j]:
                    j += 1
                first = np.nonzero(sup[k:min(j, k + 5)] == 1)[0]
                if first.size:
                    tp_, ts = f["t"][k + first[0]], f["t"][j - 1]
                    hit = np.any((te >= tp_ - 1e-6) & (te <= ts + 1e-6) & (ps == 1))
                    ev["long" if ts - tp_ >= 0.3 else "short"].append(not hit)
                k = j
    return {g: (float(np.mean(v)) if v else float("nan"), len(v)) for g, v in ev.items()}


def main():
    global WIN_T
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(DATA, "**", "episode_*.npz"), recursive=True))
    with Pool(32) as pool:
        R = pool.map(read, paths, chunksize=16)
    eps, wins, fulls = [r[0] for r in R], [r[1] for r in R], [r[2] for r in R]
    offs = np.cumsum([0] + [e["n"] for e in eps])
    ep_of = np.repeat(np.arange(len(eps)), [e["n"] for e in eps])
    cat = {k: np.concatenate([w[k] for w in wins]) for k in wins[0]}
    WIN_T = cat["t_end"]
    split = np.array([e["split"] for e in eps])[ep_of]
    mode = np.array([e["mode"] for e in eps])[ep_of]
    rows = []
    for v in a.views.split(","):
        P = np.load(os.path.join(MODELS, f"{v}_s{a.seed}", "test_pred.npz"))
        idx = P["idx"]
        assert np.all(split[idx] == "test") and len(idx) == np.sum(split == "test"), f"{v}: window mapping mismatch"
        full = lambda key, fill=-9: (lambda arr: (arr.__setitem__(idx, P[key]) or arr))(np.full(len(ep_of), fill, dtype=float))  # noqa: E731
        pt, pL, pR, pmu = full("terrain"), full("sup_L"), full("sup_R"), full("mu", np.nan)
        sel = np.zeros(len(ep_of), bool)
        sel[idx] = True
        walk = sel & (mode == "walk")
        y = cat["label"]
        m1m = walk & (y >= 0)
        present = sorted(set(y[m1m].tolist()))
        M1 = np.mean([f1(y[m1m], pt[m1m], k) for k in present])
        M2 = np.mean([np.mean([f1(cat[s][walk], pp[walk], k) for k in (0, 1, 2)]) for s, pp in (("sL", pL), ("sR", pR))])
        M2p = np.mean([f1(cat[s][walk], pp[walk], 1) for s, pp in (("sL", pL), ("sR", pR))])
        c = walk & (y >= 0) & ((cat["sL"] > 0) | (cat["sR"] > 0)) & np.isfinite(cat["mu"])
        M3 = np.mean(np.abs(pmu[c] - cat["mu"][c]))
        e5 = m5(eps, fulls, offs, (pL, pR))
        A = json.load(open(os.path.join(JOBS, f"{v}_s{a.seed}.json")))["test"]
        mine = {"M1": M1, "M2": M2, "M2_partial": M2p, "M3": M3, "M5_short_miss": e5["short"][0], "M5_long_miss": e5["long"][0]}
        rows.append({"view": v, "mine": mine, "A": {k: A.get(k) for k in mine}, "M5_n_mine": {g: e5[g][1] for g in e5},
                     "M5_n_A": A.get("M5_n")})
    print(f"[B17-1] {len(eps)} episodes, {len(ep_of)} windows, test {np.sum(split == 'test')}; seed {a.seed}; mine / A / |diff|")
    for r in rows:
        print(f"{r['view']:15s} " + "  ".join(f"{k} {r['mine'][k]:.4f}/{r['A'][k]:.4f}/{abs(r['mine'][k] - r['A'][k]):.4f}"
                                             for k in r["mine"]) + f"  M5 n mine {r['M5_n_mine']} A {r['M5_n_A']}")
    os.makedirs("results/v2/qa_b", exist_ok=True)
    json.dump(rows, open("results/v2/qa_b/recompute_seed%d.json" % a.seed, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
