#!/usr/bin/env python
"""Fit the rule-based detector's thresholds on the TRAIN split and score val / test (walk mode).

  python scripts/rule_detector_fit.py --data data/v2_pilot/h5 data/v1 --out results/v2/rule_detector
  (final: --data data/v2 once it exists; one command)

Kinematic signals are computed once per episode (cached in <out>/kin_*.npz). Fit order: contact (H_C, V_C) ->
T_PART -> decision list in order (T_SLIP, T_SLOPE, T_EDGE + T_PFRAC, T_ROUGH), each a 1-D grid search maximising the
F1 of its own class against the rest on train windows not already claimed by an earlier rule. Scores: terrain
macro-F1 over the classes present (as fla_terrain.models.evaluate._block), support F1 = macro over (airborne,
partial, full) per foot, mean of L and R; per dataset and split. Courses (data/v2_course) are never read.
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from walking import rule_detector as R  # noqa: E402

_K = {}


def kin_episode(path):
    if "k" not in _K:
        _K["k"] = R.Kinematics()
    j = json.load(open(path))
    if j["mode"] != "walk":
        return None
    z = np.load(path[:-5] + ".npz")
    arr = {k: z[k] for k in z.files if k in ("t", "joint_pos", "joint_vel", "imu_gyro", "imu_gravity", "cmd_vel",
                                            "joint_target", "joint_torque", "label_terrain", "label_support_L",
                                            "label_support_R")}
    zz, v, tilt = _K["k"].episode(arr["joint_pos"], arr["joint_vel"], arr["imu_gyro"], arr["imu_gravity"])
    tp = j["terrain_params"]
    return {"id": j["episode_id"], "split": tp.get("range_split"), "cls": tp.get("class"), "fall": j.get("fall_time"),
            "jn": j["joint_names"], "arr": arr, "z": zz, "v": v, "tilt": tilt, "data": os.path.dirname(path)}


def signals(e, th):
    """frame signals from cached kinematics (mirror of rule_detector.frame_signals)."""
    z, v, tilt, a = e["z"], e["v"], e["tilt"], e["arr"]
    low = z.min(1, keepdims=True)
    c = (z - low < th["H_C"]) & (np.abs(v[:, :, 2]) < th["V_C"])
    if "joint_torque" in a:
        ia = [e["jn"].index(f"{s}_ankle_pitch_joint") for s in ("left", "right")]
        c &= np.abs(a["joint_torque"][:, ia]) > th["TAU_C"]
    ds = c.all(1)
    still = (np.abs(v[:, :, 2]) < th["V_C"]).all(1)
    track = None
    if "joint_target" in a:
        il = [i for i, n in enumerate(e["jn"]) if any(k in n for k in R.LEG)]
        track = np.abs(a["joint_target"][:, il] - a["joint_pos"][:, il]).mean(1)
    return {"z": z, "v": v, "tilt": tilt, "contact": c, "track": track,
            "dslip": np.where(ds, np.linalg.norm(v[:, 0, :2] - v[:, 1, :2], axis=1), np.nan),
            "dz": np.where(still, np.abs(z[:, 0] - z[:, 1]), np.nan)}


def windows(eps, th):
    rows = []
    for e in eps:
        sig = signals(e, th)
        a = e["arr"]
        for s in R.window_starts(len(a["t"]), e["fall"], a["t"]):
            f = R.window_features(sig, s)
            last = s + R.W - 1
            rows.append((e, f, int(a["label_terrain"][last]), int(a["label_support_L"][last]),
                         int(a["label_support_R"][last])))
    return rows


def best_threshold(x, y, grid):
    best = (-1, grid[0])
    for g in grid:
        p = x > g
        tp, fp, fn = np.sum(p & y), np.sum(p & ~y), np.sum(~p & y)
        f = 2 * tp / max(2 * tp + fp + fn, 1)
        best = max(best, (f, g))
    return best[1], best[0]


def fit(eps):
    th = dict(R.DEFAULTS)
    # contact: agreement of (not airborne) with label_support, train frames (every 5th)
    zs, vs, ts, lab = [], [], [], []
    has_tau = all("joint_torque" in e["arr"] for e in eps)
    for e in eps:
        lo = e["z"].min(1, keepdims=True)
        zs.append((e["z"] - lo)[::5]); vs.append(np.abs(e["v"][::5, :, 2]))
        if has_tau:
            ia = [e["jn"].index(f"{s}_ankle_pitch_joint") for s in ("left", "right")]
            ts.append(np.abs(e["arr"]["joint_torque"][::5][:, ia]))
        lab.append(np.stack([e["arr"]["label_support_L"], e["arr"]["label_support_R"]], 1)[::5] > 0)
    zs, vs, lab = np.concatenate(zs).ravel(), np.concatenate(vs).ravel(), np.concatenate(lab).ravel()
    ts = np.concatenate(ts).ravel() if has_tau else np.full_like(zs, np.inf)
    best = (-1, None)
    for h in (0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06):
        for vc in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0):
            for tc in ((0.0, 1.0, 2.0, 5.0, 10.0, 20.0) if has_tau else (0.0,)):
                p = (zs < h) & (vs < vc) & (ts > tc)
                bal = 0.5 * (np.mean(p[lab]) + np.mean(~p[~lab]))
                best = max(best, (bal, (h, vc, tc)))
    th["H_C"], th["V_C"], th["TAU_C"] = best[1]
    rep = {"contact_balanced_acc": round(best[0], 3), "torque_rule": has_tau}
    rows = windows(eps, th)
    # T_PART on stance feet at the last frame: partial (1) vs full (2)
    dev, ylab = [], []
    for e, f, t, sl, sr in rows:
        for k, s in ((0, sl), (1, sr)):
            if f["_c"][-1, k] and s > 0:
                dev.append(f["_dev"][-1, k]); ylab.append(s == 1)
    th["T_PART"], rep["f1_partial_train"] = best_threshold(np.array(dev), np.array(ylab), np.linspace(0.01, 0.4, 40))
    y = np.array([t for _, _, t, _, _ in rows])
    present = sorted(set(y[y >= 0].tolist()))
    feats = {k: np.array([f[k] if f[k] is not None else np.nan for _, f, *_ in rows])
             for k in ("slip", "tilt_med", "dz_max", "tilt_std", "track")}
    has_track = np.isfinite(feats["track"]).all()
    pf = np.array([float(np.mean((f["_dev"] > th["T_PART"])[f["_c"]])) if f["_c"].any() else 0.0 for _, f, *_ in rows])
    m = y >= 0

    def macro(th_):
        p = np.zeros(len(y), int)
        r3, r5, r4, r1 = (feats["tilt_std"] > th_["T_ROUGH"]), ((feats["dz_max"] > th_["T_EDGE"]) | (pf > th_["T_PFRAC"])), \
            (feats["tilt_med"] > th_["T_SLOPE"]), (feats["slip"] > th_["T_SLIP"]) | \
            ((feats["track"] > th_["T_TRACK"]) if has_track else False)
        p[r3] = 3
        p[r5] = 5
        p[r4] = 4
        p[r1] = 1
        return f1_score(y[m], p[m], labels=present, average="macro", zero_division=0)

    grids = {"T_SLIP": np.linspace(0.05, 1.0, 39), **({"T_TRACK": np.linspace(0.01, 0.5, 50)} if has_track else {}), "T_SLOPE": np.linspace(0.02, 0.4, 39),
             "T_EDGE": np.linspace(0.02, 0.3, 29), "T_PFRAC": np.linspace(0.05, 1.01, 25),
             "T_ROUGH": np.linspace(0.02, 0.4, 39)}
    for k in grids:            # start with every rule switched off (threshold above its grid)
        th[k] = float(grids[k][-1]) * 10
    for _ in range(2):         # two passes of coordinate search in decision-list order
        for k, g in grids.items():
            best = max((macro({**th, k: float(x)}), float(x)) for x in np.r_[g, th[k]])
            th[k] = best[1]
    rep["train_macro_f1"] = round(float(macro(th)), 3)
    rep["track_rule"] = bool(has_track)
    return th, rep


def score(eps, th):
    yt, pt, ys, ps = [], [], [], []
    for e in eps:
        sig = signals(e, th)
        a = e["arr"]
        for s in R.window_starts(len(a["t"]), e["fall"], a["t"]):
            cls, sup = R.decide(R.window_features(sig, s), th)
            last = s + R.W - 1
            yt.append(int(a["label_terrain"][last])); pt.append(cls)
            ys.append([int(a["label_support_L"][last]), int(a["label_support_R"][last])]); ps.append(sup)
    yt, pt, ys, ps = map(np.array, (yt, pt, ys, ps))
    m = yt >= 0
    present = sorted(set(yt[m].tolist()))
    tf1 = f1_score(yt[m], pt[m], labels=present, average="macro", zero_division=0)
    sf1 = np.mean([f1_score(ys[:, k], ps[:, k], labels=[0, 1, 2], average="macro", zero_division=0) for k in (0, 1)])
    pf1 = np.mean([f1_score(ys[:, k], ps[:, k], labels=[1], average="macro", zero_division=0) for k in (0, 1)])
    rec = {c: float(np.mean(pt[m][yt[m] == c] == c)) for c in present}
    return {"n": int(m.sum()), "terrain_f1": float(tf1), "terrain_acc": float(np.mean(pt[m] == yt[m])),
            "support_f1": float(sf1), "partial_f1": float(pf1), "recall": rec, "classes": present}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", default="results/v2/rule_detector")
    a = ap.parse_args()
    assert not any("course" in d for d in a.data), "courses are test only"
    files = [f for d in a.data for f in sorted(glob.glob(os.path.join(d, "*.json")))]
    with Pool(40) as p:
        eps = [e for e in p.map(kin_episode, files, chunksize=4) if e is not None]
    th, rep = fit([e for e in eps if e["split"] == "train"])
    os.makedirs(a.out, exist_ok=True)
    res = {"thresholds": th, "fit": rep, "data": a.data, "scores": {}}
    print(f"[rule detector] fitted on train ({sum(e['split'] == 'train' for e in eps)} walk episodes of {a.data})")
    print("  thresholds: " + ", ".join(f"{k} {v:.3g}" for k, v in th.items()) + f"\n  fit: {rep}")
    for d in a.data:
        for sp in ("train", "val", "test"):
            sel = [e for e in eps if e["data"] == d.rstrip("/") and e["split"] == sp]
            if not sel:
                continue
            s = score(sel, th)
            res["scores"][f"{d}:{sp}"] = s
            print(f"  {d:22s} {sp:5s} n {s['n']:6d}  terrain F1 {s['terrain_f1']:.3f} acc {s['terrain_acc']:.3f}  "
                  f"support F1 {s['support_f1']:.3f} (partial {s['partial_f1']:.3f})  recall "
                  + " ".join(f"{c}:{r:.2f}" for c, r in s["recall"].items()))
    json.dump(res, open(os.path.join(a.out, "thresholds.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
