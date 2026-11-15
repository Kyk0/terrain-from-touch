"""EXPLORATORY (post hoc, not part of the G1 gate): where is friction observable? From saved predictions only.

Subset: test windows, mode walk, label >= 0, at least one foot in contact, finite mu (the D7 criterion-3 subset).
Friction MAE for P, P+N, P+N+S (per seed), split by
  - true terrain class;
  - hard_slip mu bin: test_out 0.08-0.12, test_in 0.285-0.295 (true mu of the window);
  - motion source;
  - saturation: window contains >= 1 stance frame (contact and total Fz > 20 N) with |total shear| / Fz >= 0.95 mu
    on either foot (the independent evaluation's definition, scripts/walk_qa_friction.py), from the sensor (noisy) zone sums.
Reference: MAE of predicting the train-set mean mu of the model's predicted class (and of the true class).
  $PY scripts/analyze_friction.py --data /path/to/fla-terrain/data/v1 --out results/v1
Writes <out>/analysis.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
import numpy as np  # noqa: E402

from fla_terrain.models.dataset import W, FrameData, cache_path  # noqa: E402
from fla_terrain.models.provenance import require_clean  # noqa: E402
from fla_terrain.schema import TERRAIN_LABELS  # noqa: E402

FZ_MIN, SAT = 20.0, 0.95


def saturated_windows(fd):
    ntot, stot = fd.X[:, fd.groups["Ntot"]], fd.X[:, fd.groups["Stot"]]
    mu = fd.lab["mu"]
    sat = np.zeros(len(fd.X), bool)
    for i, k in enumerate(("sup_L", "sup_R")):
        fz = ntot[:, i]
        sh = np.hypot(stot[:, 2 * i], stot[:, 2 * i + 1])
        st = (fd.lab[k] > 0) & (fz > FZ_MIN)
        with np.errstate(invalid="ignore", divide="ignore"):
            sat |= st & np.isfinite(mu) & (sh / np.maximum(fz, 1e-6) >= SAT * mu)
    cs = np.concatenate([[0], np.cumsum(sat)])
    return (cs[fd.starts + W] - cs[fd.starts]) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/v1")
    ap.add_argument("--sets", default="P,P+N,P+N+S")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    git = require_clean(args.allow_dirty)
    fd = FrameData.build(args.data, cache=cache_path(args.data))
    sp = fd.split_indices()
    satw = saturated_windows(fd)
    # train-set class means of mu (same subset definition on train windows)
    ytr = fd.win_labels(sp["train"])
    ctr = (ytr["terrain"] >= 0) & ((ytr["sup_L"] > 0) | (ytr["sup_R"] > 0)) & np.isfinite(ytr["mu"])
    cmean = {c: float(ytr["mu"][ctr & (ytr["terrain"] == c)].mean()) for c in range(6)}
    res = {"git_commit": git["commit"], "git_dirty": git["dirty"], "command": " ".join(sys.argv),
           "train_class_mean_mu": cmean, "sets": {}}
    for s in args.sets.split(","):
        per_seed = []
        for seed in (0, 1, 2):
            p = os.path.join(args.out, f"pred_{s}_s{seed}.npz")
            if not os.path.exists(p):
                continue
            d = np.load(p)
            idx = d["idx"]
            y = fd.win_labels(idx)
            ep = fd.ep_of_win[idx]
            mode = np.array([fd.eps[e]["mode"] for e in ep])
            src = np.array([fd.eps[e]["source"] for e in ep])
            m = (mode == "walk") & (y["terrain"] >= 0) & ((y["sup_L"] > 0) | (y["sup_R"] > 0)) & np.isfinite(y["mu"])
            err = np.abs(d["mu"] - y["mu"])
            ref_pred = np.abs(np.array([cmean.get(int(c), np.nan) for c in d["terrain"]]) - y["mu"])
            ref_true = np.abs(np.array([cmean.get(int(c), np.nan) for c in y["terrain"]]) - y["mu"])
            r = {}

            def put(name, mm):
                r[name] = {"mae": float(err[mm].mean()) if mm.any() else float("nan"), "n": int(mm.sum()),
                           "ref_pred_class_mean": float(ref_pred[mm].mean()) if mm.any() else float("nan"),
                           "ref_true_class_mean": float(ref_true[mm].mean()) if mm.any() else float("nan")}
            put("all", m)
            for c in range(6):
                put(f"class:{TERRAIN_LABELS[c]}", m & (y["terrain"] == c))
            put("hard_slip:test_out 0.08-0.12", m & (y["terrain"] == 1) & (y["mu"] >= 0.08) & (y["mu"] <= 0.12))
            put("hard_slip:test_in 0.285-0.295", m & (y["terrain"] == 1) & (y["mu"] >= 0.285) & (y["mu"] <= 0.295))
            for so in ("walk_g113", "walk_g80", "walk_sonic"):
                put(f"source:{so}", m & (src == so))
            put("saturated", m & satw[idx])
            put("not saturated", m & ~satw[idx])
            put("hard_slip saturated", m & satw[idx] & (y["terrain"] == 1))
            put("hard_slip not saturated", m & ~satw[idx] & (y["terrain"] == 1))
            per_seed.append(r)
        res["sets"][s] = per_seed
        a = per_seed[0]
        print(f"{s:6s} " + "; ".join(f"{k} {np.mean([q[k]['mae'] for q in per_seed]):.3f}" for k in
                                     ("all", "class:hard_slip", "saturated", "not saturated")), flush=True)
        print(f"       reference (train mean of predicted class) all {a['all']['ref_pred_class_mean']:.3f}; "
              f"saturated window fraction {a['saturated']['n'] / max(a['all']['n'], 1):.3f}", flush=True)
    json.dump(res, open(os.path.join(args.out, "analysis.json"), "w"), indent=1, default=float)
    print("ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
