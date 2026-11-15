"""T6 classical baseline at scale: hand features + LightGBM for P, P+N, P+N+S; evaluated like the net (D7).

  $PY scripts/train_baseline.py --data /path/to/fla-terrain/data/v1 --out results/v1 [--max-train 150000]
Features (fla_terrain.models.features): proprio stats for P; Fz per zone stats, CoP and CoP velocity, max force
rise rate (touchdown), contact area, loaded fraction for N; |shear|/Fz per zone and per foot for S.
Train windows are subsampled to --max-train (uniformly, seed 0) for speed; val for early stopping; test once.
Prints the leakage test first. Writes <out>/baseline.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

from fla_terrain.models.dataset import cache_path, W, FrameData, leakage_report  # noqa: E402
from fla_terrain.models.evaluate import evaluate  # noqa: E402
from fla_terrain.models.provenance import require_clean  # noqa: E402
from fla_terrain.models.features import window_features  # noqa: E402
from fla_terrain.sim.sole import SoleConfig  # noqa: E402

warnings.filterwarnings("ignore")
PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.5, verbose=-1, n_jobs=32)


def channel_names(fd, input_set):
    J = fd.eps[0]["J"]
    Z = fd.Z
    names = {"P": [f"q{j}" for j in range(J)] + [f"qd{j}" for j in range(J)] + [f"gyro_{a}" for a in "xyz"] +
            [f"grav_{a}" for a in "xyz"] + ["cmd_vx", "cmd_vy", "cmd_wz"],
             "N": [f"Fz_{s}{z}" for s in "LR" for z in range(Z)],
             "S": [f"F{a}_{s}{z}" for s in "LR" for a in "xy" for z in range(Z)]}
    return sum((names[p] for p in input_set.split("+")), [])


def feats(fd, input_set, idx, chunk=20000):
    cols = fd.columns(input_set)
    names = channel_names(fd, input_set)
    zc = SoleConfig().zone_centers()
    out = []
    for i in range(0, len(idx), chunk):
        st = fd.starts[idx[i:i + chunk]]
        X = fd.X[st[:, None] + np.arange(W)][:, :, cols]
        out.append(window_features(X, names, zc)[0])
    return np.concatenate(out)


def fit_predict(kind, Ftr, ytr, Fva, yva, Fte, mtr, mva):
    if mtr.sum() == 0:
        return np.full(len(Fte), np.nan)
    M = (lgb.LGBMClassifier if kind == "clf" else lgb.LGBMRegressor)(random_state=0, **PARAMS)
    M.fit(Ftr[mtr], ytr[mtr], eval_set=[(Fva[mva], yva[mva])], callbacks=[lgb.early_stopping(30, verbose=False)])
    return M.predict(Fte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/v1")
    ap.add_argument("--sets", default="P,P+N,P+N+S")
    ap.add_argument("--max-train", type=int, default=150000)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    git = require_clean(args.allow_dirty)
    t0 = time.time()
    fd = FrameData.build(args.data, cache=cache_path(args.data))
    leak = leakage_report(fd)
    print("\n".join(leak), flush=True)
    if not leak[-1].endswith("PASS"):
        sys.exit("leakage test failed")
    sp = fd.split_indices()
    rng = np.random.default_rng(0)
    tr = np.sort(rng.choice(sp["train"], min(args.max_train, len(sp["train"])), replace=False))
    va, te = sp["val"], sp["test"]
    Y = {k: fd.win_labels(i) for k, i in (("tr", tr), ("va", va), ("te", te))}
    res = {}
    for s in args.sets.split(","):
        F = {k: feats(fd, s, i) for k, i in (("tr", tr), ("va", va), ("te", te))}
        pred = {}
        for name, kind in (("terrain", "clf"), ("mu", "reg"), ("logk", "reg"), ("slope", "reg"), ("sup_L", "clf"),
                           ("sup_R", "clf")):
            known = {k: (Y[k]["terrain"] >= 0) & (np.isfinite(Y[k][name]) if kind == "reg" else True) for k in Y}
            if name.startswith("sup"):
                known = {k: np.ones(len(Y[k][name]), bool) for k in Y}
            pred[name] = fit_predict(kind, F["tr"], Y["tr"][name], F["va"], Y["va"][name], F["te"], known["tr"],
                                     known["va"])
        for k in ("terrain", "sup_L", "sup_R"):
            pred[k] = pred[k].astype(int)
        r = evaluate(fd, te, pred)
        r["n_features"] = int(F["tr"].shape[1])
        res[s] = r
        h = r["headline_walk"]
        print(f"  {s:8s} features {F['tr'].shape[1]:4d}: walk F1 {h['terrain_f1']:.3f} muMAE {h['mu_mae']:.3f} "
              f"logK {h['logk_mae']:.3f} supF1 {h['support_f1']:.3f} | G1-1 F1 {r['g1_grip_slip_f1']:.3f} | "
              f"{time.time()-t0:.0f} s", flush=True)
    os.makedirs(args.out, exist_ok=True)
    res["_provenance"] = {"git_commit": git["commit"], "git_dirty": git["dirty"], "command": " ".join(sys.argv)}
    json.dump(res, open(os.path.join(args.out, "baseline.json"), "w"), indent=1, default=float)
    print(f"BASELINE_DONE {time.time()-t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
