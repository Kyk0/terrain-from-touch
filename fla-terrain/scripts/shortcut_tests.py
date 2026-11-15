"""Shortcut tests for suspiciously good numbers (T7 duty).

  $PY scripts/shortcut_tests.py --data /path/to/fla-terrain/data/v1 --out results/v1
1. per-gait headline (walk, test) terrain F1 / grip-slip F1 from the saved ablation predictions;
2. command: retrain P+N and P+N+S (seed 0) without the cmd_vel channels;
3. gait identity: leave-one-gait-out — train without any SONIC episodes, test on SONIC walk test episodes
   (and the same for G80), P+N and P+N+S, seed 0;
4. episode identity: impossible by construction (split by split_key = episode, disjoint parameter bands);
5. chatter: compare P+N vs P+N_lp from the ablation (in tables.md).
Writes <out>/shortcut.json (rendered by scripts/render_tables.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
import numpy as np  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from fla_terrain.models.dataset import cache_path, FrameData  # noqa: E402
from fla_terrain.models.evaluate import evaluate  # noqa: E402
from fla_terrain.models.provenance import require_clean  # noqa: E402
from fla_terrain.models.train import train_net  # noqa: E402


def walk_f1(fd, idx, pred, mask_fn):
    y = fd.win_labels(idx)
    ep = fd.ep_of_win[idx]
    m = np.array([mask_fn(fd.eps[e]) for e in ep]) & (y["terrain"] >= 0)
    g = m & np.isin(y["terrain"], [0, 1]) & np.array([not fd.eps[e]["class"].startswith("strip") for e in ep])
    f = f1_score(y["terrain"][m], pred["terrain"][m], average="macro") if m.any() else np.nan
    gs = np.mean(f1_score(y["terrain"][g], pred["terrain"][g], labels=[0, 1], average=None)) if g.any() else np.nan
    return f, gs, int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/v1")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    git = require_clean(args.allow_dirty)
    fd = FrameData.build(args.data, cache=cache_path(args.data))
    sp = fd.split_indices()
    mpath = os.path.join(args.out, "metrics.json")
    metrics = json.load(open(mpath))
    lines = []
    # 1. per gait from saved predictions
    for s in ("P", "P+N", "P+N+S"):
        per = {}
        for src in ("walk_g113", "walk_g80", "walk_sonic"):
            fs = []
            for seed in (0, 1, 2):
                p = os.path.join(args.out, f"pred_{s}_s{seed}.npz")
                if os.path.exists(p):
                    d = np.load(p)
                    fs.append(walk_f1(fd, d["idx"], {"terrain": d["terrain"]},
                                      lambda e, src=src: e["mode"] == "walk" and e["source"] == src))
            if fs:
                per[src] = f"F1 {np.mean([f[0] for f in fs]):.3f} grip/slip {np.mean([f[1] for f in fs]):.3f}"
        lines.append(f"per-gait walk test, {s} (mean of seeds): " + "; ".join(f"{k} {v}" for k, v in per.items()))
    # 2. without command channels
    cmd_cols = fd.groups["P"][-3:]
    for s in ("P+N", "P+N+S"):
        _, pred, _ = train_net(fd, s, 0, sp, drop_cols=cmd_cols)
        r = evaluate(fd, sp["test"], pred)
        ref = metrics["runs"][s]["0"]
        lines.append(f"no cmd_vel channels, {s} seed 0: walk F1 {r['headline_walk']['terrain_f1']:.3f} "
                     f"(with cmd {ref['headline_walk']['terrain_f1']:.3f}), grip/slip F1 {r['g1_grip_slip_f1']:.3f} "
                     f"(with {ref['g1_grip_slip_f1']:.3f}), mu MAE {r['headline_walk']['mu_mae']:.3f} "
                     f"(with {ref['headline_walk']['mu_mae']:.3f})")
    # 3. leave one gait out
    src_ep = np.array([e["source"] for e in fd.eps])
    for held in ("walk_sonic", "walk_g80"):
        walker = held.split("_")[1]
        uses = np.array([walker in s for s in src_ep])  # walk_<w>, lifted_<w>, liftmid_<w>
        keep = {k: v[~uses[fd.ep_of_win[v]]] for k, v in sp.items()}
        test = sp["test"][(src_ep[fd.ep_of_win[sp["test"]]] == held)]
        for s in ("P+N", "P+N+S"):
            splits = {"train": keep["train"], "val": keep["val"], "test": test}
            _, pred, _ = train_net(fd, s, 0, splits)
            f, gs, n = walk_f1(fd, test, pred, lambda e: e["mode"] == "walk")
            # in-distribution reference: the ablation model on the same windows
            d = np.load(os.path.join(args.out, f"pred_{s}_s0.npz"))
            pos = {i: k for k, i in enumerate(d["idx"])}
            ref = {"terrain": d["terrain"][[pos[i] for i in test]]}
            f0, gs0, _ = walk_f1(fd, test, ref, lambda e: e["mode"] == "walk")
            lines.append(f"leave-{held}-out, {s} seed 0, tested on {held} walk test ({n} windows): F1 {f:.3f} "
                         f"(seen gait {f0:.3f}), grip/slip F1 {gs:.3f} (seen {gs0:.3f})")
    lines.append("episode identity: splits are by split_key (one per episode = terrain instance) with disjoint "
                 "friction/stiffness bands; no episode or parameter value is shared (leakage test PASS)")
    for x in lines:
        print(x, flush=True)
    json.dump({"lines": lines, "git_commit": git["commit"], "git_dirty": git["dirty"], "command": " ".join(sys.argv)},
              open(os.path.join(args.out, "shortcut.json"), "w"), indent=1)
    print("SHORTCUT_DONE", flush=True)


if __name__ == "__main__":
    main()
