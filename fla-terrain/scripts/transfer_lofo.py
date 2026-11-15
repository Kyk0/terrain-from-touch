"""EXPLORATORY (post hoc, not part of the G1 gate): transfer to an unseen gait.

Leave-one-source-out for each walking source (G113, G80, SONIC) x {P, P+N, P+N+S} x seeds: train and select (val)
without ANY episode driven by that walker (walk, lifted and lifted-mid), evaluate on the held-out source's walk test
windows. Seen-gait reference: the main ablation model of the same input set and seed on the same windows.
Metrics: terrain macro-F1 (classes present), grip/slip F1 (D7 definition), friction MAE (windows with a foot in
contact, label >= 0).
  $PY scripts/transfer_lofo.py --data /path/to/fla-terrain/data/v1 --out results/v1 [--seeds 0,1,2]
Writes <out>/lofo.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from fla_terrain.models.dataset import FrameData, cache_path  # noqa: E402
from fla_terrain.models.provenance import require_clean  # noqa: E402
from fla_terrain.models.train import train_net  # noqa: E402

SOURCES = ("walk_g113", "walk_g80", "walk_sonic")


def scores(fd, idx, pred):
    y = fd.win_labels(idx)
    ep = fd.ep_of_win[idx]
    known = y["terrain"] >= 0
    f = f1_score(y["terrain"][known], pred["terrain"][known], average="macro")
    flat = np.array([not fd.eps[e]["class"].startswith("strip") for e in ep])
    g = known & flat & np.isin(y["terrain"], [0, 1])
    gs = float(np.mean(f1_score(y["terrain"][g], pred["terrain"][g], labels=[0, 1], average=None, zero_division=0)))
    c = known & ((y["sup_L"] > 0) | (y["sup_R"] > 0)) & np.isfinite(y["mu"])
    return {"terrain_f1": float(f), "grip_slip_f1": gs, "mu_mae": float(np.abs(pred["mu"][c] - y["mu"][c]).mean()),
            "n": int(len(idx))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/v1")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--sets", default="P,P+N,P+N+S")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    git = require_clean(args.allow_dirty)
    fd = FrameData.build(args.data, cache=cache_path(args.data))
    sp = fd.split_indices()
    src_ep = np.array([e["source"] for e in fd.eps])
    mode_ep = np.array([e["mode"] for e in fd.eps])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"git_commit": git["commit"], "git_dirty": git["dirty"], "command": " ".join(sys.argv), "runs": {}}
    for held in SOURCES:
        walker = held.split("_")[1]
        uses = np.array([s.endswith("_" + walker) for s in src_ep])  # walk_w, lifted_w, liftmid_w
        tr = {k: v[~uses[fd.ep_of_win[v]]] for k, v in sp.items() if k != "test"}
        te = sp["test"][(src_ep[fd.ep_of_win[sp["test"]]] == held) & (mode_ep[fd.ep_of_win[sp["test"]]] == "walk")]
        for s in args.sets.split(","):
            for seed in [int(x) for x in args.seeds.split(",")]:
                _, pred, info = train_net(fd, s, seed, {"train": tr["train"], "val": tr["val"], "test": te}, device=dev)
                r = {"unseen": scores(fd, te, pred), "val_loss": info["val_loss"]}
                p = os.path.join(args.out, f"pred_{s}_s{seed}.npz")
                if os.path.exists(p):  # seen-gait reference from the main ablation
                    d = np.load(p)
                    pos = np.full(len(fd.starts), -1)
                    pos[d["idx"]] = np.arange(len(d["idx"]))
                    k = pos[te]
                    r["seen"] = scores(fd, te, {key: d[key][k] for key in ("terrain", "mu")})
                out["runs"].setdefault(held, {}).setdefault(s, {})[str(seed)] = r
                print(f"  held-out {held:10s} {s:6s} s{seed}: unseen F1 {r['unseen']['terrain_f1']:.3f} grip/slip "
                      f"{r['unseen']['grip_slip_f1']:.3f} muMAE {r['unseen']['mu_mae']:.3f} | seen F1 "
                      f"{r.get('seen', {}).get('terrain_f1', float('nan')):.3f}", flush=True)
                json.dump(out, open(os.path.join(args.out, "lofo.json"), "w"), indent=1, default=float)
    print("LOFO_DONE", flush=True)


if __name__ == "__main__":
    main()
