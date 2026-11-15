"""Ablation runner (T7): TerrainNet x seeds x input sets, evaluated on test per D7; writes metrics.json + tables.md.

  $PY scripts/run_ablation.py --data /path/to/fla-terrain/data/v1 --out results/v1 [--seeds 0,1,2]
Resumable: runs already present in metrics.json are skipped (unless --force). The frame cache lives next to the
data (<data>_cache.npz).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
import numpy as np  # noqa: E402
import torch  # noqa: E402

from fla_terrain.models.dataset import INPUT_SETS_ALL, INPUT_SETS_EXPLORATORY, FrameData, cache_path, leakage_report  # noqa: E402
from fla_terrain.models.provenance import require_clean  # noqa: E402
from fla_terrain.models.evaluate import evaluate  # noqa: E402
from fla_terrain.models.train import train_net  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/v1")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--sets", default=",".join(INPUT_SETS_ALL + INPUT_SETS_EXPLORATORY))
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    git = require_clean(args.allow_dirty)
    os.makedirs(args.out, exist_ok=True)
    mpath = os.path.join(args.out, "metrics.json")
    metrics = json.load(open(mpath)) if os.path.exists(mpath) and not args.force else {"runs": {}}
    t0 = time.time()
    fd = FrameData.build(args.data, cache=cache_path(args.data))
    leak = leakage_report(fd)
    print("\n".join(leak), flush=True)
    if not leak[-1].endswith("PASS"):
        sys.exit("leakage test failed")
    splits = fd.split_indices()
    metrics.update({"git_commit": git["commit"], "git_dirty": git["dirty"], "dataset": args.data, "command": " ".join(sys.argv),
                    "leakage": leak, "windows": {k: int(len(v)) for k, v in splits.items()},
                    "episodes": len(fd.eps)})
    print(f"data: {len(fd.eps)} episodes, windows {metrics['windows']}, load {time.time()-t0:.0f} s", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for s in args.sets.split(","):
        for seed in [int(x) for x in args.seeds.split(",")]:
            if str(seed) in metrics["runs"].get(s, {}):
                continue
            _, pred, info = train_net(fd, s, seed, splits, epochs=args.epochs, device=dev)
            res = evaluate(fd, splits["test"], pred)
            res["train_info"] = info
            metrics["runs"].setdefault(s, {})[str(seed)] = res
            np.savez_compressed(os.path.join(args.out, f"pred_{s}_s{seed}.npz"), idx=splits["test"],
                                **{k: v for k, v in pred.items()})
            json.dump(metrics, open(mpath, "w"), indent=1, default=float)
            h = res["headline_walk"]
            print(f"  {s:12s} s{seed}: walk F1 {h.get('terrain_f1', float('nan')):.3f} muMAE {h['mu_mae']:.3f} "
                  f"supF1 {h['support_f1']:.3f} | G1-1 F1 {res['g1_grip_slip_f1']:.3f} | strip steps "
                  f"{res['strip'].get('median_latency_steps', float('nan'))}", flush=True)
    print(f"ABLATION_DONE {time.time()-t0:.0f} s -> {mpath}", flush=True)


if __name__ == "__main__":
    main()
