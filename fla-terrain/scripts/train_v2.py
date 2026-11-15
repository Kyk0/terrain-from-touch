"""Train and evaluate v2 models on a v0.3 dataset; every model's weights are saved.

  $PY scripts/train_v2.py --data /path/to/fla-terrain/data/v2_pilot --tag pilot \
      --views P+Nreal+Sreal,P+Nreal+Sblk,P+Nreal,P+Nblk+Sblk,P+Nblk,P --seeds 0
Models -> /path/to/fla-terrain/models/<tag>/<view>_s<seed>/ ; metrics -> results/v2/<tag>/metrics.json.
Evaluation: fla_terrain.models.evaluate (v1 D7 code) on the test split, plus per-class support F1.
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

from fla_terrain.models.dataset_v2 import FrameDataV2, cache_path_v2  # noqa: E402
from fla_terrain.models.evaluate import evaluate  # noqa: E402
from fla_terrain.models.provenance import require_clean  # noqa: E402
from fla_terrain.models.v2 import MODELS_ROOT, train_v2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--views", default="P+Nreal+Sreal,P+Nreal+Sblk,P+Nreal,P+Nblk+Sblk,P+Nblk,P")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    git = require_clean(args.allow_dirty)
    out = os.path.join("results", "v2", args.tag)
    os.makedirs(out, exist_ok=True)
    mpath = os.path.join(out, "metrics.json")
    metrics = json.load(open(mpath)) if os.path.exists(mpath) else {"runs": {}}
    t0 = time.time()
    fd = FrameDataV2.build(args.data, cache=cache_path_v2(args.data))
    sp = fd.split_indices()
    metrics.update({"git_commit": git["commit"], "git_dirty": git["dirty"], "data": args.data, "command": " ".join(sys.argv),
                    "windows": {k: int(len(v)) for k, v in sp.items()}, "episodes": len(fd.eps)})
    print(f"data: {len(fd.eps)} episodes, windows {metrics['windows']}, frames {len(fd.V)}, load {time.time()-t0:.0f} s",
          flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for view in args.views.split(","):
        for seed in [int(x) for x in args.seeds.split(",")]:
            name = f"{args.tag}/{view}_s{seed}"
            _, pred, info = train_v2(fd, view, seed, sp, name, args.data, epochs=args.epochs, device=dev)
            r = evaluate(fd, sp["test"], pred)
            r["train_info"] = info
            r["model_dir"] = os.path.join(MODELS_ROOT, name)
            metrics["runs"].setdefault(view, {})[str(seed)] = r
            json.dump(metrics, open(mpath, "w"), indent=1, default=float)
            h = r["headline_walk"]
            pc = h.get("support_f1_per_class", [np.nan] * 3)
            print(f"  {view:14s} s{seed}: F1 {h.get('terrain_f1', np.nan):.3f} grip/slip {r['g1_grip_slip_f1']:.3f} "
                  f"muMAE {h['mu_mae']:.3f} supF1 {h['support_f1']:.3f} (air {pc[0]:.3f} part {pc[1]:.3f} full {pc[2]:.3f}) "
                  f"edge delay {r['edge_delay'].get('walk', {}).get('median_delay_s', np.nan)} "
                  f"miss {r['edge_delay'].get('walk', {}).get('miss_rate', np.nan):.2f}", flush=True)
    print(f"TRAIN_V2_DONE {time.time()-t0:.0f} s -> {mpath}", flush=True)


if __name__ == "__main__":
    main()
