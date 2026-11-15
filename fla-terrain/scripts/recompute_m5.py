"""A11: recompute M5 (edge events) in every job JSON of a v2 run with the whole-frame split (>= 15 frames = long,
D14 '0.3 s or longer'), from stored predictions; no retraining. Sensitivity test sets whose predictions were not
stored (shifted, pads, g1, nosat for the nominal-trained models) are re-predicted from the saved weights.
The old float-time values are kept under 'M5_float_old'.
  $PY scripts/recompute_m5.py --out results/v2/src3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
import numpy as np  # noqa: E402

from fla_terrain.models.dataset_v2 import FrameDataV2, cache_path_v2  # noqa: E402
from fla_terrain.models.provenance import git_state  # noqa: E402
from fla_terrain.models.v2eval import edge_events  # noqa: E402

DATA = "/path/to/fla-terrain/data"
KEYS = ("M5_long_miss", "M5_long_delay_s", "M5_short_miss", "M5_short_delay_s", "M5_n")
_FD = {}


def fd_of(p):
    if p not in _FD:
        _FD[p] = FrameDataV2.build(p, cache=cache_path_v2(p), workers=16)
    return _FD[p]


def update(block, fd, idx, pred, git):
    m5 = edge_events(fd, idx, pred)
    if "M5_float_old" not in block:
        block["M5_float_old"] = {k: block.get(k) for k in KEYS}
    block.update({"M5_long_miss": m5["long"]["miss_rate"], "M5_long_delay_s": m5["long"]["median_delay_s"],
                  "M5_short_miss": m5["short"]["miss_rate"], "M5_short_delay_s": m5["short"]["median_delay_s"],
                  "M5_n": {g: m5[g]["n"] for g in m5}, "M5_rule": "long = >= 15 frames at 50 Hz (A11)",
                  "M5_recomputed_at": git})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/v2/src3")
    args = ap.parse_args()
    git = git_state()["commit"]
    from fla_terrain.models.v2 import predict_saved
    jobs = sorted(glob.glob(f"{args.out}/jobs/*/*.json"))
    for k, p in enumerate(jobs):
        r = json.load(open(p))
        kind = r["job"]["kind"]
        tp = np.load(os.path.join(r["model_dir"], "test_pred.npz"))
        if kind in ("main", "logo"):
            fd = fd_of(f"{DATA}/v2")
            update(r["test"], fd, tp["idx"], {"sup_L": tp["sup_L"], "sup_R": tp["sup_R"]}, git)
        elif kind == "sens_nosat":
            fd = fd_of(f"{DATA}/v2_sens/nosat")
            update(r["tests"]["nosat"], fd, tp["idx"], {"sup_L": tp["sup_L"], "sup_R": tp["sup_R"]}, git)
        else:  # sens (nominal-trained): nominal from stored predictions, other sets from the saved weights
            for st, blk in r["tests"].items():
                fd = fd_of(f"{DATA}/v2_sens/{st}")
                if st == "nominal":
                    idx, pred = tp["idx"], {"sup_L": tp["sup_L"], "sup_R": tp["sup_R"]}
                else:
                    srcs = r["job"]["sources"]
                    ok = np.array([e["source"] in srcs or e["source"] in ("stand", "rig") or
                                   (e["source"].startswith(("lifted_", "liftmid_")) and
                                    e["source"].split("_", 1)[1] in {s.split("_", 1)[1] for s in srcs}) for e in fd.eps])
                    idx = fd.split_indices()["test"]
                    idx = idx[ok[fd.ep_of_win[idx]]]
                    pred = predict_saved(r["model_dir"], fd, idx, device="cuda")
                update(blk, fd, idx, pred, git)
        json.dump(r, open(p, "w"), indent=1, default=float)
        if k % 20 == 0:
            print(f"  {k + 1}/{len(jobs)} {p}", flush=True)
    print(f"M5_RECOMPUTED {len(jobs)} job files", flush=True)


if __name__ == "__main__":
    main()
