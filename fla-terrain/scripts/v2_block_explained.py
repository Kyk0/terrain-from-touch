"""How much of the realistic point map N_real do the four block sums N_blk explain? (A7 pilot question)

Per foot and frame with foot load > 20 N (recorded maps, i.e. after the sensor model; middle part included in N_real):
  R2_uniform : predict every point as its block sum spread uniformly over the block's points (middle points = 0)
  R2_linear  : best linear predictor of the 152-point map from the 4 block sums (+ intercept), fitted on train frames
R2 = 1 - SSE / SST with SST around the per-point mean over the evaluated frames; evaluated on test frames, per terrain
class. Also the fraction of the map's total load in the middle part.
  $PY scripts/v2_block_explained.py --data /path/to/fla-terrain/data/v2_pilot/h5
"""
from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
import numpy as np  # noqa: E402

from fla_terrain.models.dataset_v2 import FrameDataV2, cache_path_v2  # noqa: E402
from fla_terrain.sim.pointmap import BLOCK4  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    fd = FrameDataV2.build(args.data, cache=cache_path_v2(args.data))
    split = np.array([e["range_split"] for e in fd.eps])
    cls = np.array([e["class"] for e in fd.eps])
    N = fd.PTS[..., 2].astype(np.float32)                                     # [F, 2, P]
    Nb = fd.BLK[..., 2]                                                       # [F, 2, 4]
    # per-episode part labels and uniform-spread matrices (part boundaries are randomized per episode)
    part_f = np.zeros((len(N), N.shape[2]), np.int8)
    U_e = []
    ep_of_frame = np.zeros(len(N), int)
    for i, e in enumerate(fd.eps):
        sl = slice(e["offset"], e["offset"] + e["T"])
        part_f[sl] = np.asarray(e["part"])
        ep_of_frame[sl] = i
        W4 = np.asarray(e["block_w4"], float)
        U_e.append(W4 / np.maximum(W4.sum(0, keepdims=True), 1e-9))
    res = {}
    tr = (split[ep_of_frame] == "train")
    X, Y = [], []
    for f in (0, 1):
        m = tr & (N[:, f].sum(1) > 20)
        X.append(Nb[m, f])
        Y.append(N[m, f])
    X, Y = np.concatenate(X), np.concatenate(Y)
    Xa = np.column_stack([X, np.ones(len(X))])
    B, *_ = np.linalg.lstsq(Xa, Y, rcond=None)
    for c in sorted(set(cls)):
        te = (split[ep_of_frame] == "test") & (cls[ep_of_frame] == c)
        ys, yu, yl, mids = [], [], [], []
        for f in (0, 1):
            m = te & (N[:, f].sum(1) > 20)
            y = N[m, f]
            ys.append(y)
            yu.append(np.stack([U_e[e] @ nb for e, nb in zip(ep_of_frame[m], Nb[m, f])]) if m.any()
                      else np.zeros((0, y.shape[1])))
            yl.append(np.column_stack([Nb[m, f], np.ones(m.sum())]) @ B)
            mids.append((y * (part_f[m] == 1)).sum(1) / np.maximum(y.sum(1), 1e-9))
        y, yu, yl = np.concatenate(ys), np.concatenate(yu), np.concatenate(yl)
        sst = ((y - y.mean(0)) ** 2).sum()
        r = {"frames": int(len(y)), "R2_uniform": float(1 - ((y - yu) ** 2).sum() / sst),
             "R2_linear": float(1 - ((y - yl) ** 2).sum() / sst), "middle_share_mean": float(np.concatenate(mids).mean())}
        res[c] = r
        print(f"{c:10s} frames {r['frames']:6d}: R2 uniform-in-block {r['R2_uniform']:.3f}, best linear from 4 blocks "
              f"{r['R2_linear']:.3f}, middle share of load {100 * r['middle_share_mean']:.1f} %")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
