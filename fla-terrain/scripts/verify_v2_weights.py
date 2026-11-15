"""Reload saved v2 models and reproduce their stored test predictions (test_pred.npz) from the weights.
  $PY scripts/verify_v2_weights.py --root /path/to/fla-terrain/models/v2_src3 [--per-group 2 | --all]
Data set of each model: its config.json 'data_dir' (cache built on demand)."""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
import numpy as np  # noqa: E402

from fla_terrain.models.dataset_v2 import FrameDataV2, cache_path_v2  # noqa: E402
from fla_terrain.models.v2 import predict_saved  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/path/to/fla-terrain/models/v2_src3")
ap.add_argument("--per-group", type=int, default=2)
ap.add_argument("--all", action="store_true")
a = ap.parse_args()
groups = {}
for d in sorted(glob.glob(f"{a.root}/*/*/config.json")):
    groups.setdefault(d.split("/")[-3], []).append(os.path.dirname(d))
fds, ok, n = {}, 0, 0
for g, dirs in groups.items():
    for md in (dirs if a.all else dirs[:: max(1, len(dirs) // a.per_group)][: a.per_group]):
        cfg = json.load(open(f"{md}/config.json"))
        dd = cfg["data_dir"]
        if dd not in fds:
            fds = {dd: FrameDataV2.build(dd, cache=cache_path_v2(dd), workers=16)}
        st = np.load(f"{md}/test_pred.npz")
        p = predict_saved(md, fds[dd], st["idx"], device="cuda")
        same = all(np.array_equal(p[k], st[k]) for k in ("terrain", "sup_L", "sup_R")) and \
            all(np.allclose(p[k], st[k], rtol=1e-4, atol=1e-5) for k in ("mu", "logk", "slope"))
        ok += same
        n += 1
        print(f"  {g}/{os.path.basename(md)}: {'reproduced' if same else 'MISMATCH'}", flush=True)
print(f"VERIFY {ok}/{n} models reproduce their stored test predictions")
