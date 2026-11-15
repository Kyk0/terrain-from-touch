#!/usr/bin/env python
"""B17 item 4: label-shuffle control for P+Nreal+Sblk (C2 with P), seed 0, the training implementation's train_v2 (read-only, same recipe),
weights written to b_runs/qa_b_models (not A's models dir). Every TRAIN and VAL episode's terrain labels are remapped by
its own random permutation of the 6 classes (-1 kept); test labels untouched. Test walk macro-F1 must be ~chance."""
import copy, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fla_terrain.models.dataset_v2 import FrameDataV2, cache_path_v2  # noqa: E402
from fla_terrain.models.v2 import train_v2  # noqa: E402

D = "/path/to/fla-terrain/data/v2"
fd = FrameDataV2.build(D, cache=cache_path_v2(D))
sp = fd.split_indices()
rng = np.random.default_rng(4321)
lab = fd.lab["terrain"].copy()
true = fd.lab["terrain"]
for em in fd.eps:
    if em["range_split"] not in ("train", "val"):
        continue
    sl = slice(em["offset"], em["offset"] + em["T"])
    perm = rng.permutation(6)
    x = lab[sl]
    k = x >= 0
    x[k] = perm[x[k]]
    lab[sl] = x
fd2 = copy.copy(fd)
fd2.lab = dict(fd.lab)
fd2.lab["terrain"] = lab
te = sp["test"]
assert np.array_equal(lab[fd.last[te]], true[fd.last[te]])
_, pred, info = train_v2(fd2, "P+Nreal+Sblk", 0, sp, "qa_b/shuffle_P+Nreal+Sblk_s0", D,
                         models_root="/path/to/fla-terrain/b_runs/qa_b_models")
y = true[fd.last[te]].astype(int)
mode = np.array([e["mode"] for e in fd.eps])[fd.ep_of_win[te]]
m = (mode == "walk") & (y >= 0)
p = np.asarray(pred["terrain"])[m]
yy = y[m]
f1 = np.mean([2 * np.sum((yy == c) & (p == c)) / max(2 * np.sum((yy == c) & (p == c)) + np.sum((yy != c) & (p == c)) + np.sum((yy == c) & (p != c)), 1) for c in sorted(set(yy.tolist()))])
res = {"test_walk_macro_f1": float(f1), "acc": float(np.mean(p == yy)), "pred_dist": (np.bincount(p, minlength=6) / len(p)).round(3).tolist(),
       "best_epoch": info.get("best_epoch"), "real_model_M1_seed0": 0.8865}
print("[B17-4]", res)
os.makedirs("results/v2/qa_b", exist_ok=True)
json.dump(res, open("results/v2/qa_b/label_shuffle.json", "w"), indent=1)
