#!/usr/bin/env python
"""B16: label-shuffle control. Same architecture and training (A's train_net, imported read-only) on P+N+S, but the
terrain labels of every TRAIN and VAL episode are remapped through an episode-specific random permutation of the 6
classes (-1 kept; the whole episode keeps one map; test labels untouched). Test macro-F1 (walk) must be ~chance.
  python scripts/qa_label_shuffle.py [--seed 0]"""
import argparse, copy, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fla_terrain.models.dataset import FrameData  # noqa: E402
from fla_terrain.models.train import train_net  # noqa: E402


def macro_f1(y, p):
    f = []
    for k in range(6):
        tp, fp, fn = np.sum((y == k) & (p == k)), np.sum((y != k) & (p == k)), np.sum((y == k) & (p != k))
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--set", default="P+N+S")
    a = ap.parse_args()
    root = "/path/to/fla-terrain/data/v1"
    fd = FrameData.build(root, cache=root + "_cache.npz")
    splits = fd.split_indices()
    rng = np.random.default_rng(1234 + a.seed)
    fd2 = copy.copy(fd)
    fd2.lab = dict(fd.lab)
    lab = fd.lab["terrain"].copy()
    true_lab = fd.lab["terrain"]
    n_changed = 0
    for e, em in enumerate(fd.eps):
        if em["range_split"] not in ("train", "val"):
            continue
        sl = slice(em["offset"], em["offset"] + em["T"])
        perm = rng.permutation(6)
        x = lab[sl]
        known = x >= 0
        x[known] = perm[x[known]]
        lab[sl] = x
        n_changed += int(np.sum(x[known] != true_lab[sl][known]))
    fd2.lab["terrain"] = lab
    assert np.array_equal(lab[fd.last[splits["test"]]], true_lab[fd.last[splits["test"]]])
    _, pred, info = train_net(fd2, a.set, a.seed, splits, device="cuda")
    te = splits["test"]
    y = true_lab[fd.last[te]].astype(int)
    mode = np.array([em["mode"] for em in fd.eps])[fd.ep_of_win[te]]
    m = (mode == "walk") & (y >= 0)
    f1 = macro_f1(y[m], pred["terrain"][m])
    maj = np.bincount(pred["terrain"][m], minlength=6) / m.sum()
    out = {"set": a.set, "seed": a.seed, "frames_relabelled_frac": n_changed / max(1, int(np.sum(true_lab >= 0))),
           "test_walk_macro_f1": f1, "test_walk_acc": float(np.mean(y[m] == pred["terrain"][m])),
           "pred_class_dist": maj.round(3).tolist(), "train_info": info}
    print(f"[B16] label-shuffle control {a.set} seed {a.seed}: test walk macro-F1 {f1:.3f} (chance 1/6 = 0.167 for "
          f"uniform guessing; real model 0.86), acc {out['test_walk_acc']:.3f}, predicted class dist {out['pred_class_dist']}, "
          f"best epoch {info['best_epoch']}")
    json.dump(out, open("results/v1/qa_label_shuffle_B16.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
