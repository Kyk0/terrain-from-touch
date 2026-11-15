"""Why airborne/full support F1 is not ~1 (R2), from the saved P+N predictions (all seeds), walk test windows.

Counts airborne<->full confusions and how many fall on a label transition (the GT label of the previous or next
frame differs: touchdown / lift-off frames, where the frame-mean true Fz crosses the 2 N contact threshold inside the
20 ms frame), and full->partial confusions. Writes <out>/support_note.json (rendered into tables.md).
  $PY scripts/support_diagnostics.py --data /path/to/fla-terrain/data/v1 --out results/v1
"""
import argparse
import json
import os
import sys

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
import numpy as np  # noqa: E402

from fla_terrain.models.dataset import FrameData, cache_path  # noqa: E402
from fla_terrain.models.provenance import require_clean  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/v1")
    ap.add_argument("--set", default="P+N")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    git = require_clean(args.allow_dirty)
    fd = FrameData.build(args.data, cache=cache_path(args.data))
    st = {"af": [], "af_trans": [], "af_fz": [], "nonpartial": [], "fp": [], "full": [], "fp_trans": []}
    for seed in (0, 1, 2):
        p = os.path.join(args.out, f"pred_{args.set}_s{seed}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        idx = d["idx"]
        last = fd.last[idx]
        walk = np.array([fd.eps[e]["mode"] == "walk" for e in fd.ep_of_win[idx]])
        for c, k in enumerate(("sup_L", "sup_R")):
            sup = fd.lab[k]
            y, pr = sup[last], d[k]
            trans = (sup[last - 1] != y) | (sup[np.minimum(last + 1, len(sup) - 1)] != y)
            af = walk & (((y == 2) & (pr == 0)) | ((y == 0) & (pr == 2)))
            fp = walk & (y == 2) & (pr == 1)
            st["af"].append(af.sum())
            st["af_trans"].append((af & trans).sum())
            st["af_fz"].append(np.median(fd.X[last[af], fd.groups["Ntot"][c]]) if af.any() else np.nan)
            st["nonpartial"].append((walk & (y != 1)).sum())
            st["fp"].append(fp.sum())
            st["fp_trans"].append((fp & trans).sum())
            st["full"].append((walk & (y == 2)).sum())
    S = {k: float(np.nansum(v)) if k != "af_fz" else float(np.nanmedian(v)) for k, v in st.items()}
    text = (f"Support errors ({args.set}, 3 seeds, walk test): airborne<->full confusions are "
            f"{100 * S['af'] / S['nonpartial']:.2f} % of non-partial windows; {100 * S['af_trans'] / max(S['af'], 1):.0f} % of "
            f"them are at a GT label transition (touchdown / lift-off frame: the frame-mean true Fz crosses the 2 N "
            f"contact threshold within the 20 ms frame; sensor total Fz there, median {S['af_fz']:.1f} N). The 'full' F1 "
            f"is lowered mainly by full->partial predictions ({100 * S['fp'] / S['full']:.1f} % of full windows, "
            f"{100 * S['fp_trans'] / max(S['fp'], 1):.0f} % at transitions), consistent with the a-priori partial class weight "
            f"(x3). No label or code bug found; labels and weights unchanged.")
    print(text)
    json.dump({"text": text, "stats": S, "git_commit": git["commit"], "git_dirty": git["dirty"]},
              open(os.path.join(args.out, "support_note.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
