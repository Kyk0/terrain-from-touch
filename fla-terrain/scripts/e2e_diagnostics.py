"""Diagnostics on generated episodes (50 Hz data as the model sees it).

Per terrain class and foot, over stance frames (label_support == 2, both this and the previous frame):
  hf_fz   = std of the first difference of the foot's total Fz / mean total Fz   (chatter measure)
  near_mu = fraction of stance frames with |shear|/Fz within 5 % of prop_mu       (sliding at the cone limit)
Label sanity per foot: fractions airborne / partial / full, contact_gt off while total Fz > 10 N
(dropouts), and the force-based airborne fraction (total Fz < 5 N).
  $PY scripts/e2e_diagnostics.py --data DIR
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from fla_terrain.schema import TERRAIN_LABELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    acc = defaultdict(lambda: defaultdict(list))
    for p in sorted(glob.glob(f"{args.data}/**/episode_*.npz", recursive=True)):
        meta = json.load(open(p[:-4] + ".json"))
        z = np.load(p)
        ft = meta.get("fall_time")
        keep = np.ones(len(z["t"]), bool) if ft is None else z["t"] <= ft
        for s in "LR":
            F = z[f"foot_zone_{s}"][keep].sum(1)
            fz = F[:, 2]
            sh = np.hypot(F[:, 0], F[:, 1])
            sup = z[f"label_support_{s}"][keep]
            lab = z["label_terrain"][keep]
            mu = z["prop_mu"][keep]
            cg = z[f"contact_gt_{s}"][keep]
            st = (sup == 2)
            pair = st[1:] & st[:-1]
            for c in np.unique(lab[lab >= 0]):
                k = (TERRAIN_LABELS[int(c)], meta["mode"])
                a = acc[k]
                cm = lab == c
                pm = pair & cm[1:]
                dfz = np.diff(fz)[pm]
                if pm.sum() > 2:
                    a["hf_fz"].append(dfz.std() / fz[1:][pm].mean())
                sm = st & cm & (fz > 5)
                a["near_mu"].append(np.mean(np.abs(sh[sm] / fz[sm] - mu[sm]) <= 0.05 * mu[sm]) if sm.any() else np.nan)
                a["ratio_med"].append(np.median(sh[sm] / fz[sm]) if sm.any() else np.nan)
                a["mu"].append(np.nanmean(mu[cm]))
                a["sup"].append(np.bincount(sup[cm], minlength=3) / cm.sum())
                a["dropout"].append(np.mean(~cg[cm] & (fz[cm] > 10)))
                a["air_force"].append(np.mean(fz[cm] < 5))
    print(f"{'class/mode':22s} {'mu':>5s} {'hf_Fz':>6s} {'near_mu':>7s} {'|sh|/Fz med':>11s} | support air/part/full | "
          f"dropout air(Fz<5N)")
    for k in sorted(acc):
        a = acc[k]
        sup = np.mean(a["sup"], 0)
        print(f"{k[0] + '/' + k[1]:22s} {np.nanmean(a['mu']):5.2f} {np.nanmean(a['hf_fz']):6.3f} "
              f"{np.nanmean(a['near_mu']):7.3f} {np.nanmean(a['ratio_med']):11.3f} | {sup[0]:.3f}/{sup[1]:.3f}/{sup[2]:.3f}"
              f"        | {np.mean(a['dropout']):.3f}   {np.mean(a['air_force']):.3f}")


if __name__ == "__main__":
    main()
