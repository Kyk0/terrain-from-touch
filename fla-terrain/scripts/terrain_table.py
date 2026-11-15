"""T4 check: train/val/test parameter ranges per terrain class and parameter, with an overlap test.

  $PY scripts/terrain_table.py [--n 5000]
1. Band definitions (fla_terrain.sim.terrains.BANDS) must be pairwise disjoint across splits, and test must have
   one band inside the train span and one outside it.
2. Empirical: sample n terrains per class and split; per class and parameter (mu of every region, stiffness of
   soft regions), the sampled ranges of different splits must not overlap.
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
import numpy as np  # noqa: E402

from fla_terrain.sim.terrains import CLASSES, RIGID_STIFFNESS, SPLITS, band_table, check_band_overlap, sample_terrain  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    args = ap.parse_args()
    print(f"{'param':8s} {'split':5s} {'band':9s} {'lo':>7s} {'hi':>7s}")
    for p, s, b, lo, hi in band_table():
        print(f"{p:8s} {s:5s} {b:9s} {lo:7.4f} {hi:7.4f}")
    errs = check_band_overlap()
    rng = np.random.default_rng(0)
    print(f"\nsampled ranges ({args.n} terrains per class and split; soft stiffness in kN/m):")
    print(f"{'class':18s} {'param':10s} " + " ".join(f"{s:>17s}" for s in SPLITS) + "  overlap")
    for cls in CLASSES:
        vals = {"mu": {s: [] for s in SPLITS}, "K_soft": {s: [] for s in SPLITS}}
        for s in SPLITS:
            for _ in range(args.n // (10 if cls.startswith(("rough", "edge", "strip")) else 1)):
                cfg = sample_terrain(cls, rng, s)
                for r in cfg["regions"]:
                    vals["mu"][s].append(r["mu"])
                    if r["stiffness"] != RIGID_STIFFNESS:
                        vals["K_soft"][s].append(r["stiffness"] / 1e3)
        for p, d in vals.items():
            if not any(d.values()):
                continue
            bad = sum(_foreign(v, s, p) for s in SPLITS for v in d[s])
            txt = " ".join(f"{np.min(d[s]):8.3f}-{np.max(d[s]):<8.3f}" if d[s] else f"{'-':>17s}" for s in SPLITS)
            print(f"{cls:18s} {p:10s} {txt}  {'OVERLAP (%d values)' % bad if bad else 'none'}")
            if bad:
                errs.append(f"{cls}/{p}: {bad} sampled values fall in another split's band")
    print(f"\nOVERLAP TEST {'PASS' if not errs else 'FAIL'}" + ("" if not errs else ": " + "; ".join(errs)))
    return 0 if not errs else 1


def _foreign(v, split, p):
    """True if value v (sampled for `split`) lies in a band of another split, or in no band at all."""
    from fla_terrain.sim.terrains import BANDS, stiffness_from_tc
    names = ["soft_tc"] if p == "K_soft" else ["mu_grip", "mu_slip"]
    owners = set()
    for name in names:
        for s, bands in BANDS[name].items():
            for _, lo, hi in bands:
                if p == "K_soft":
                    lo, hi = sorted((stiffness_from_tc(hi) / 1e3, stiffness_from_tc(lo) / 1e3))
                if lo - 1e-9 <= v <= hi + 1e-9:
                    owners.add(s)
    return owners != {split}


if __name__ == "__main__":
    sys.exit(main())
