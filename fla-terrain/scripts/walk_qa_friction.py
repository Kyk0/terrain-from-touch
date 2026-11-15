#!/usr/bin/env python
"""B10: physical basis of the friction signal, from schema arrays only.

  python scripts/walk_qa_friction.py --data /path/to/fla-terrain/data/v1 [--data2 ...]

Walk-mode episodes of class hard_grip / hard_slip, frames before any fall (label_terrain != -1).
Per foot: F = sum of zone forces (sole frame, after the sensor model), u = |(Fx, Fy)| / Fz.
Stance = run of contact_gt frames (>= 3 frames) with Fz > FZ_MIN; phase = early (first 20 %), mid, late (last 20 %).
Flat foot = heel row (i_along 0) and toe row (i_along 3) each carry >= 10 % of the foot's Fz.
Saturated = u >= 0.95 * mu (prop_mu): the contact is at the friction limit, i.e. sliding.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

FZ_MIN = 20.0          # N; below this the ratio is noise dominated (sensor noise 0.15 N shear, 0.25 N normal)
SAT = 0.95
SRC = {"G113_m2200": "G113", "G80_m4200": "G80", "sonic_k1_it50000": "SONIC"}


def stances(contact, fz):
    ok = contact & (fz > FZ_MIN)
    out, s = [], None
    for i, v in enumerate(np.append(ok, False)):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if i - s >= 3:
                out.append((s, i))
            s = None
    return out


def load(dirs):
    eps = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            j = json.load(open(f))
            tp = j["terrain_params"]
            if isinstance(tp, str):
                continue
            if j["mode"] != "walk" or tp.get("class") not in ("hard_grip", "hard_slip"):
                continue
            eps.append((j, f[:-5] + ".npz"))
    return eps


def analyse(eps):
    # per (source, class): list of (u, phase, flat, mu) per stance frame; per stance: (mu, any_sat, any_sat_flat)
    frames = defaultdict(list)
    steps = defaultdict(list)
    epi = defaultdict(list)       # (src, cls) -> (mu, fell)
    for j, npz in eps:
        src = SRC.get(j["motion_source"], j["motion_source"])
        cls = j["terrain_params"]["class"]
        z = np.load(npz)
        valid = z["label_terrain"] != -1
        mu_all = z["prop_mu"]
        epi[(src, cls)].append((float(np.nanmedian(mu_all[valid])) if valid.any() else np.nan, bool(j.get("fell"))))
        for side in "LR":
            F = z[f"foot_zone_{side}"]                    # [T, 8, 3]
            tot = F.sum(1)
            fz = tot[:, 2]
            u = np.hypot(tot[:, 0], tot[:, 1]) / np.maximum(fz, 1e-6)
            heel = F[:, 0:2, 2].sum(1) / np.maximum(fz, 1e-6)
            toe = F[:, 6:8, 2].sum(1) / np.maximum(fz, 1e-6)
            flat = (heel >= 0.1) & (toe >= 0.1)
            for a, b in stances(z[f"contact_gt_{side}"].astype(bool) & valid, fz):
                n = b - a
                ph = np.full(n, 1)
                ph[: max(1, int(round(0.2 * n)))] = 0
                ph[n - max(1, int(round(0.2 * n))):] = 2
                mu = mu_all[a:b]
                sat = u[a:b] >= SAT * mu
                for k in range(n):
                    frames[(src, cls)].append((u[a + k], ph[k], flat[a + k], mu[k], sat[k]))
                steps[(src, cls)].append((float(np.median(mu)), bool(sat.any()), bool((sat & flat[a:b]).any()), n))
    return {k: np.array(v, float) for k, v in frames.items()}, {k: np.array(v, float) for k, v in steps.items()}, epi


def pct(x):
    return "%.3f/%.3f/%.3f" % tuple(np.percentile(x, [50, 90, 99])) if len(x) else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    a = ap.parse_args()
    eps = load(a.data)
    fr, st, epi = analyse(eps)
    srcs = sorted({k[0] for k in fr})
    print(f"[B10] {len(eps)} walk episodes (hard_grip/hard_slip) from {a.data}; stance Fz > {FZ_MIN:.0f} N; "
          f"u = |Fxy|/Fz per foot; saturated u >= {SAT} mu")
    print("hard_grip u median/p90/p99 (all frames | flat-foot frames), by stance phase early/mid/late:")
    for s in srcs:
        d = fr.get((s, "hard_grip"))
        if d is None:
            continue
        mu = np.median(d[:, 3])
        parts = [f"{['early', 'mid', 'late'][p]} {pct(d[d[:, 1] == p, 0])} | {pct(d[(d[:, 1] == p) & (d[:, 2] > 0), 0])}"
                 for p in (0, 1, 2)]
        print(f"  {s:5s} n={len(d):6d} mu~{mu:.2f} flat {np.mean(d[:, 2]):.2f}  " + "; ".join(parts)
              + f"  sat frames {np.mean(d[:, 4]):.3f}")
    bins = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 2.0]
    print("hard_slip by mu bin: stance frames saturated (all | flat) ; steps with >= 1 saturated frame (all | flat) ; "
          "episodes, fall rate")
    for s in srcs:
        d, sp, ep = fr.get((s, "hard_slip")), st.get((s, "hard_slip")), epi.get((s, "hard_slip"))
        if d is None:
            continue
        line = []
        thr = None
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (d[:, 3] >= lo) & (d[:, 3] < hi)
            ms = (sp[:, 0] >= lo) & (sp[:, 0] < hi)
            me = [f for mu, f in ep if lo <= mu < hi]
            if not ms.any():
                continue
            fsat, fstep = np.mean(d[m, 4]), np.mean(sp[ms, 1])
            line.append(f"[{lo:.2f},{hi:.2f}) {fsat:.2f}|{np.mean(d[m & (d[:, 2] > 0), 4]) if (m & (d[:, 2] > 0)).any() else float('nan'):.2f} ; "
                        f"{fstep:.2f}|{np.mean(sp[ms, 2]):.2f} ; {len(me)} ep, fall {np.mean(me) if me else float('nan'):.2f}")
            if fstep >= 0.5:
                thr = hi
        print(f"  {s}:")
        for x in line:
            print("    " + x)
        never = np.mean([not (sp[sp[:, 0] == sp[:, 0], 1].any())]) if len(sp) else float("nan")
        print(f"    -> slides in >= half of the steps for mu < {thr if thr is not None else 'none of the bins'}; "
              f"all hard_slip steps with a saturated frame: {np.mean(sp[:, 1]):.2f}")
    # per-episode: fraction of hard_slip episodes that never saturate
    print("hard_slip episodes with no saturated stance frame at all (friction never reached):")
    for s in srcs:
        es = [(j, n) for j, n in eps if SRC.get(j["motion_source"]) == s and j["terrain_params"]["class"] == "hard_slip"]
        never = 0
        for j, n in es:
            z = np.load(n)
            v = z["label_terrain"] != -1
            hit = False
            for side in "LR":
                tot = z[f"foot_zone_{side}"].sum(1)
                fz = tot[:, 2]
                u = np.hypot(tot[:, 0], tot[:, 1]) / np.maximum(fz, 1e-6)
                hit |= bool(((u >= SAT * z["prop_mu"]) & (fz > FZ_MIN) & z[f"contact_gt_{side}"].astype(bool) & v).any())
            never += not hit
        print(f"  {s}: {never}/{len(es)}")


if __name__ == "__main__":
    main()
