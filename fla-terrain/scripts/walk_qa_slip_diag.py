#!/usr/bin/env python
"""B10 diagnostic (not part of the dataset): true foot slip vs the sole-frame friction utilisation.

  python scripts/walk_qa_slip_diag.py [--episodes 4] [--seconds 15]

the training implementation's sole model (flat, terrain priority 1), mu in MUS, G113 / G80 (steady 0.2-0.5 m/s) and SONIC.
Per 50 Hz frame and foot (10 physics steps averaged, noise-free SoleSensor):
  u      = |sum Fxy| / sum Fz in the sole frame (what the sensor sees),
  slide  = mean Fz-weighted tangential speed of the foot's contact points relative to the ground > SLIDE_V,
  flat   = heel row and toe row each >= 10 % of the foot's Fz (same rule as walk_qa_friction.py).
Writes data/qa_b/slip_diag.npz and prints: P(slide | u >= 0.95 mu), P(u >= 0.95 mu | slide), and per mu the fraction
of stance phases (steps) with true sliding, per source.
"""
import argparse
import os
import sys
from multiprocessing import Pool

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fla_terrain.sim import build_model  # noqa: E402
from fla_terrain.sim.sole import SoleSensor  # noqa: E402
from walking.k1_velocity import K1VelocityPolicy  # noqa: E402
from walking.scenarios import make_schedule  # noqa: E402

MUS = (0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.9)
SLIDE_V = 0.05          # m/s; above soft-contact creep (mid-stance p90 <= 0.01 at mu 0.9); MuJoCo's regularized friction reaches the cone at ~0.01-0.05 m/s slip (NOTES_A T2)
FZ_MIN = 20.0
OUT = "/path/to/fla-terrain/data/qa_b"
_S = {}


def job(a):
    src_name, mu, seed, seconds = a
    m, info = build_model({"type": "flat", "mu": mu})
    d = mujoco.MjData(m)
    if src_name == "SONIC":
        if "sonic" not in _S:
            from walking.sonic_k1 import SonicK1
            _S["sonic"] = SonicK1()
        src = _S["sonic"]
        sched = lambda t: np.zeros(3)  # noqa: E731
    else:
        src = K1VelocityPolicy(src_name)
        sched = make_schedule("steady", np.random.default_rng(seed))
    rng = np.random.default_rng(seed)
    src.reset(m, d, rng)
    sens = SoleSensor(m, info["sole_cfg"])
    zg = {s: set(info["zone_geoms"][s]) for s in "LR"}
    terr = set(info["terrain_geoms"])
    fb = info["foot_bodies"]
    rows, acc, f6, vel = [], [], np.zeros(6), np.zeros(6)
    vacc = {s: [0.0, 0.0] for s in "LR"}
    for k in range(int(seconds / m.opt.timestep)):
        src.pre_step(m, d, sched(k * m.opt.timestep))
        mujoco.mj_step(m, d)
        if src.fallen(m, d):
            break
        F = sens.read(d)
        acc.append(np.stack([F["L"], F["R"]]))
        for s in "LR":
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, fb[s], vel, 0)   # [ang, lin] at body com, world axes
            w, v0, c0 = vel[:3].copy(), vel[3:].copy(), d.xipos[fb[s]]
            for c in range(d.ncon):
                g1, g2 = d.contact.geom[c]
                if not ((g1 in zg[s] and g2 in terr) or (g2 in zg[s] and g1 in terr)):
                    continue
                mujoco.mj_contactForce(m, d, c, f6)
                p = d.contact.pos[c]
                vp = v0 + np.cross(w, p - c0)
                n = d.contact.frame[c][:3]
                vt = np.linalg.norm(vp - np.dot(vp, n) * n)
                vacc[s][0] += f6[0] * vt
                vacc[s][1] += f6[0]
        if (k + 1) % 10 == 0:
            Fm = np.mean(acc, 0)
            acc = []
            for i, s in enumerate("LR"):
                tot = Fm[i].sum(0)
                fz = tot[2]
                vslip = vacc[s][0] / vacc[s][1] if vacc[s][1] > 0 else 0.0
                heel, toe = Fm[i][0:2, 2].sum() / max(fz, 1e-6), Fm[i][6:8, 2].sum() / max(fz, 1e-6)
                rows.append((i, k * m.opt.timestep, fz, np.hypot(tot[0], tot[1]) / max(fz, 1e-6), vslip,
                             heel >= 0.1 and toe >= 0.1))
                vacc[s] = [0.0, 0.0]
    return src_name, mu, seed, np.array(rows)


def steps_of(r):
    """stance runs per foot: consecutive frames with Fz > FZ_MIN (>= 3 frames)."""
    out = []
    for foot in (0, 1):
        x = r[r[:, 0] == foot]
        on = np.append(x[:, 2] > FZ_MIN, False)
        s = None
        for i, v in enumerate(on):
            if v and s is None:
                s = i
            elif not v and s is not None:
                if i - s >= 3:
                    out.append(x[s:i])
                s = None
    return out


def slip_distance_table(path=os.path.join(OUT, "slip_diag.npz")):
    """Per source and mu: median slip distance per step (mm, Fz-weighted contact slip speed x 20 ms, whole stance)
    and fraction of steps that slide >= 5 mm / >= 20 mm. Sliding distance, not frame counts: brief 1-2 frame
    transients (load transfer) exist on every surface and are not 'the foot slides'."""
    z = np.load(path)
    print("slip distance per step (whole stance): median mm | frac steps >= 5 mm | frac >= 20 mm")
    for s in ("G113_m2200", "G80_m4200", "SONIC"):
        out = []
        for mu in MUS:
            st = [x for k in z.files if k.startswith(f"{s}_{mu}_") for x in steps_of(z[k])]
            if not st:
                continue
            dist = np.array([np.sum(x[:, 4]) * 0.02 * 1000 for x in st])
            out.append(f"{mu}: {np.median(dist):.0f}|{np.mean(dist >= 5):.2f}|{np.mean(dist >= 20):.2f}")
        print(f"  {s:10s} " + "  ".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()
    if a.summarize_only:
        slip_distance_table()
        return
    jobs = [(s, mu, e, a.seconds) for s in ("G113_m2200", "G80_m4200", "SONIC") for mu in MUS for e in range(a.episodes)]
    with Pool(min(40, len(jobs))) as p:
        res = p.map(job, jobs, chunksize=1)
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "slip_diag.npz"), **{f"{s}_{mu}_{e}": r for s, mu, e, r in res})
    print(f"[B10 diag] true slip vs sole u; slide = contact slip speed > {SLIDE_V} m/s; stance Fz > {FZ_MIN} N; "
          f"{a.episodes} eps x {a.seconds:.0f} s per (source, mu)")
    for s in ("G113_m2200", "G80_m4200", "SONIC"):
        R = [(mu, r) for ss, mu, e, r in res if ss == s and len(r)]
        allr = np.concatenate([np.column_stack([np.full(len(r), mu), r]) for mu, r in R])
        st = allr[allr[:, 3] > FZ_MIN]
        sat = st[:, 4] >= 0.95 * st[:, 0]
        slide = st[:, 5] > SLIDE_V
        flat = st[:, 6] > 0
        prec = lambda m: np.mean(slide[m]) if m.any() else float("nan")  # noqa: E731
        print(f"{s:10s} P(slide|sat) all {prec(sat):.2f} flat {prec(sat & flat):.2f}; "
              f"P(sat|slide) all {np.mean(sat[slide]):.2f} flat {np.mean(sat[slide & flat]) if (slide & flat).any() else float('nan'):.2f}; "
              f"on mu>=0.6: sat frames {np.mean(sat[st[:, 0] >= 0.6]):.3f}, slide frames {np.mean(slide[st[:, 0] >= 0.6]):.3f}")
        line = []
        for mu in MUS:
            stp = [x for m_, r in R if m_ == mu for x in steps_of(r)]
            if not stp:
                continue
            def mid(x):
                e = max(1, int(round(0.2 * len(x))))
                return x[e:len(x) - e] if len(x) > 2 * e else x[:0]
            fs = np.mean([(x[:, 4] > SLIDE_V).any() for x in stp])
            fm = np.mean([(mid(x)[:, 4] > SLIDE_V).any() for x in stp])
            fsat = np.mean([(x[:, 3] >= 0.95 * mu).any() for x in stp])
            line.append(f"mu {mu}: any {fs:.2f} mid {fm:.2f} sensor-sat {fsat:.2f}")
        print("   steps sliding (true, any phase | mid-stance 20-80 %) vs steps with a sensor-saturated frame:")
        print("   " + "; ".join(line[:4]))
        print("   " + "; ".join(line[4:]))
    slip_distance_table()


if __name__ == "__main__":
    main()
