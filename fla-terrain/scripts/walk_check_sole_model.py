#!/usr/bin/env python
"""B5: walking policies on the training implementation's sole model (fla_terrain.sim.build_model).

16 episodes x 20 s per (policy, mu); fall rate, speed, steps/s (same runner as B1/B3) and the
SoleSensor total-shear / Fz ratio during stance (does it saturate at mu on the slippery floor?).
  python scripts/walk_check_sole_model.py [--episodes 16]
"""
import argparse
import os
import sys
from multiprocessing import Pool

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.environ.get("FLA_A_ROOT"):  # temporary: the training implementation tree with fla_terrain/data (not yet on main)
    sys.path.insert(0, os.environ["FLA_A_ROOT"])
from fla_terrain.sim import build_model  # noqa: E402
from fla_terrain.sim.sole import SoleSensor  # noqa: E402
from walking.evaluate import run_episode  # noqa: E402
from walking.k1_velocity import K1VelocityPolicy  # noqa: E402

_M = {}


def model(mu):
    if mu not in _M:
        m, info = build_model({"type": "flat", "mu": mu})
        info = dict(info, spawn=(0.0, 0.0, 0.58), support=None)
        _M[mu] = (m, info)
    return _M[mu]


def job(a):
    pol, mu, seed, vx = a
    m, info = model(mu)
    r = run_episode(m, info, (vx, 0.0, 0.0), policy=pol, seed=seed)
    r.update(policy=pol, mu=mu)
    return r


def shear_ratio(pol, mu, seed=0, seconds=12.0):
    """Per physics step, per foot in stance (Fz > 50 N): |sum Fxy| / sum Fz over the 8 zones."""
    m, info = model(mu)
    d = mujoco.MjData(m)
    src = K1VelocityPolicy(pol)
    src.reset(m, d, np.random.default_rng(seed))
    sens = SoleSensor(m)
    r, r50, acc = [], [], []
    for i in range(int(seconds / m.opt.timestep)):
        src.pre_step(m, d, (0.4, 0, 0) if i > 500 else (0, 0, 0))
        mujoco.mj_step(m, d)
        F = sens.read(d)
        tot = np.array([F["L"].sum(0), F["R"].sum(0)])
        acc.append(tot)
        for f in tot:
            if f[2] > 50:
                r.append(np.hypot(f[0], f[1]) / f[2])
        if len(acc) == 10:  # 50 Hz frame average
            fa = np.mean(acc, 0)
            acc = []
            r50 += [np.hypot(f[0], f[1]) / f[2] for f in fa if f[2] > 50]
    return np.percentile(r, [50, 95, 99, 100]), np.percentile(r50, [50, 99, 100]), src.fallen(m, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=16)
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    jobs = [(p, mu, s, float(rng.uniform(0.3, 0.5)))
            for p in ("G113_m2200", "G80_m4200") for mu in (1.0, 0.2) for s in range(a.episodes)]
    with Pool(min(40, len(jobs))) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    print(f"[B5] sole model (fla_terrain.sim.build_model, 2x4 zones, terrain priority 1), {a.episodes} eps x 20 s")
    print(f"{'policy':11s} {'mu':>4s} {'fall':>5s} {'vx':>6s} {'track':>5s} {'steps/s':>7s} "
          f"{'|Fxy|/Fz p50/p95/p99/max (500Hz)':>34s} {'50Hz p50/p99/max':>17s}")
    for p in ("G113_m2200", "G80_m4200"):
        for mu in (1.0, 0.2):
            rs = [r for r in rows if r["policy"] == p and r["mu"] == mu]
            q, q50, _ = shear_ratio(p, mu)
            print(f"{p:11s} {mu:4.1f} {np.mean([r['fell'] for r in rs]):5.2f} "
                  f"{np.nanmean([r['vx'] for r in rs]):6.3f} "
                  f"{np.nanmean([r['vx'] / r['cmd'][0] for r in rs]):5.2f} "
                  f"{np.mean([r['steps_per_s'] for r in rs]):7.2f} "
                  f"{'/'.join(f'{x:.3f}' for x in q):>34s} {'/'.join(f'{x:.3f}' for x in q50):>17s}")


if __name__ == "__main__":
    main()
