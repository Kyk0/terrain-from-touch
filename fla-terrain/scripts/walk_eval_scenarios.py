#!/usr/bin/env python
"""B6: tracking of each command schedule by each policy (flat, mu 1.0, base MJCF).
  python scripts/walk_eval_scenarios.py [--episodes 16]
vx_err / wz_err = RMS(1 s moving average of body velocity - same average of the command delayed 0.5 s),
over walking time (command non-zero, t > 2 s). The 1 s average removes the within-step sway."""
import argparse, os, sys
from multiprocessing import Pool
import mujoco
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from walking import K1VelocityPolicy, load_k1_model  # noqa: E402
from walking.k1_velocity import quat_rotate_inverse  # noqa: E402
from walking.scenarios import KINDS, make_schedule  # noqa: E402

LAG = 0.5

def job(a):
    pol, kind, seed = a
    m = load_k1_model(); d = mujoco.MjData(m)
    src = K1VelocityPolicy(pol); rng = np.random.default_rng(seed)
    src.reset(m, d, rng); sch = make_schedule(kind, rng)
    ev, ew, fell, ts = [], [], False, m.opt.timestep
    for i in range(int(20.0 / ts)):
        t = i * ts
        src.pre_step(m, d, sch(t)); mujoco.mj_step(m, d)
        if (i + 1) % 10: continue
        if src.fallen(m, d): fell = True; break
        c = sch(max(t - LAG, 0.0))
        vb = quat_rotate_inverse(d.qpos[3:7], d.qvel[0:3])
        ev.append((t, vb[0], d.qvel[5], c[0], c[2]))
    e = np.array(ev); k = np.ones(50) / 50
    sm = np.stack([np.convolve(e[:, j], k, "valid") for j in range(1, 5)], 1); tt = e[49:, 0]
    sel = (tt > 2.5) & (np.abs(sm[:, 2:]).max(1) > 0.05)
    if not sel.any(): return pol, kind, fell, np.nan, np.nan
    return (pol, kind, fell, float(np.sqrt(np.mean((sm[sel, 0] - sm[sel, 2]) ** 2))),
            float(np.sqrt(np.mean((sm[sel, 1] - sm[sel, 3]) ** 2))))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--episodes", type=int, default=16); a = ap.parse_args()
    jobs = [(p, k, s) for p in ("G113_m2200", "G80_m4200") for k in KINDS for s in range(a.episodes)]
    with Pool(40) as pool: rows = pool.map(job, jobs, chunksize=1)
    print(f"[B6] schedules, {a.episodes} eps x 20 s, flat mu 1.0; RMS errors vs command delayed {LAG}s")
    print(f"{'policy':11s} {'kind':13s} {'fall':>5s} {'vx_rms':>7s} {'wz_rms':>7s}")
    for p in ("G113_m2200", "G80_m4200"):
        for k in KINDS:
            r = [x for x in rows if x[0] == p and x[1] == k]
            print(f"{p:11s} {k:13s} {np.mean([x[2] for x in r]):5.2f} {np.mean([x[3] for x in r]):7.3f} {np.mean([x[4] for x in r]):7.3f}")
