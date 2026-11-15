#!/usr/bin/env python
"""B12: recorded (sensor) foot shear and Fz vs the simulator's true foot wrench, per stance, G113 walking.
20 hard_grip + 20 hard_slip episodes (train bands, per-episode random sole parameters as dataset v2), 20 s each.
Per 50 Hz frame: true = simulator foot wrench (SoleSensor) averaged over the frame's physics steps; recorded = sum over
points of the recorder's output frame. Stance = run of frames with true Fz > 20 N (>= 5 frames). Ratio per stance =
mean over stance frames of |recorded shear| / mean of |true shear| (and the same for Fz). Also the per-episode summed
offset of the point sensor model (bias of the summed shear)."""
import os, sys
from multiprocessing import Pool
import mujoco, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fla_terrain.sim.build import build_model_v2  # noqa: E402
from fla_terrain.sim.pointmap import SoleV2Params  # noqa: E402
from fla_terrain.sim.recorder import EpisodeRecorder  # noqa: E402
from fla_terrain.sim.sole import SoleSensor  # noqa: E402
from fla_terrain.sim.terrains import sample_terrain  # noqa: E402
from walking import K1VelocityPolicy  # noqa: E402


def episode(a):
    cls, seed = a
    rng = np.random.default_rng([seed, 7])
    cfg = sample_terrain(cls, rng, "train")
    prm = SoleV2Params.sample(rng)
    if os.environ.get("QA_NO_SAT"):   # diagnostic: per-point saturation off, everything else identical
        import dataclasses
        prm = dataclasses.replace(prm, sat_normal=1e9, sat_shear=1e9)
    m, info = build_model_v2(cfg, params=prm)
    d = mujoco.MjData(m)
    src = K1VelocityPolicy("G113_m2200")
    src.reset(m, d, rng, base_pos=(0, 0, 0.58), base_yaw=cfg["yaw"])
    rec = EpisodeRecorder(m, info, motion_source="G113", mode="walk", split_key="qa", rng=rng)
    rec.reset(d)
    ss = SoleSensor(m, info["sole_cfg"])
    acc, n, true = np.zeros((2, 3)), 0, []
    nf = 0
    for i in range(int(20.0 / m.opt.timestep)):
        src.pre_step(m, d, (0.35, 0, 0) if i * m.opt.timestep > 1 else (0, 0, 0))
        mujoco.mj_step(m, d)
        rec.step(d)
        w = ss.read_full(d)["wrench"]
        for k, s in enumerate("LR"):
            if w[s]:
                acc[k] += w[s][0]
        n += 1
        if len(rec.frames["F"]) > nf:
            nf = len(rec.frames["F"]); true.append(acc / n); acc, n = np.zeros((2, 3)), 0
        if (i + 1) % 10 == 0 and src.fallen(m, d):
            break
    Fr = np.asarray(rec.frames["F"]).sum(2)          # [T, 2, 3] recorded sums
    Tr = np.asarray(true)[:len(Fr)]
    out = []
    for k in (0, 1):
        on = np.append(Tr[:, k, 2] > 20, False)
        s0 = None
        for t, v in enumerate(on):
            if v and s0 is None:
                s0 = t
            elif not v and s0 is not None:
                if t - s0 >= 5:
                    sl = slice(s0, t)
                    rs, ts = np.hypot(*Fr[sl, k, :2].T).mean(), np.hypot(*Tr[sl, k, :2].T).mean()
                    out.append((k, rs / max(ts, 1e-9), Fr[sl, k, 2].mean() / Tr[sl, k, 2].mean(), ts))
                s0 = None
    off = rec.psensor.offset.sum(1)                   # [2, 3] summed per-point offsets per foot
    return cls, out, off


if __name__ == "__main__":
    jobs = [(c, s) for c in ("hard_grip", "hard_slip") for s in range(20)]
    with Pool(40) as p:
        res = p.map(episode, jobs, chunksize=1)
    print(("[NO SATURATION] " if os.environ.get("QA_NO_SAT") else "") + "[B12] G113, 20 eps per class, random sole params (dataset v2 ranges); per stance recorded / true")
    print(f"{'class':10s} {'foot':4s} {'n':>4s} | shear ratio p5 / median / p95 | Fz ratio p5 / median / p95 | true |shear| median")
    for cls in ("hard_grip", "hard_slip"):
        for k, f in ((0, "L"), (1, "R")):
            r = np.array([x for c, o, _ in res if c == cls for x in o if x[0] == k])
            q = lambda v: "/".join(f"{x:.3f}" for x in np.percentile(v, [5, 50, 95]))  # noqa: E731
            print(f"{cls:10s} {f:4s} {len(r):4d} | {q(r[:, 1]):>24s} | {q(r[:, 2]):>24s} | {np.median(r[:, 3]):6.1f} N")
    off = np.concatenate([o for _, _, o in res])
    print(f"summed per-point offset per foot (episodes x feet = {len(off)}): |shear part| median {np.median(np.hypot(*off[:, :2].T)):.2f} N, "
          f"p95 {np.percentile(np.hypot(*off[:, :2].T), 95):.2f} N; Fz part |median| {np.median(np.abs(off[:, 2])):.2f} N")
