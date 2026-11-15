#!/usr/bin/env python
"""B8: single-process throughput with sole sensor + EpisodeRecorder, and a lifted-mode check.

  OMP_NUM_THREADS=1 python scripts/walk_throughput.py
Throughput: flat mu 1.0, dt 0.002, 20 s episode, recorder.step after every mj_step, write() excluded.
Lifted: base welded to a mocap 0.3 m above the standing height (as gen_data.py 'lifted'), source keeps
running 10 s; checks NaN, max joint speed, and for SONIC the joint tracking error / clip switches.
"""
import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fla_terrain.sim import EpisodeRecorder, build_model  # noqa: E402
from walking.k1_velocity import K1VelocityPolicy  # noqa: E402
from walking.sonic_k1 import SonicK1  # noqa: E402


def make(name):
    return SonicK1() if name == "SONIC" else K1VelocityPolicy(name)


def throughput(name, seconds=20.0):
    m, info = build_model({"type": "flat", "mu": 1.0})
    d = mujoco.MjData(m)
    src = make(name)
    rng = np.random.default_rng(0)
    src.reset(m, d, rng)
    rec = EpisodeRecorder(m, info, motion_source=src.name, mode="walk", split_key="tp", rng=rng)
    rec.reset(d)
    cmd = np.zeros(3) if name == "SONIC" else np.array([0.4, 0.0, 0.0])
    n = int(seconds / m.opt.timestep)
    t0 = time.perf_counter()
    for _ in range(n):
        src.pre_step(m, d, cmd)
        mujoco.mj_step(m, d)
        rec.step(d, cmd)
    wall = time.perf_counter() - t0
    # same without recorder, to split the cost
    src.reset(m, d, rng)
    t1 = time.perf_counter()
    for _ in range(n):
        src.pre_step(m, d, cmd)
        mujoco.mj_step(m, d)
    wall_nr = time.perf_counter() - t1
    return seconds / wall, seconds / wall_nr, src.fallen(m, d), len(rec.frames["F"])


def lifted(name, seconds=10.0, dz=0.3):
    m, info = build_model({"type": "flat", "mu": 1.0}, base_rig=True)
    d = mujoco.MjData(m)
    src = make(name)
    src.reset(m, d, np.random.default_rng(1))
    b = info["base_body"]
    d.qpos[2] += dz
    d.qvel[:6] = 0.0
    mujoco.mj_forward(m, d)
    d.mocap_pos[0] = d.xpos[b]
    d.mocap_quat[0] = d.xquat[b]
    cmd = np.zeros(3) if name == "SONIC" else np.array([0.4, 0.0, 0.0])
    maxv, jerr, nan = 0.0, [], False
    for k in range(int(seconds / m.opt.timestep)):
        src.pre_step(m, d, cmd)
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
            nan = True
            break
        maxv = max(maxv, float(np.abs(d.qvel[6:]).max()))
        if name == "SONIC" and k % 10 == 9:
            jerr.append(float(np.abs(d.qpos[src.qadr] - src.clip["dof_pos"][src.frame]).mean()))
    extra = ""
    if name == "SONIC":
        j = np.array(jerr)
        extra = (f" joint err mean {j.mean():.3f} rad (first 1 s {j[:50].mean():.3f}, last 1 s {j[-50:].mean():.3f}),"
                 f" clips played {len(src.clip_log)}")
    else:
        extra = f" action |a| mean {np.abs(src.action).mean():.2f}, gait freq {src.gait_freq:.2f} Hz"
    return nan, maxv, extra


if __name__ == "__main__":
    print(f"[B8] load {os.getloadavg()[0]:.1f} on {os.cpu_count()} cores; OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    for name in ("G113_m2200", "G80_m4200", "SONIC"):
        sps, sps_nr, fell, frames = throughput(name)
        print(f"[throughput] {name:11s} {sps:5.1f} sim-s/wall-s with sole+recorder ({sps_nr:5.1f} without); "
              f"fell={fell} frames={frames}")
    for name in ("G113_m2200", "G80_m4200", "SONIC"):
        nan, maxv, extra = lifted(name)
        print(f"[lifted] {name:11s} NaN={nan} max |qd| {maxv:5.1f} rad/s;{extra} -> {'PASS' if not nan and maxv < 50 else 'FAIL'}")
