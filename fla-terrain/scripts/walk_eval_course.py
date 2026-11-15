#!/usr/bin/env python
"""B8 part 2: walking sources on the mixed-terrain course (walking/course.py), the training implementation's sole model.

  python scripts/walk_eval_course.py --sources G113_m2200,G80_m4200,SONIC --courses 20 [--seconds 60]

Velocity policies: 1 s stand, then vx = v_nominal (0.35 m/s), heading hold (wz = 0). SONIC: straight clips
(ClipLibrary.straight()), heading +x, cmd ignored. A run finishes if it walks `seconds` (after the 1 s stand)
without falling (MotionSource.fallen) and without leaving the course (|y| > 4 m or base below z = -0.3).
Fall location = chunk under the base x at the fall. Same course seeds for every source.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fla_terrain.sim import build_model  # noqa: E402
from fla_terrain.sim.sole import SoleConfig  # noqa: E402
from walking.course import HALF_WIDTH, START_PAD, chunk_at, make_course  # noqa: E402
from walking.k1_velocity import yaw_of  # noqa: E402

K_YAW, WZ_MAX, K_Y, VY_MAX = 1.0, 0.5, 0.5, 0.2   # command-side heading hold (operator with a joystick)

OUT = "/path/to/fla-terrain/b_runs/k1v2/course_eval"
_S = {}


def source(name):
    if name not in _S:
        if name == "SONIC":
            from walking.sonic_k1 import ClipLibrary, SonicK1
            lib = ClipLibrary()
            _S[name] = SonicK1(clip_names=lib.straight())
        else:
            from walking.k1_velocity import K1VelocityPolicy
            _S[name] = K1VelocityPolicy(name)
    return _S[name]


def run(args):
    name, seed, seconds, sole, auto_t, hold = args
    cfg = make_course(seed)
    if auto_t:   # as gen_data.py course_auto_T: stand 1 s + start pad + planned chunks + 5 s (total, incl. stand)
        seconds = START_PAD / cfg["v_nominal"] + cfg["params"]["planned_s"] + 5.0
    m, info = build_model(cfg, sole_cfg=SoleConfig(version=sole))
    d = mujoco.MjData(m)
    src = source(name)
    rng = np.random.default_rng(1000 + seed)
    if name == "SONIC":
        src.reset(m, d, rng, base_yaw=0.0)
        d.qpos[0] += cfg["spawn"][0]
        mujoco.mj_forward(m, d)
        stand = 0.0
    else:
        src.reset(m, d, rng, base_pos=(cfg["spawn"][0], 0.0, 0.58), base_yaw=0.0)
        stand = 1.0
    dt = m.opt.timestep
    n = int(round((seconds + stand) / dt))
    dec = int(round(0.02 / dt))
    cmd = np.array([cfg["v_nominal"], 0.0, 0.0])
    crossed_t = None
    res = {"source": name, "seed": seed, "finished": True, "hold": hold, "seconds": seconds, "t_fail": None, "why": None, "fail_class": None,
           "fail_variant": None, "order": cfg["params"]["order"]}
    for i in range(n):
        t = i * dt
        if hold and t >= stand and i % dec == 0 and name != "SONIC":
            yaw = yaw_of(d.qpos[3:7])
            cmd = np.array([cfg["v_nominal"], float(np.clip(-K_Y * d.qpos[1], -VY_MAX, VY_MAX)),
                            float(np.clip(-K_YAW * yaw, -WZ_MAX, WZ_MAX))])
        src.pre_step(m, d, cmd if t >= stand else np.zeros(3))
        mujoco.mj_step(m, d)
        if (i + 1) % dec:
            continue
        if crossed_t is None and d.qpos[0] >= cfg["course_end_x"]:
            crossed_t = t - stand
        why = None
        if src.fallen(m, d):
            why = "fall"
        elif abs(d.qpos[1]) > HALF_WIDTH or d.qpos[2] < -0.3:
            why = "off_course"
        if why:
            c = chunk_at(cfg, float(d.qpos[0]))
            res.update(finished=False, t_fail=round(t - stand, 2), why=why,
                       fail_class=(c["class"] if c else ("start" if d.qpos[0] < cfg["chunks"][0]["x0"] else "end")),
                       fail_variant=(c.get("variant") if c else None))
            break
    x = float(d.qpos[0])
    res.update(crossed_all=crossed_t is not None and (res["t_fail"] is None or crossed_t <= res["t_fail"]),
               t_crossed=crossed_t, x_final=x, y_final=float(d.qpos[1]), speed=(x - cfg["spawn"][0]) / max(t - stand, 1e-6),
               chunks_crossed=int(sum(b <= x for b in cfg["boundaries"][1:])), n_chunks=len(cfg["chunks"]),
               classes_entered=sorted({c["class"] for c in cfg["chunks"] if c["x0"] <= x}))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="G113_m2200,G80_m4200,SONIC")
    ap.add_argument("--courses", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--procs", type=int, default=40)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--sole", default="v1", help="the training implementation's sole version: v1 (2x4 zones) or v2 (blocks)")
    ap.add_argument("--auto-T", action="store_true", help="duration per course as the recorded course runs")
    ap.add_argument("--heading-hold", action="store_true", help="command-side yaw + lateral correction")
    a = ap.parse_args()
    jobs = [(s, a.seed0 + k, a.seconds, a.sole, a.auto_T, a.heading_hold) for s in a.sources.split(",") for k in range(a.courses)]
    with Pool(min(a.procs, len(jobs))) as p:
        rows = p.map(run, jobs, chunksize=1)
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(os.path.join(OUT, f"{a.tag}.json"), "w"), indent=0)
    print(f"[course] sole {a.sole}, {a.courses} courses x {a.seconds:.0f} s, seeds {a.seed0}..{a.seed0 + a.courses - 1}")
    print(f"heading hold: {a.heading_hold}; duration: {'auto (as recorded runs)' if a.auto_T else f'{a.seconds:.0f} s'}")
    print(f"{'source':11s} {'no-fail':>9s} {'crossed':>8s} {'speed':>6s} {'chunks':>7s} | failures by chunk type (fall / off course)")
    for s in a.sources.split(","):
        rs = [r for r in rows if r["source"] == s]
        fails = Counter((r["fail_class"] + (f"-{r['fail_variant']}" if r["fail_variant"] else ""), r["why"])
                        for r in rs if not r["finished"])
        fl = ", ".join(f"{k[0]} {v}" + ("" if k[1] == "fall" else " (off)") for k, v in sorted(fails.items()))
        print(f"{s:11s} {sum(r['finished'] for r in rs):3d} of {len(rs):2d} {sum(r['crossed_all'] for r in rs):4d}/{len(rs):2d} {np.mean([r['speed'] for r in rs]):6.3f} "
              f"{np.mean([r['chunks_crossed'] for r in rs]):4.1f}/{np.mean([r['n_chunks'] for r in rs]):.0f} | {fl or '-'}")
    exposure = defaultdict(int)
    for r in rows:
        for c in r["classes_entered"]:
            exposure[(r["source"], c)] += 1
    print("runs that entered each chunk type: " + "; ".join(
        f"{s}: " + " ".join(f"{c} {exposure[(s, c)]}" for c in ("hard_grip", "hard_slip", "soft", "rough", "slope", "edge"))
        for s in a.sources.split(",")))


if __name__ == "__main__":
    main()
