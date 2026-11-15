#!/usr/bin/env python
"""B8 part 3: every walking source on the training implementation's v2 foot (SoleConfig(version='v2')) vs the v1 sole, flat hard-grip.

  python scripts/walk_check_v2_foot.py [--episodes 64]
Same evaluation as B1 (walk_eval_mujoco.py flat): 64 eps x 20 s, mu 1.0, 75 % forward 0.3-0.5 m/s + yaw / lateral
commands, 1 s stand; SONIC: random clips, cmd ignored. Check per source: fall rate within 5 points of v1 and forward
speed within 10 % of v1 (velocity policies: body vx on forward-command episodes; SONIC: planar speed).
v1 reference = the same script on the v1 sole (same seeds), and the original B1 / A1 numbers.
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fla_terrain.sim import build_model  # noqa: E402
from fla_terrain.sim.sole import SoleConfig  # noqa: E402
from walking.evaluate import run_episode  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from walk_eval_mujoco import flat_commands  # noqa: E402

REF = {"G113_m2200": (0.0, 0.399, "B1 base MJCF"), "G80_m4200": (0.0, 0.396, "B1 base MJCF"),
       "SONIC": (0.0, 0.396, "A1 v1 sole, planar speed")}
_C = {}


def job(a):
    src_name, version, seed, cmd = a
    key = version
    if key not in _C:
        m, info = build_model({"type": "flat", "mu": 1.0}, sole_cfg=SoleConfig(version=version))
        _C[key] = (m, dict(info, spawn=(0.0, 0.0, 0.58), support=None))
    m, info = _C[key]
    if src_name == "SONIC":
        if "sonic" not in _C:
            from walking.sonic_k1 import SonicK1
            _C["sonic"] = SonicK1()
        r = run_episode(m, info, (0, 0, 0), duration=20.0, seed=seed, warmup=0.0, source=_C["sonic"])
    else:
        r = run_episode(m, info, cmd, duration=20.0, policy=src_name, seed=seed)
    r.update(src=src_name, version=version, speed=r["dist"] / max(r["t_end"], 1e-6))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--sources", default="G113_m2200,G80_m4200,SONIC")
    a = ap.parse_args()
    srcs = a.sources.split(",")
    cmds = flat_commands(a.episodes, np.random.default_rng(12345))
    jobs = [(s, v, k, c) for s in srcs for v in ("v1", "v2") for k, c in enumerate(cmds)]
    with Pool(40) as p:
        rows = p.map(job, jobs, chunksize=1)
    print(f"[B8 part 3] flat hard-grip mu 1.0, {a.episodes} eps x 20 s; v2 foot = SoleConfig(version='v2')")
    print(f"{'source':11s} {'sole':4s} {'fall':>5s} {'speed':>6s} {'steps/s':>7s} | check vs v1 sole (fall +-5 pts, speed +-10 %)")
    for s in srcs:
        res = {}
        for v in ("v1", "v2"):
            rs = [r for r in rows if r["src"] == s and r["version"] == v]
            fwd = [r for r in rs if r["cmd"][1] == 0 and r["cmd"][2] == 0 and r["cmd"][0] > 0]
            sp = np.mean([r["speed"] for r in rs]) if s == "SONIC" else np.nanmean([r["vx"] for r in fwd])
            res[v] = (np.mean([r["fell"] for r in rs]), sp, np.mean([r["steps_per_s"] for r in rs]))
        f1, s1, _ = res["v1"]
        f2, s2, st2 = res["v2"]
        ok = abs(f2 - f1) <= 0.05 and abs(s2 - s1) <= 0.1 * s1
        ref = REF.get(s, REF["G113_m2200"])
        okr = abs(f2 - ref[0]) <= 0.05 and abs(s2 - ref[1]) <= 0.1 * ref[1]
        for v in ("v1", "v2"):
            f, sp, st = res[v]
            tail = (f"fall {f2 - f1:+.3f}, speed {100 * (s2 / s1 - 1):+.1f} % -> {'PASS' if ok else 'FAIL'}; vs {ref[2]} "
                    f"(fall {ref[0]:.2f}, speed {ref[1]:.3f}): {'PASS' if okr else 'FAIL'}") if v == "v2" else ""
            print(f"{s:11s} {v:4s} {f:5.3f} {sp:6.3f} {st:7.2f} | {tail}")


if __name__ == "__main__":
    main()
