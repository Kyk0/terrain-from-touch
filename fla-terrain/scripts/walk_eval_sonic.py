#!/usr/bin/env python
"""A1: SONIC-K1 clip playback in MuJoCo, chained walking clips, 20 s episodes.

  python scripts/walk_eval_sonic.py --episodes 32 [--conds sole_mu1.0_dt0.005,...]
Conditions: sole_* = the training implementation's build_model (2x4 sole, terrain priority 1); base_* = k1_play_kit MJCF via
walking/eval_terrains (all geoms set to mu). Speed = mean planar root speed (m/s) over the episode.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.environ.get("FLA_A_ROOT"):  # temporary: the training implementation tree with fla_terrain/data (not yet on main)
    sys.path.insert(0, os.environ["FLA_A_ROOT"])
from walking import eval_terrains as T  # noqa: E402
from walking.evaluate import run_episode  # noqa: E402
from walking.sonic_k1 import SonicK1  # noqa: E402

OUT = "/path/to/fla-terrain/b_runs/eval"


def build(cond):
    kind, mu, dt = cond.split("_")
    dt = float(dt[2:])
    if kind == "sole":
        from fla_terrain.sim import build_model
        m, info = build_model({"type": "flat", "mu": float(mu[2:])}, physics_dt=dt)
        return m, dict(info, spawn=(0.0, 0.0, 0.58), support=None)
    if kind == "base":
        m, info = T.flat(friction=float(mu[2:]))
    elif kind.startswith("bump"):
        m, info = T.bumps(height=float(kind[4:]) / 100, friction=float(mu[2:]))
    else:
        raise ValueError(cond)
    m.opt.timestep = dt
    return m, info


_C = {}


def job(a):
    cond, seed, dur = a
    if "src" not in _C:
        _C["src"] = SonicK1()
    if cond not in _C:
        _C[cond] = build(cond)
    m, info = _C[cond]
    src = _C["src"]
    src.physics_dt = m.opt.timestep
    r = run_episode(m, info, (0, 0, 0), duration=dur, seed=seed, warmup=0.0, source=src)
    r.update(cond=cond, clips=[c[1] for c in src.clip_log], speed=r["dist"] / max(r["t_end"], 1e-6))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--conds", default="sole_mu1.0_dt0.005,sole_mu1.0_dt0.002,base_mu1.0_dt0.005,"
                                        "sole_mu0.3_dt0.005,bump1_mu1.0_dt0.005")
    ap.add_argument("--procs", type=int, default=40)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    conds = a.conds.split(",")
    jobs = [(c, s, a.duration) for c in conds for s in range(a.episodes)]
    with Pool(min(a.procs, len(jobs))) as p:
        rows = p.map(job, jobs, chunksize=1)
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(os.path.join(OUT, f"sonic{a.tag}.json"), "w"), indent=0)
    print(f"[A1] SONIC-K1 it50000, chained walking clips, {a.episodes} eps x {a.duration:.0f} s")
    print(f"{'condition':22s} {'fall':>5s} {'t_fall_med':>10s} {'speed':>6s} {'vx':>6s} {'steps/s':>7s} "
          f"{'clips/ep':>8s} {'bumpTD':>6s}")
    for c in conds:
        rs = [r for r in rows if r["cond"] == c]
        tf = [r["t_fall"] for r in rs if r["fell"]]
        print(f"{c:22s} {np.mean([r['fell'] for r in rs]):5.2f} "
              f"{(f'{np.median(tf):.1f}' if tf else '-'):>10s} {np.mean([r['speed'] for r in rs]):6.3f} "
              f"{np.nanmean([r['vx'] for r in rs]):6.3f} {np.mean([r['steps_per_s'] for r in rs]):7.2f} "
              f"{np.mean([len(r['clips']) for r in rs]):8.1f} {np.mean([r['bump_td'] for r in rs]):6.1f}")
    fr = np.mean([r["fell"] for r in rows if r["cond"] == conds[0]])
    print(f"[check A1] {conds[0]} fall rate {fr:.3f} (< 0.333 required): {'PASS' if fr < 1/3 else 'FAIL'}")


if __name__ == "__main__":
    main()
