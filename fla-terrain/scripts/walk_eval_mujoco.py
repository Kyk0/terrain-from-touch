#!/usr/bin/env python
"""Headless batch evaluation of the K1 velocity walking policy in MuJoCo.

  flat (B1):  python scripts/walk_eval_mujoco.py flat  --policy G113_m2200 --episodes 64
  sweep (B3): python scripts/walk_eval_mujoco.py sweep --policy G113_m2200 --episodes 32
  one cond.:  python scripts/walk_eval_mujoco.py sweep --only mu0.2,bump2cm --episodes 4

Writes per-episode JSON to b_runs/eval/<mode>_<policy>.json and prints a summary table.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from walking import eval_terrains as T  # noqa: E402
from walking.evaluate import run_episode  # noqa: E402

OUT = "/path/to/fla-terrain/b_runs/eval"

# condition name -> (builder, kwargs)
SWEEP = {
    "flat_mu1.0": (T.flat, dict(friction=1.0)),
    "mu0.5": (T.flat, dict(friction=0.5)),
    "mu0.3": (T.flat, dict(friction=0.3)),
    "mu0.2": (T.flat, dict(friction=0.2)),
    **{f"soft_tc{tc}": (T.flat, dict(solref=(tc, 1.0), solimp=T.SOFT_SOLIMP))
       for tc in (0.05, 0.1, 0.2, 0.3)},
    "bump0.5cm": (T.bumps, dict(height=0.005)),
    "bump1cm": (T.bumps, dict(height=0.01)),
    "bump2cm": (T.bumps, dict(height=0.02)),
    **{f"slope{d}_{w}": (T.slope, dict(deg=d, direction=w))
       for w in ("up", "down", "side") for d in (5, 7.5, 10)},
    "edge_drop5cm": (T.platform, dict(drop=0.05)),
    "edge_drop10cm": (T.platform, dict(drop=0.10)),
    "strip30cm_h5cm": (T.strip, dict(width=0.30)),
    "strip20cm_h5cm": (T.strip, dict(width=0.20)),
}

_CACHE = {}


def _model(cond, seed):
    fn, kw = SWEEP[cond]
    key = (cond, seed if fn is T.bumps else 0)
    if key not in _CACHE:
        kw = dict(kw)
        if fn is T.bumps:
            kw["seed"] = seed % 8  # 8 bump layouts per height
        _CACHE[key] = fn(**kw)
    return _CACHE[key]


def flat_commands(n, rng):
    """75 % forward 0.3-0.5 m/s; the rest yaw, lateral and mixed commands."""
    cmds = []
    for i in range(n):
        r = i % 8
        if r < 6:
            cmds.append((rng.uniform(0.3, 0.5), 0.0, 0.0))
        elif r == 6:
            cmds.append((0.3, 0.0, rng.choice([-1, 1]) * rng.uniform(0.3, 0.5)))
        else:
            cmds.append((rng.uniform(0.0, 0.3), rng.choice([-1, 1]) * 0.2, 0.0))
    return cmds


def _job(a):
    cond, seed, cmd, policy, duration = a
    model, info = _model(cond, seed)
    if policy == "SONIC":  # clip playback, no command; one source object per worker process
        if "sonic" not in _CACHE:
            from walking.sonic_k1 import SonicK1
            _CACHE["sonic"] = SonicK1()
        r = run_episode(model, info, (0.0, 0.0, 0.0), duration=duration, seed=seed, warmup=0.0,
                        source=_CACHE["sonic"])
        r["clips"] = [c[1] for c in _CACHE["sonic"].clip_log]
    else:
        r = run_episode(model, info, cmd, duration=duration, policy=policy, seed=seed)
    r["speed"] = r["dist"] / max(r["t_end"], 1e-6)
    r["cond"] = cond
    return r


def summarize(rows, cond):
    rs = [r for r in rows if r["cond"] == cond]
    fr = np.mean([r["fell"] for r in rs])
    tf = [r["t_fall"] for r in rs if r["fell"]]
    fwd = [r for r in rs if r["cmd"][1] == 0 and r["cmd"][2] == 0]
    ratio = np.nanmean([r["vx"] / r["cmd"][0] for r in fwd if r["cmd"][0] > 0]) if fwd else float("nan")
    return dict(cond=cond, n=len(rs), fall_rate=float(fr),
                vx=float(np.nanmean([r["vx"] for r in fwd])) if fwd else float("nan"),
                cmd_vx=float(np.mean([r["cmd"][0] for r in fwd])) if fwd else float("nan"),
                track=float(ratio),
                med_t_fall=float(np.median(tf)) if tf else None,
                steps_s=float(np.mean([r["steps_per_s"] for r in rs])),
                steps=float(np.mean([r["touchdowns"] for r in rs])),
                partial_td=float(np.mean([r["partial_td"] for r in rs])),
                off_td=float(np.mean([r["off_td"] for r in rs])),
                bump_td=float(np.mean([r["bump_td"] for r in rs])),
                speed=float(np.mean([r["speed"] for r in rs])),
                reach=float(np.mean([(r["off_td"] + r["partial_td"]) > 0 for r in rs])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["flat", "sweep"])
    ap.add_argument("--policy", default="G113_m2200", help="G113_m2200 | G80_m4200 | SONIC")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--procs", type=int, default=40)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    rng = np.random.default_rng(12345)
    jobs = []
    if a.mode == "flat":
        n = a.episodes or 64
        for s, c in enumerate(flat_commands(n, rng)):
            jobs.append(("flat_mu1.0", s, c, a.policy, a.duration))
        conds = ["flat_mu1.0"]
    else:
        n = a.episodes or 32
        conds = a.only.split(",") if a.only else list(SWEEP)
        for cond in conds:
            for s in range(n):
                jobs.append((cond, s, (float(rng.uniform(0.3, 0.5)), 0.0, 0.0), a.policy, a.duration))
    t0 = time.time()
    with Pool(min(a.procs, len(jobs))) as p:
        rows = p.map(_job, jobs, chunksize=1)
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"{a.mode}_{a.policy}{a.tag}.json")
    json.dump(rows, open(out, "w"), indent=0)
    print(f"[eval] {a.mode} policy={a.policy} episodes={len(rows)} wall={time.time()-t0:.0f}s -> {out}")
    print(f"{'condition':18s} {'n':>3s} {'fall':>5s} {'vx':>6s} {'cmd':>5s} {'track':>5s} "
          f"{'t_fall_med':>10s} {'steps/s':>7s} {'steps':>6s} {'partialTD':>9s} {'offTD':>5s} {'bumpTD':>6s} {'speed':>6s} {'reach':>5s}")
    summ = []
    for c in conds:
        s = summarize(rows, c)
        summ.append(s)
        tf = f"{s['med_t_fall']:.1f}" if s["med_t_fall"] is not None else "-"
        print(f"{c:18s} {s['n']:3d} {s['fall_rate']:5.2f} {s['vx']:6.3f} {s['cmd_vx']:5.2f} "
              f"{s['track']:5.2f} {tf:>10s} {s['steps_s']:7.2f} {s['steps']:6.1f} {s['partial_td']:9.1f} {s['off_td']:5.1f} {s['bump_td']:6.1f} {s['speed']:6.3f} {s['reach']:5.2f}")
    json.dump(summ, open(out.replace(".json", "_summary.json"), "w"), indent=1)
    if a.mode == "flat":
        oth = [r for r in rows if r["cmd"][1] != 0 or r["cmd"][2] != 0]
        for r in oth[:4]:
            print(f"  cmd={np.round(r['cmd'], 2).tolist()} fell={r['fell']} vx={r['vx']:.2f} "
                  f"vy={r['vy']:.2f} wz={r['wz']:.2f}")
        fr = summ[0]["fall_rate"]
        print(f"[check B1] fall rate {fr:.3f} (< 0.333 required): {'PASS' if fr < 1/3 else 'FAIL'}")


if __name__ == "__main__":
    main()
