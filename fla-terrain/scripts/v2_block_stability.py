"""Are MuJoCo's per-block contact totals on the v2 foot trustworthy? (A6 question 1)

For a rigid foot with several coplanar contact blocks on flat rigid ground, the split of the load between blocks is
statically indeterminate; only the foot-level wrench (total force, CoP) is determined by the dynamics. This script
records, at every physics step (500 Hz), the per-block normal forces from MuJoCo and the foot-level CoP, standing
(PD stand, 6 s) and walking (G113, 20 s, 0.4 m/s), flat rigid ground, and reports for loaded steps (foot Fz > 20 N):
  - one-block dominance: fraction of steps with one block carrying > 80 % (and > 95 %) of the foot's Fz;
  - jumpiness: median |change of block shares| between consecutive physics steps (sum over blocks);
  - consistency with the CoP: a block split consistent with a uniform linear pressure field must put the
    front/back and left/right load ratios where the CoP is; reported as the mean |CoP_y from blocks (block centres)
    - CoP_y from contact points| and the same for x.
  $PY scripts/v2_block_stability.py [--sole v2]
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from fla_terrain.sim import SoleConfig, SoleSensor, StandMotionSource, build_model  # noqa: E402


def run(source, cfg, T):
    m, info = build_model({"type": "flat", "mu": 1.0}, sole_cfg=cfg)
    d = mujoco.MjData(m)
    if source == "stand":
        src = StandMotionSource(info)
        src.reset(m, d)
        cmd = None
    else:
        from walking.k1_velocity import K1VelocityPolicy
        src = K1VelocityPolicy("G113_m2200")
        src.reset(m, d, np.random.default_rng(0))
        cmd = (0.4, 0.0, 0.0)
    S = SoleSensor(m, cfg)
    cen = cfg.zone_centers()
    rows = []
    for k in range(int(T / m.opt.timestep)):
        src.pre_step(m, d, cmd)
        mujoco.mj_step(m, d)
        if k * m.opt.timestep < 1.5:
            continue
        r = S.read_full(d)
        for s in "LR":
            fz = r["F"][s][:, 2]
            rows.append((s, k, fz.copy(), r["cop"][s].copy()))
    return rows, cen


def stats(rows, cen, name):
    out = {}
    for s in "LR":
        R = [(k, fz, cop) for side, k, fz, cop in rows if side == s and fz.sum() > 20]
        if not R:
            continue
        ks = np.array([k for k, _, _ in R])
        fz = np.array([f for _, f, _ in R])
        cop = np.array([c for _, _, c in R])
        sh = fz / fz.sum(1, keepdims=True)
        cons = np.diff(ks) == 1
        jump = np.abs(np.diff(sh, axis=0)).sum(1)[cons]
        cop_blk = sh @ cen
        out[s] = (len(R), (sh.max(1) > 0.8).mean(), (sh.max(1) > 0.95).mean(), np.median(jump), np.percentile(jump, 99),
                  np.abs(cop_blk[:, 0] - cop[:, 0]).mean() * 1e3, np.abs(cop_blk[:, 1] - cop[:, 1]).mean() * 1e3)
    for s, v in out.items():
        print(f"{name:6s} {s}: loaded steps {v[0]}, one block >80 % {100*v[1]:.1f} % (>95 % {100*v[2]:.1f} %), "
              f"share jump per step median {v[3]:.4f} p99 {v[4]:.3f}, |CoP(blocks) - CoP(contacts)| x {v[5]:.1f} mm "
              f"y {v[6]:.1f} mm")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sole", default="v2")
    args = ap.parse_args()
    cfg = SoleConfig(version=args.sole)
    for src, T in (("stand", 6.0), ("walk", 20.0)):
        rows, cen = run(src, cfg, T)
        stats(rows, cen, src)


if __name__ == "__main__":
    main()
