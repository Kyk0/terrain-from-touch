"""Headless episode runner for the walking policy: falls, tracked speed, steps, edge contacts."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .k1_velocity import K1VelocityPolicy, quat_rotate_inverse

WEIGHT_N = 193.0
TD_ON, TD_OFF = 0.075 * WEIGHT_N, 0.035 * WEIGHT_N  # touchdown hysteresis (play.py values)
MIN_AIR_TICKS = 4  # foot must be unloaded >= 0.08 s (4 policy ticks) before the next touchdown counts


def _foot_geoms(model, foot_ids):
    out = []
    for b in foot_ids:
        g = [i for i in range(model.ngeom) if model.geom_bodyid[i] == b
             and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX and model.geom_contype[i]]
        out.append(g[0])
    return out


def _touches(model, data, body, geoms):
    for c in range(data.ncon):
        con = data.contact[c]
        if (model.geom_bodyid[con.geom1] == body and con.geom2 in geoms) or \
           (model.geom_bodyid[con.geom2] == body and con.geom1 in geoms):
            return True
    return False


def foot_loads(model, data, foot_ids):
    """Normal force (N) between each foot body and world (terrain) geoms."""
    f6 = np.zeros(6)
    load = [0.0, 0.0]
    for c in range(data.ncon):
        con = data.contact[c]
        b1, b2 = model.geom_bodyid[con.geom1], model.geom_bodyid[con.geom2]
        for k, fb in enumerate(foot_ids):
            if (b1 == fb and b2 == 0) or (b2 == fb and b1 == 0):
                mujoco.mj_contactForce(model, data, c, f6)
                load[k] += abs(f6[0])
    return load


_GX, _GY = np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 5))
_GRID = np.stack([_GX.ravel(), _GY.ravel()], 1)


def support_fraction(model, data, geom, support):
    half = model.geom_size[geom]
    R = data.geom_xmat[geom].reshape(3, 3)
    p = data.geom_xpos[geom]
    pts = p + (_GRID[:, :1] * half[0]) * R[:, 0] + (_GRID[:, 1:] * half[1]) * R[:, 1] - half[2] * R[:, 2]
    return float(np.mean([support(x, y) for x, y, _ in pts]))


def run_episode(model, info, cmd, duration=20.0, policy="G113_m2200", seed=0, warmup=1.0, source=None, **pkw):
    """source: an existing MotionSource (e.g. SonicK1); default builds K1VelocityPolicy(policy)."""
    rng = np.random.default_rng(seed)
    src = source if source is not None else K1VelocityPolicy(policy, **pkw)
    data = mujoco.MjData(model)
    if source is None:
        src.reset(model, data, rng, base_pos=info["spawn"])
    else:  # source keeps its own initial pose; lift it by the ground height under the spawn point
        src.reset(model, data, rng)
        data.qpos[2] += info["spawn"][2] - 0.58 if info.get("spawn") else 0.0
        mujoco.mj_forward(model, data)
    cmd = np.asarray(cmd, float)
    foot_ids = src.foot_ids
    fgeoms = _foot_geoms(model, foot_ids)
    support = info.get("support")
    dec = src.decimation
    n = int(round(duration / model.opt.timestep))
    state = [False, False]
    air = [0, 0]
    touchdowns, partial_td, td_fracs, bump_td, off_td = 0, 0, [], 0, 0
    bump_ids = {i for i in range(model.ngeom)
                if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("bump")}
    vxs, wzs = [], []
    t_fall = None
    start_xy = data.qpos[0:2].copy()
    for i in range(n):
        t = i * model.opt.timestep
        # warm-up: stand (zero command) so the robot settles on the terrain first
        src.pre_step(model, data, cmd if t >= warmup else np.zeros(3))
        mujoco.mj_step(model, data)
        if (i + 1) % dec:
            continue
        if src.fallen(model, data):
            t_fall = t
            break
        load = foot_loads(model, data, foot_ids)
        for k in range(2):
            if not state[k] and load[k] > TD_ON:
                state[k] = True
                if t >= warmup:
                    touchdowns += 1
                    if bump_ids:
                        bump_td += int(_touches(model, data, foot_ids[k], bump_ids))
                    if support is not None:
                        fr = support_fraction(model, data, fgeoms[k], support)
                        td_fracs.append(fr)
                        partial_td += int(0.05 < fr < 0.95)
                        off_td += int(fr <= 0.05)
            elif state[k]:
                air[k] = air[k] + 1 if load[k] < TD_OFF else 0
                if air[k] >= MIN_AIR_TICKS:
                    state[k], air[k] = False, 0
        if t >= warmup + 2.0:
            vb = quat_rotate_inverse(data.qpos[3:7], data.qvel[0:3])
            vxs.append(vb[:2].copy())
            wzs.append(data.qvel[5])
    t_end = t_fall if t_fall is not None else duration
    walk_t = max(t_end - warmup, 1e-6)
    v = np.mean(vxs, 0) if vxs else np.array([np.nan, np.nan])
    return {
        "seed": seed, "cmd": cmd.tolist(), "fell": t_fall is not None,
        "t_fall": t_fall, "t_end": t_end,
        "vx": float(v[0]), "vy": float(v[1]),
        "wz": float(np.mean(wzs)) if wzs else float("nan"),
        "dist": float(np.linalg.norm(data.qpos[0:2] - start_xy)),
        "x_final": float(data.qpos[0]), "y_final": float(data.qpos[1]),
        "touchdowns": touchdowns, "steps_per_s": touchdowns / walk_t,
        "partial_td": partial_td, "off_td": off_td, "bump_td": bump_td,
        "min_td_frac": float(min(td_fracs)) if td_fracs else None,
    }
