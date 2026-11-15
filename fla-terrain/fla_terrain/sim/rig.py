"""Foot-only rig (mode 'rig'): two K1 feet with the same sole zones and contact parameters, pressed onto a
terrain with randomized load, tilt and sliding velocity, like hand tests of the real sole.

Each foot: carrier body on x/y/z slides (carries the load mass) -> foot body (named like the K1 foot) on pitch
and roll hinges at the ankle, held by soft position servos (so the sole can conform or stand on an edge).
x/y slides are driven by velocity servos (force-limited, so high friction can make the foot stick).
No robot proprioception: joint arrays are zeros, gravity (0, 0, -1), command zeros.
"""
from __future__ import annotations

import mujoco
import numpy as np

from ..schema import DEFAULT_J
from . import k1
from .build import add_terrain, terrain_regions
from .sole import SoleConfig, add_sole

ANKLE_H = -k1.FOOT_BOX_POS[2] + k1.FOOT_BOX_HALF[2]  # ankle above sole bottom (0.038 m)


def build_foot_rig(terrain_cfg: dict, sole_cfg: SoleConfig | None = None, physics_dt: float = 0.002,
                   load_kg=(10.0, 10.0), y_sep: float = 0.15, v2_params=None, tilt_kp: float = 20.0, lock_tilt: bool = False):
    """v2_params (pointmap.SoleV2Params): build the v2 block sole and attach the point layout for the recorder."""
    if v2_params is not None:
        sole_cfg = SoleConfig(version="v2", boundaries=tuple(v2_params.boundaries),
                              mid_clearance=v2_params.mid_clearance, point_mask=v2_params.mask)
    sole_cfg = sole_cfg or SoleConfig()
    base = mujoco.MjSpec.from_file(k1.K1_XML)
    bm = base.compile()
    fb = mujoco.mj_name2id(bm, mujoco.mjtObj.mjOBJ_BODY, k1.FOOT_BODIES["L"])
    fg = [g for g in range(bm.ngeom) if bm.geom_bodyid[g] == fb and bm.geom_contype[g]][0]
    spec = mujoco.MjSpec()
    spec.option.timestep = physics_dt
    for k in ("impratio", "tolerance", "cone", "solver", "integrator", "iterations"):
        setattr(spec.option, k, getattr(base.option, k))
    tg = add_terrain(spec, terrain_cfg)
    for s, y, load in (("L", y_sep, load_kg[0]), ("R", -y_sep, load_kg[1])):
        c = spec.worldbody.add_body(name=f"rig_{s}", pos=[0, y, 0.3])
        for ax, a in (("x", [1, 0, 0]), ("y", [0, 1, 0]), ("z", [0, 0, 1])):
            c.add_joint(name=f"rig_{s}_{ax}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=a, damping=5.0)
        c.mass = float(load)
        c.inertia = [0.01, 0.01, 0.01]
        c.explicitinertial = True
        f = c.add_body(name=k1.FOOT_BODIES[s], pos=[0, 0, 0])
        f.mass = 0.5
        f.inertia = [0.0012, 0.0012, 0.00025]
        f.explicitinertial = True
        if not lock_tilt:  # lock_tilt: foot held exactly flat (no hinges, no tilt servos)
            f.add_joint(name=f"rig_{s}_pitch", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-0.6, 0.6],
                        damping=0.5)
            f.add_joint(name=f"rig_{s}_roll", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], range=[-0.4, 0.4],
                        damping=0.5)
        f.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(k1.FOOT_BOX_POS), size=list(k1.FOOT_BOX_HALF),
                   friction=list(bm.geom_friction[fg]), solref=list(bm.geom_solref[fg]),
                   solimp=list(bm.geom_solimp[fg]), condim=int(bm.geom_condim[fg]))
        for ax in ("x", "y"):
            a = spec.add_actuator(name=f"v_{s}_{ax}", target=f"rig_{s}_{ax}", trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.set_to_velocity(kv=2000.0)
            a.forcelimited = True
            a.forcerange = [-150, 150]
        for ax in (() if lock_tilt else ("pitch", "roll")):
            a = spec.add_actuator(name=f"p_{s}_{ax}", target=f"rig_{s}_{ax}", trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.set_to_position(kp=tilt_kp, kv=0.5 * tilt_kp / 20.0)
        add_sole(spec, s, sole_cfg)
    m = spec.compile()
    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
    regions, greg = terrain_regions(terrain_cfg)
    info = {
        "rig": True, "physics_dt": physics_dt, "terrain_cfg": terrain_cfg, "sole_cfg": sole_cfg,
        "zone_geoms": {s: [gid(f"sole_{s}_{n}") for n in sole_cfg.region_names()] for s in "LR"},
        "terrain_geoms": [gid(n) for n in tg], "terrain_regions": regions,
        "terrain_geom_region": dict(zip([gid(n) for n in tg], greg)),
        "foot_bodies": {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, k1.FOOT_BODIES[s]) for s in "LR"},
        "base_body": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "rig_L"),
        "carrier": {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"rig_{s}") for s in "LR"},
        "joint_names": list(k1.JOINT_NAMES), "load_kg": list(load_kg),
    }
    if v2_params is not None:
        info["point_layout"] = sole_cfg.point_layout()
        info["sole_v2_params"] = v2_params
    return m, info


def rig_proprio(data):
    z = np.zeros(DEFAULT_J)
    return z, z, np.zeros(3), np.array([0.0, 0.0, -1.0])


class RigMotion:
    """Random rig protocol: phases of 1.5-4 s, each with a sliding velocity (40 % zero = press/stick),
    tilt targets, a load modulation and (15 %) a lift-off of one foot."""

    name = "foot_rig"
    physics_dt = 0.002

    def __init__(self, info, rng, duration: float):
        self.info = info
        self.rng = rng
        self.phases = []
        t = 0.5
        while t < duration:
            dur = float(rng.uniform(1.5, 4.0))
            ph = {"t": t, "dur": dur}
            for s in "LR":
                v = 0.0 if rng.random() < 0.4 else float(rng.uniform(0.02, 0.25))
                ang = rng.uniform(-np.pi, np.pi)
                ph[s] = {"v": [v * np.cos(ang), v * np.sin(ang)], "tilt": rng.uniform([-0.25, -0.15], [0.25, 0.15]).tolist(),
                         "load_mod": float(rng.uniform(-0.3, 0.6)), "lift": bool(rng.random() < 0.15)}
            self.phases.append(ph)
            t += dur

    def reset(self, model, data, rng=None):
        mujoco.mj_resetData(model, data)
        from .motion import terrain_height_gap
        mujoco.mj_forward(model, data)
        for s in "LR":
            c = self.info["carrier"][s]
            jz = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"rig_{s}_z")
            p = data.xpos[c] - [0, 0, ANKLE_H]  # sole bottom centre below the ankle
            gap = terrain_height_gap(model, data, self.info, np.array([p + [dx, dy, 0] for dx in (-0.06, 0.11)
                                                                       for dy in (-0.03, 0.03)])).min()
            data.qpos[model.jnt_qposadr[jz]] = -(gap - 0.002)
        mujoco.mj_forward(model, data)

    def _phase(self, t):
        for ph in reversed(self.phases):
            if t >= ph["t"]:
                return ph
        return None

    def pre_step(self, model, data, cmd_vel=None):
        t = data.time
        ph = self._phase(t)
        data.xfrc_applied[:] = 0
        per = model.nu // 2
        for i, s in enumerate("LR"):
            base = per * i
            if ph is None:
                data.ctrl[base:base + per] = 0
                continue
            p = ph[s]
            data.ctrl[base + 0], data.ctrl[base + 1] = p["v"]
            if per == 4:
                data.ctrl[base + 2], data.ctrl[base + 3] = p["tilt"]
            c = self.info["carrier"][s]
            w = model.body_subtreemass[c] * 9.81
            frac = np.clip((t - ph["t"]) / 0.3, 0, 1)
            fz = p["load_mod"] * w * frac * np.sin(np.pi * min(1.0, (t - ph["t"]) / ph["dur"]))
            if p["lift"] and 0.3 * ph["dur"] < t - ph["t"] < 0.7 * ph["dur"]:
                fz = -1.3 * w  # lift the foot off
            data.xfrc_applied[c, 2] = -fz  # load_mod > 0 presses down

    def fallen(self, model, data):
        return False
