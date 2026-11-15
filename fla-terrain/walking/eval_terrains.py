"""Evaluation-only terrains for the walking policy (the independent evaluation). The dataset terrains live in
fla_terrain.sim (the training implementation); these exist only to measure where the policy can walk.

Every builder returns (MjModel, info) where info has:
  spawn: (x, y, z) base position, support(x, y) -> bool  (for edge / strip partial-support stats)
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .k1_velocity import K1_MJCF

BASE_Z = 0.58


def _spec():
    s = mujoco.MjSpec.from_file(K1_MJCF)
    ground = [g for g in s.geoms if g.name == "ground"][0]
    return s, ground


def _finish(s, friction, timestep=0.002):
    m = s.compile()
    m.opt.timestep = timestep
    m.geom_condim[:] = np.maximum(m.geom_condim, 3)
    m.geom_friction[:, 0] = friction  # MuJoCo combines with max(): set robot geoms too
    return m


def _box(s, pos, half, quat=(1, 0, 0, 0), name=""):
    g = s.worldbody.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = list(half)
    g.pos = list(pos)
    g.quat = list(quat)
    if name:
        g.name = name
    return g


SOFT_SOLIMP = (0.1, 0.95, 0.1, 0.5, 2.0)  # soft impedance ramp over 10 cm; default is (0.9, 0.95, 0.001)


def flat(friction=1.0, solref=None, solimp=None):
    s, ground = _spec()
    if solref is not None:
        ground.solref = list(solref)
    if solimp is not None:
        ground.solimp = list(solimp)
    return _finish(s, friction), {"spawn": (0.0, 0.0, BASE_Z), "support": None}


def bumps(height, friction=1.0, seed=0, n=900, area=(0.35, 12.0, -1.5, 1.5)):
    """Scattered rigid boxes, all of height `height`, 4-10 cm footprint, random yaw."""
    s, _ = _spec()
    rng = np.random.default_rng(seed)
    x0, x1, y0, y1 = area
    for i in range(n):
        hx, hy = rng.uniform(0.02, 0.05, 2)
        yaw = rng.uniform(0, math.pi)
        _box(s, (rng.uniform(x0, x1), rng.uniform(y0, y1), height / 2), (hx, hy, height / 2),
             (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)), name=f"bump{i}")
    return _finish(s, friction), {"spawn": (0.0, 0.0, BASE_Z), "support": None,
                                  "coverage": None}


def slope(deg, direction="up", friction=1.0):
    """Tilted ground plane through the origin. up: rises along +x; down: falls along +x;
    side: rises along +y (robot walks across the slope)."""
    s, ground = _spec()
    a = math.radians(deg)
    if direction == "up":       # rotate about -y so the normal tilts toward -x
        q = (math.cos(a / 2), 0, -math.sin(a / 2), 0)
    elif direction == "down":
        q = (math.cos(a / 2), 0, math.sin(a / 2), 0)
    elif direction == "side":   # rises along +y
        q = (math.cos(a / 2), math.sin(a / 2), 0, 0)
    else:
        raise ValueError(direction)
    ground.quat = list(q)
    return _finish(s, friction), {"spawn": (0.0, 0.0, BASE_Z + 0.02), "support": None}


def platform(drop, edge_x=1.0, friction=1.0):
    """Robot starts on a raised platform (top at z=drop) whose edge is edge_x ahead."""
    s, _ = _spec()
    L = 3.0
    _box(s, (edge_x - L, 0.0, drop / 2), (L, 3.0, drop / 2), name="platform")
    sup = lambda x, y: x <= edge_x  # noqa: E731
    return _finish(s, friction), {"spawn": (0.0, 0.0, BASE_Z + drop), "support": sup,
                                  "edge": ("x", edge_x), "top": drop}


def strip(width, height=0.05, length=14.0, friction=1.0):
    """Robot walks along a raised strip of the given width (narrow step, partial support)."""
    s, _ = _spec()
    _box(s, (length / 2 - 1.0, 0.0, height / 2), (length / 2, width / 2, height / 2), name="strip")
    sup = lambda x, y: abs(y) <= width / 2 and -1.0 <= x <= length - 1.0  # noqa: E731
    return _finish(s, friction), {"spawn": (0.0, 0.0, BASE_Z + height), "support": sup,
                                  "edge": ("y", width / 2), "top": height}


def static_sink(model, info, seconds=3.0):
    """Max contact penetration (m) of the K1 standing still: the 'stiffness' of a soft ground.
    K_eff = 193 N / sink is the whole-robot (both feet) effective vertical stiffness."""
    from .k1_velocity import K1VelocityPolicy
    d = mujoco.MjData(model)
    p = K1VelocityPolicy(init_noise=0.0)
    p.reset(model, d, np.random.default_rng(0), base_pos=info["spawn"])
    for _ in range(int(seconds / model.opt.timestep)):
        p.pre_step(model, d, (0.0, 0.0, 0.0))
        mujoco.mj_step(model, d)
    return max([-d.contact[c].dist for c in range(d.ncon)] or [0.0])
