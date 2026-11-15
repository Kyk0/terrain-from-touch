"""Build the K1 MuJoCo model with the virtual sole and a terrain.

``build_model(terrain_cfg, sole_cfg=None) -> (mjModel, info)``

Terrain geoms get ``priority=1``, so for every foot-terrain contact MuJoCo uses the
terrain's friction / solref / solimp / condim (instead of the default max/mean mixing).
That is how terrain friction and stiffness are set without touching the robot.
Terrain geoms are in geom group 2 (used by the support-fraction ray casts).

terrain_cfg keys (all optional):
  type: 'flat' | 'slope'       (T4 adds more)
  mu: friction (default 1.0, the value the walking policy was trained/played with)
  slope_deg: for 'slope', ground rises along +x (robot faces uphill at yaw 0)
  solref, solimp: contact softness of the terrain (None = base MJCF default)
  label: schema terrain label id (default 0 hard-grip)
"""
from __future__ import annotations

import copy

import mujoco
import numpy as np

from . import k1
from .sole import SoleConfig, add_sole

TERRAIN_GROUP = 2


def _quat_y(angle: float) -> list:
    return [np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0]


def _quat_from_z(n) -> list:
    """Quaternion rotating +z onto unit vector n."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    ax = np.cross([0.0, 0.0, 1.0], n)
    s = np.linalg.norm(ax)
    if s < 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    ang = np.arctan2(s, n[2])
    ax = ax / s
    return [np.cos(ang / 2), *(np.sin(ang / 2) * ax)]


def slope_normal(slope_deg: float, uphill_dir: float = 0.0) -> np.ndarray:
    """Plane normal for a slope rising by slope_deg towards world azimuth uphill_dir [rad]."""
    t = np.deg2rad(slope_deg)
    return np.array([-np.sin(t) * np.cos(uphill_dir), -np.sin(t) * np.sin(uphill_dir), np.cos(t)])


def _legacy_geoms(cfg: dict) -> list[dict]:
    """terrain_cfg without 'geoms': a single plane (flat or slope)."""
    typ = cfg.get("type", "flat")
    g = {"type": "plane", "mu": cfg.get("mu", 1.0), "solref": cfg.get("solref"), "solimp": cfg.get("solimp"),
         "region": 0}
    if typ == "slope":
        g["normal"] = list(slope_normal(float(cfg["slope_deg"]), float(cfg.get("uphill_dir", 0.0))))
    elif typ != "flat":
        raise ValueError(f"terrain type {typ!r} needs a 'geoms' list")
    return [g]


def add_terrain(spec: mujoco.MjSpec, cfg: dict) -> list[str]:
    """Add terrain geoms. cfg['geoms'] = list of dicts with keys:
    type ('plane' | 'box'), pos, size (box half sizes), quat or normal (plane), mu, solref, solimp, region (int
    index into cfg['regions']). Without 'geoms', cfg describes one plane (type flat/slope, mu, slope_deg...)."""
    for g in list(spec.worldbody.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_PLANE:
            spec.delete(g)
    names = []
    for i, gd in enumerate(cfg.get("geoms") or _legacy_geoms(cfg)):
        g = spec.worldbody.add_geom()
        g.name = f"terrain_{i}"
        if gd["type"] == "plane":
            g.type = mujoco.mjtGeom.mjGEOM_PLANE
            g.size = [0, 0, 1]
            if gd.get("normal") is not None:
                g.quat = _quat_from_z(gd["normal"])
        elif gd["type"] == "box":
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = list(gd["size"])
            if gd.get("quat") is not None:
                g.quat = list(gd["quat"])
        else:
            raise ValueError(gd["type"])
        if gd.get("pos") is not None:
            g.pos = list(gd["pos"])
        g.friction = [float(gd.get("mu", 1.0)), 0.005, 0.0001]
        g.condim = 3
        g.priority = 1
        g.group = TERRAIN_GROUP
        g.rgba = [0.5, 0.5, 0.5, 1]
        if gd.get("solref") is not None:
            g.solref = list(gd["solref"])
        if gd.get("solimp") is not None:
            g.solimp = list(gd["solimp"])
        names.append(g.name)
    return names


def terrain_regions(cfg: dict) -> tuple[list[dict], list[int]]:
    """(regions, region index per terrain geom). Regions: label, mu, stiffness, slope (rad)."""
    geoms = cfg.get("geoms") or _legacy_geoms(cfg)
    regions = cfg.get("regions") or [{"label": int(cfg.get("label", 0)), "mu": float(cfg.get("mu", 1.0)),
                                       "stiffness": float(cfg.get("stiffness", np.nan)),
                                       "slope": float(np.deg2rad(cfg.get("slope_deg", 0.0)))}]
    return regions, [int(g.get("region", 0)) for g in geoms]


def add_base_rig(spec: mujoco.MjSpec, solref=(0.004, 1.0)) -> None:
    """Constrained-base rig: a mocap body welded to the trunk (move it via data.mocap_pos/quat)."""
    b = spec.worldbody.add_body()
    b.name = "rig_mocap"
    b.mocap = True
    b.pos = [0, 0, 1.0]  # = trunk pos in the MJCF, so the weld's relative pose is identity at qpos0
    eq = spec.add_equality()
    eq.name = "rig_weld"
    eq.type = mujoco.mjtEq.mjEQ_WELD
    eq.objtype = mujoco.mjtObj.mjOBJ_BODY
    eq.name1 = "rig_mocap"
    eq.name2 = k1.BASE_BODY
    eq.solref = list(solref)


def build_model(terrain_cfg: dict | None = None, sole_cfg: SoleConfig | None = None, xml: str = k1.K1_XML,
                physics_dt: float = 0.002, base_rig: bool = False):
    """Return (mjModel, info). ``info`` holds ids/names needed by SoleSensor, recorder and motion sources."""
    terrain_cfg = copy.deepcopy(terrain_cfg or {"type": "flat", "mu": 1.0})
    sole_cfg = sole_cfg or SoleConfig()
    spec = mujoco.MjSpec.from_file(xml)
    spec.option.timestep = physics_dt
    zones = {s: add_sole(spec, s, sole_cfg) for s in "LR"}
    terrain_geoms = add_terrain(spec, terrain_cfg)
    if base_rig:
        add_base_rig(spec)
    m = spec.compile()
    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
    jnames = [m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    assert tuple(jnames) == k1.JOINT_NAMES, "joint order changed"
    act_joints = [m.joint(m.actuator_trnid[i, 0]).name for i in range(m.nu)]
    info = {
        "xml": xml,
        "physics_dt": physics_dt,
        "terrain_cfg": terrain_cfg,
        "sole_cfg": sole_cfg,
        "zone_geoms": {s: [gid(n) for n in zones[s]] for s in "LR"},
        "terrain_geoms": [gid(n) for n in terrain_geoms],
        "terrain_regions": terrain_regions(terrain_cfg)[0],
        "terrain_geom_region": dict(zip([gid(n) for n in terrain_geoms], terrain_regions(terrain_cfg)[1])),
        "foot_bodies": {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, k1.FOOT_BODIES[s]) for s in "LR"},
        "base_body": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, k1.BASE_BODY),
        "joint_names": jnames,
        "actuator_joints": act_joints,
        "qpos_adr": np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in jnames]),
        "qvel_adr": np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in jnames]),
        "act_qpos_adr": np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]),
        "act_qvel_adr": np.array([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]),
        "mass": float(m.body_subtreemass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, k1.BASE_BODY)]),
        "base_rig": base_rig,
    }
    return m, info


def build_model_v2(terrain_cfg: dict | None = None, params=None, pads: bool = False, **kw):
    """K1 with sole v2: contact blocks (pads=False, map by reconstruction) or one rigid pad per sensing point
    (pads=True, cross-check). params: pointmap.SoleV2Params (nominal if None). info gains point_layout and
    sole_v2_params for the recorder."""
    from .pointmap import SoleV2Params
    prm = params or SoleV2Params()
    cfg = SoleConfig(version="v2pads" if pads else "v2", boundaries=tuple(prm.boundaries),
                     mid_clearance=prm.mid_clearance, point_mask=prm.mask)
    m, info = build_model(terrain_cfg, sole_cfg=cfg, **kw)
    info["point_layout"] = cfg.point_layout()
    info["sole_v2_params"] = prm
    return m, info
