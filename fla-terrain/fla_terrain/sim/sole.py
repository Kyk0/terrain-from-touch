"""Programmatic virtual sole: zone geoms tiling the foot collision box, and the zone force reader.

The base MJCF foot collision is one box per foot. ``add_sole`` replaces it by
``n_along x n_across`` boxes that tile exactly the same footprint and thickness, with the
same contact parameters. Body inertia is explicit in the MJCF, so the zone geoms add no
mass; kinematics are untouched.

Sole frame (schema): origin at the centre of the sole bottom, axes = foot body
(``*_ankle_roll_link``) axes: x forward (toe), y left, z up out of the ground.
Force = force exerted BY the ground ON the sole, so Fz >= 0 is load and Fx, Fy are shear.
Zone index z = i_along * n_across + i_across, i_along = 0 at the heel, i_across = 0 at -y.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from ..schema import zone_layout
from . import k1

ZONE_GROUP = 0  # collision geoms of the robot stay in group 0


V2_BLOCKS = ("back_r", "back_l", "mid", "front_r", "front_l")  # _r = -y half, _l = +y half (sole frame)
BOUNDARY_PRESETS = {  # (back part ends, front part starts) as fractions of the sole length from the heel
    "thirds": (1 / 3, 2 / 3),
    # K1 foot mesh (Left_Foot.STL, the MJCF visual mesh): flat heel to body x = -2 mm, groove 4-15 mm deep to
    # x = +14 mm, flat front after it (toe spring from ~+78 mm); in sole fractions (sole x from -64 to +116 mm):
    "mesh": ((-0.002 + 0.064) / 0.18, (0.014 + 0.064) / 0.18),
}


@dataclass
class SoleConfig:
    """version 'v1': n_along x n_across rigid zone boxes tiling the foot box (FLA-0 v1).
    version 'v2': rigid contact blocks back_r/back_l (heel part, split in halves), front_r/front_l, and a middle
    block raised by ``mid_clearance`` so it does not touch flat ground. Same outer footprint and height as the
    v1 foot box. ``boundaries`` = (back end, front start) as fractions of the sole length from the heel."""

    n_across: int = 2
    n_along: int = 4
    box_pos: tuple = k1.FOOT_BOX_POS
    box_half: tuple = k1.FOOT_BOX_HALF
    gap: float = 0.0  # gap between zone boxes [m]; 0 = exact tiling
    version: str = "v1"
    boundaries: tuple = BOUNDARY_PRESETS["thirds"]
    mid_clearance: float = 0.001  # m
    pitch: float = 0.010          # v2pads: point pitch (= PointLayout pitch)
    pad_gap: float = 0.0005       # v2pads: gap between neighbouring pads
    point_mask: str = "rect"      # PointLayout mask

    @property
    def Z(self) -> int:
        return len(self.region_names())

    def point_layout(self):
        from .pointmap import PointLayout
        return PointLayout(length=2 * self.box_half[0], width=2 * self.box_half[1], pitch=self.pitch,
                           boundaries=tuple(self.boundaries), mask=self.point_mask)

    def region_names(self) -> list[str]:
        if self.version == "v1":
            return [f"z{z}" for z in range(self.n_across * self.n_along)]
        if self.version == "v2pads":
            return [f"p{i}" for i in range(self.point_layout().P)]
        return list(V2_BLOCKS)

    def regions(self) -> list[tuple]:
        """[(name, centre (x, y, z) in sole frame, half sizes)], z = centre height above the sole bottom."""
        hx, hy, hz = self.box_half
        out = []
        if self.version == "v1":
            dx, dy = 2 * hx / self.n_along, 2 * hy / self.n_across
            for n, (cx, cy) in zip(self.region_names(), self.zone_centers()):
                out.append((n, (cx, cy, hz), (dx / 2 - self.gap / 2, dy / 2 - self.gap / 2, hz)))
            return out
        if self.version == "v2pads":  # cross-check sole: every sensing point is its own rigid pad (middle raised)
            L = self.point_layout()
            c, hp = self.mid_clearance, self.pitch / 2 - self.pad_gap / 2
            for n, (x, y), part in zip(self.region_names(), L.xy, L.part):
                cc = c if part == 1 else 0.0
                out.append((n, (x, y, hz + cc / 2), (hp, hp, hz - cc / 2)))
            return out
        b0 = -hx + 2 * hx * self.boundaries[0]
        b1 = -hx + 2 * hx * self.boundaries[1]
        c = self.mid_clearance
        for part, (xa, xb) in (("back", (-hx, b0)), ("front", (b1, hx))):
            for half, (ya, yb) in (("r", (-hy, 0.0)), ("l", (0.0, hy))):
                out.append((f"{part}_{half}", ((xa + xb) / 2, (ya + yb) / 2, hz), ((xb - xa) / 2, (yb - ya) / 2, hz)))
        out.insert(2, ("mid", ((b0 + b1) / 2, 0.0, hz + c / 2), ((b1 - b0) / 2, hy, hz - c / 2)))
        return out

    def layout(self) -> dict:
        hx, hy, hz = self.box_half
        if self.version == "v1":
            d = zone_layout(self.n_across, self.n_along)
        else:
            d = {"name": "v2_blocks", "Z": len(V2_BLOCKS), "blocks": list(V2_BLOCKS),
                 "boundaries": list(self.boundaries), "mid_clearance_m": self.mid_clearance}
        d.update({
            "frame": "sole: origin at sole bottom centre, x toe, y left, z up; force on sole from ground",
            "sole_length_m": 2 * hx, "sole_width_m": 2 * hy,
            "zone_centers_xy_m": [list(map(float, c)) for c in self.zone_centers()],
        })
        return d

    def zone_centers(self) -> np.ndarray:
        """[Z, 2] zone / block centres (x, y) in the sole frame."""
        hx, hy, _ = self.box_half
        if self.version != "v1":
            return np.array([r[1][:2] for r in self.regions()])  # v2 block centres / v2pads point positions
        dx, dy = 2 * hx / self.n_along, 2 * hy / self.n_across
        c = [(-hx + dx * (i + 0.5), -hy + dy * (j + 0.5)) for i in range(self.n_along) for j in range(self.n_across)]
        return np.array(c)

    def sole_origin_body(self) -> np.ndarray:
        """Sole frame origin in the foot body frame."""
        x0, y0, z0 = self.box_pos
        return np.array([x0, y0, z0 - self.box_half[2]])

    def to_dict(self) -> dict:
        return asdict(self)


def add_sole(spec: mujoco.MjSpec, side: str, cfg: SoleConfig) -> list[str]:
    """Replace the foot collision box of ``side`` ('L'/'R') with the sole's region boxes. Returns geom names."""
    body = spec.body(k1.FOOT_BODIES[side])
    boxes = [g for g in body.geoms if g.type == mujoco.mjtGeom.mjGEOM_BOX and (g.contype or g.conaffinity)]
    if len(boxes) != 1:
        raise RuntimeError(f"expected one collision box on {body.name}, found {len(boxes)}")
    src = boxes[0]
    if not np.allclose(src.pos, cfg.box_pos) or not np.allclose(src.size, cfg.box_half):
        raise RuntimeError(f"{body.name} collision box {list(src.pos)} {list(src.size)} differs from SoleConfig")
    props = dict(contype=src.contype, conaffinity=src.conaffinity, condim=src.condim, friction=list(src.friction),
                 solref=list(src.solref), solimp=list(src.solimp), margin=src.margin, gap=src.gap,
                 priority=src.priority, solmix=src.solmix)
    spec.delete(src)
    o = cfg.sole_origin_body()
    names = []
    for n, (cx, cy, cz), half in cfg.regions():
        name = f"sole_{side}_{n}"
        g = body.add_geom()
        g.name = name
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.pos = [o[0] + cx, o[1] + cy, o[2] + cz]
        g.size = list(half)
        g.group = ZONE_GROUP
        g.rgba = [0.2, 0.6, 0.9, 0.0]
        for k, v in props.items():
            setattr(g, k, v)
        names.append(name)
    return names


class SoleSensor:
    """Reads true (noise-free) per-zone forces in each sole frame from MuJoCo contacts.

    ``SoleSensor(model).read(data)`` -> {'L': [Z,3], 'R': [Z,3]} (Fx, Fy, Fz; N).
    ``read_full`` also returns per-foot CoP (sole frame, contact-point based) and contact flags.
    Requires a model from ``build_model`` (zone geoms named ``sole_{L,R}_z{i}``).
    """

    def __init__(self, model: mujoco.MjModel, sole_cfg: SoleConfig | None = None):
        self.m = model
        self.cfg = sole_cfg or SoleConfig()
        Z = self.cfg.Z
        self.Z = Z
        self.lut = -np.ones(model.ngeom, dtype=np.int64)  # geom -> side*Z + zone
        for si, side in enumerate("LR"):
            for z, rn in enumerate(self.cfg.region_names()):
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"sole_{side}_{rn}")
                if gid < 0:
                    raise RuntimeError(f"sole geom sole_{side}_{rn} not found; build the model with build_model")
                self.lut[gid] = si * Z + z
        self.foot_body = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, k1.FOOT_BODIES[s]) for s in "LR"]
        self.origin_b = self.cfg.sole_origin_body()
        self._f6 = np.zeros(6)

    def _contacts(self, data):
        n = data.ncon
        if n == 0:
            return None
        g = data.contact.geom[:n]
        z1, z2 = self.lut[g[:, 0]], self.lut[g[:, 1]]
        idx = np.nonzero((z1 >= 0) | (z2 >= 0))[0]
        if idx.size == 0:
            return None
        fw = np.zeros((idx.size, 3))
        for k, i in enumerate(idx):
            mujoco.mj_contactForce(self.m, data, int(i), self._f6)
            fw[k] = data.contact.frame[i].reshape(3, 3).T @ self._f6[:3]  # force of geom1 on geom2 (world)
        # force ON the sole geom: +fw if sole is geom2, -fw if sole is geom1
        on2 = z2[idx] >= 0
        fw[~on2] *= -1
        zone = np.where(on2, z2[idx], z1[idx])
        other = np.where(on2, g[idx, 0], g[idx, 1])
        return idx, zone, fw, data.contact.pos[idx], other

    def read(self, data) -> dict:
        return self.read_full(data)["F"]

    def read_full(self, data) -> dict:
        Z = self.Z
        F = np.zeros((2 * Z, 3))
        cop = np.full((2, 2), np.nan)
        contact = np.zeros(2, bool)
        touched = {"L": set(), "R": set()}
        wrench = {"L": None, "R": None}  # (total F in sole frame [3], CoP [2], Mz about the CoP)
        c = self._contacts(data)
        R = [data.xmat[b].reshape(3, 3) for b in self.foot_body]
        if c is not None:
            idx, zone, fw, pos, other = c
            side = zone // Z
            for s in (0, 1):
                sel = side == s
                if not sel.any():
                    continue
                fs = fw[sel] @ R[s]  # world -> sole frame (R columns are body axes in world)
                np.add.at(F, zone[sel], fs)
                contact[s] = True
                touched["LR"[s]].update(other[sel].tolist())
                fz = fs[:, 2]
                if fz.sum() > 1e-9:
                    o = data.xpos[self.foot_body[s]] + R[s] @ self.origin_b
                    p = (pos[sel] - o) @ R[s]
                    cop[s] = (fz[:, None] * p[:, :2]).sum(0) / fz.sum()
                    Ft = fs.sum(0)
                    mz = (p[:, 0] * fs[:, 1] - p[:, 1] * fs[:, 0]).sum() - (cop[s][0] * Ft[1] - cop[s][1] * Ft[0])
                    wrench["LR"[s]] = (Ft, cop[s].copy(), float(mz))
        return {"F": {"L": F[:Z], "R": F[Z:]}, "cop": {"L": cop[0], "R": cop[1]},
                "contact": {"L": bool(contact[0]), "R": bool(contact[1])}, "geoms": touched, "wrench": wrench}

    def zone_cop(self, F: np.ndarray) -> np.ndarray:
        """CoP (x, y) in the sole frame from zone forces [Z,3] using zone centres (what a real sole sees)."""
        fz = F[:, 2]
        if fz.sum() <= 1e-9:
            return np.full(2, np.nan)
        return (fz[:, None] * self.cfg.zone_centers()).sum(0) / fz.sum()
