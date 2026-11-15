"""Ground-truth support fraction and support state per foot.

support_frac: a grid of points on the sole bottom is projected vertically onto the terrain (group 2,
ray cast down from 10 cm above each point). Each hit gives a surface height z_i and normal n_i.
The supporting surface at point i is h_i = max_j of the plane of hit j extrapolated to point i.
Point i is supported if z_i >= h_i - drop_tol (default 2.5 cm), i.e. it is not over a drop-off
(gap, step edge, platform edge). Foot tilt (heel/toe strike), slopes and bumps <= 2 cm do not reduce
it. It is geometric: computed whether or not the foot is loaded.

label: 0 airborne (contact_gt False: frame-mean true Fz <= 2 N), 1 partial (contact and frac < full_thresh),
2 full.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .build import TERRAIN_GROUP
from .sole import SoleConfig

RAY_UP = 0.10  # rays start this far above each sole point


class SupportProbe:
    def __init__(self, model, info, n_along: int = 8, n_across: int = 3, drop_tol: float = 0.025,
                 full_thresh: float = 0.9):
        self.m = model
        cfg: SoleConfig = info["sole_cfg"]
        hx, hy, _ = cfg.box_half
        xs = np.linspace(-hx, hx, n_along + 2)[1:-1] if n_along > 1 else np.zeros(1)
        ys = np.linspace(-hy, hy, n_across + 2)[1:-1] if n_across > 1 else np.zeros(1)
        o = cfg.sole_origin_body()
        self.pts_b = np.array([[o[0] + x, o[1] + y, o[2]] for x in xs for y in ys])  # foot body frame
        self.feet = info["foot_bodies"]
        self.tol = drop_tol
        self.full_thresh = full_thresh
        self.group = np.zeros(6, np.uint8)
        self.group[TERRAIN_GROUP] = 1
        self._gid = np.zeros(1, np.int32)
        self._n = np.zeros(3)
        self.down = np.array([0.0, 0.0, -1.0])
        self.params = {"grid": [n_along, n_across], "drop_tol_m": drop_tol, "full_thresh": full_thresh,
                       "method": "vertical rays, unsupported if > drop_tol below extrapolated neighbour surfaces"}

    def frac(self, data, side: str) -> float:
        b = self.feet[side]
        R = data.xmat[b].reshape(3, 3)
        pw = data.xpos[b] + self.pts_b @ R.T
        n = len(pw)
        z = np.full(n, -np.inf)
        nrm = np.tile([0.0, 0.0, 1.0], (n, 1))
        for i, p in enumerate(pw):
            start = p + [0.0, 0.0, RAY_UP]
            dist = mujoco.mj_ray(self.m, data, start, self.down, self.group, True, -1, self._gid, self._n)
            if dist >= 0:
                z[i] = start[2] - dist
                nrm[i] = self._n
        hit = np.isfinite(z)
        if not hit.any():
            return 0.0
        xy = pw[:, :2]
        nz = np.maximum(nrm[hit, 2], 0.2)
        # h[i] = max_j z_j - (n_j . (xy_i - xy_j)) / n_jz
        dxy = xy[:, None, :] - xy[None, hit, :]
        ext = z[None, hit] - (dxy * nrm[None, hit, :2]).sum(-1) / nz[None, :]
        h = ext.max(1)
        sup = hit & (z >= h - self.tol)
        return float(sup.mean())
    def center_drop(self, data, side: str) -> float:
        """Vertical distance from the sole centre down to the terrain (inf if nothing below)."""
        b = self.feet[side]
        R = data.xmat[b].reshape(3, 3)
        c = data.xpos[b] + R @ self.pts_b.mean(0)
        d = mujoco.mj_ray(self.m, data, c + [0.0, 0.0, RAY_UP], self.down, self.group, True, -1, self._gid)
        return float(d - RAY_UP) if d >= 0 else float("inf")

    def label(self, contact: bool, frac: float) -> int:
        if not contact:
            return 0
        return 2 if frac >= self.full_thresh else 1
