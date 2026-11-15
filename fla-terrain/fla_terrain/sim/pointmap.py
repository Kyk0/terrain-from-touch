"""Sole v2 point map: foot-shaped sensing-point layout, reconstruction of the per-point force map from the
simulator's foot-level wrench, the sensor model, and the derived views (N_real, S_real, N_blk, S_blk).

Why reconstruction from the foot-level wrench (A6 question 1, scripts/v2_block_stability.py): with a rigid foot of
several coplanar blocks on rigid ground, MuJoCo's split of the load between blocks is statically indeterminate,
and while walking the rigid foot rocks and its contacts flicker at 500 Hz (one block > 95 % of the load in 33-38 %
of loaded steps; block shares jump by up to 2.0 between consecutive 2 ms steps). What the dynamics determine is the
foot-level wrench: total force, centre of pressure, torsion about the CoP. So, at every physics step:

  1. foot wrench in the sole frame from all contacts of the foot: F = (Fx, Fy, Fz), CoP (cx, cy), Mz about the CoP;
  2. per point i: gap g_i along the sole normal from the point (on the sole surface; middle points sit higher by the
     middle clearance) to the terrain (analytic ray cast against the terrain planes/boxes; negative = penetration);
  3. thin elastic layer on a rigid plane (Winkler): p_i = k * max(0, h + a x_i + b y_i - g_i); the plane (h, a, b)
     is solved so that sum p_i = Fz and the CoP of p equals (cx, cy) exactly (Newton on the active set);
     k = layer stiffness per point (from the 'compliance' parameter: mean compression under 100 N spread over the
     load-bearing points);
  4. per-point shear: translation part (Fx, Fy) * p_i / Fz plus a rotation part kappa * p_i * (-(y_i - cy), x_i - cx)
     with kappa so that the torsion about the CoP equals Mz. APPROXIMATION: real shear at a point depends on the
     local stick/slip state and elastomer; here it is shared in proportion to local pressure.
  The sensor integrates: the 150 Hz sample is the mean of the per-step maps in its 1/150 s bin.
Then the sensor model per 150 Hz sample: neighbour spread (Gaussian, mass preserving inside the mask), crosstalk
(row/column leakage), per-point gain and offset, white noise, quantization; then 50 Hz = mean of 3 samples.
Blocks (N_blk, S_blk) are DERIVED from the point map (sums over the points of each block), not taken from MuJoCo's
per-block contact forces.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np

from .build import TERRAIN_GROUP

PARTS = ("back", "mid", "front")
BLOCKS = ("back_r", "back_l", "mid", "front_r", "front_l")
BLOCK4 = ("front_l", "front_r", "back_l", "back_r")  # order of the 4-block views N_blk / S_blk


# ------------------------------------------------------------------------------------------------- layout
def foot_halfwidth_fraction(s):
    """Half-width of the foot outline as a fraction of the sole half-width, s = 0 heel .. 1 toe.
    Shape traced from the real PaXini layout screenshot (docs/real_sole_layout_g1.png): rounded heel, slightly
    narrower arch, widest at the ball, rounded toe."""
    S = [0.0, 0.05, 0.15, 0.35, 0.50, 0.62, 0.80, 0.90, 0.96, 1.0]
    F = [0.50, 0.78, 0.86, 0.82, 0.88, 1.00, 1.00, 0.88, 0.70, 0.45]
    return np.interp(s, S, F)


@dataclass
class PointLayout:
    length: float = 0.180
    width: float = 0.070
    pitch: float = 0.010
    boundaries: tuple = (1 / 3, 2 / 3)  # back end, front start (fractions of length from the heel)
    # mask 'rect': grid over the whole simulated contact footprint (the MJCF foot box) with the outer rows ON the
    #   box edges (19 x 8 at 10 mm for 180 x 70 mm). Needed because the rigid box contacts at its edges and corners;
    #   a point map that does not reach them cannot reproduce the simulator's CoP (check N2). The K1 foot mesh is
    #   itself nearly rectangular (bottom 57-68 mm wide at the heel, 70-80 mm at the front).
    # mask 'g1': foot outline traced from the PaXini G1 screenshot, cell-centred grid (~106 points; cannot reach the
    #   box corners).
    # mask 'rect_g1': the 'rect' grid (same 152 points, same array shape) with the points outside the G1 foot
    #   outline marked absent: they always read 0; the CoP is projected into the hull of the present points.
    mask: str = "rect"

    def __post_init__(self):
        if self.mask in ("rect", "rect_g1"):
            nx = int(round(self.length / self.pitch)) + 1
            ny = int(round(self.width / self.pitch)) + 1
            xs = np.linspace(-self.length / 2, self.length / 2, nx)
            ys = np.linspace(-self.width / 2, self.width / 2, ny)
        else:
            nx = int(round(self.length / self.pitch))
            ny = int(round(self.width / self.pitch))
            xs = -self.length / 2 + self.pitch * (np.arange(nx) + 0.5)
            ys = -self.width / 2 + self.pitch * (np.arange(ny) + 0.5)
        pts, rc = [], []
        for i, x in enumerate(xs):
            hw = self.width / 2 if self.mask in ("rect", "rect_g1") else \
                foot_halfwidth_fraction((x + self.length / 2) / self.length) * self.width / 2
            for j, y in enumerate(ys):
                if abs(y) <= hw + 1e-9:
                    pts.append((x, y))
                    rc.append((i, j))
        self.xy = np.array(pts)                       # [P, 2] sole frame (m)
        self.rowcol = np.array(rc)                    # [P, 2] grid indices
        self.P = len(pts)
        s = (self.xy[:, 0] + self.length / 2) / self.length
        self.part = np.where(s < self.boundaries[0], 0, np.where(s < self.boundaries[1], 1, 2))  # 0 back 1 mid 2 front
        # half: 0 = -y ('_r'), 1 = +y ('_l'), 2 = on the centre line (counts 1/2 to each half in the block views)
        self.half = np.where(np.abs(self.xy[:, 1]) < 1e-9, 2, (self.xy[:, 1] > 0).astype(int))
        W = np.zeros((self.P, len(BLOCKS)))  # block membership weights [P, 5] (rows sum to 1)
        for i, (p, h) in enumerate(zip(self.part, self.half)):
            if p == 1:
                W[i, 2] = 1.0
            else:
                r, l = (0, 1) if p == 0 else (3, 4)
                W[i, r] += 1.0 if h == 0 else (0.5 if h == 2 else 0.0)
                W[i, l] += 1.0 if h == 1 else (0.5 if h == 2 else 0.0)
        self.block_w = W
        self.block = W.argmax(1)
        if self.mask == "rect_g1":
            hw = foot_halfwidth_fraction((self.xy[:, 0] + self.length / 2) / self.length) * self.width / 2
            self.present = np.abs(self.xy[:, 1]) <= hw + 1e-9
        else:
            self.present = np.ones(self.P, bool)

    def table(self) -> dict:
        return {"name": "foot_points_v2", "mask": self.mask, "P": int(self.P), "Z": int(self.P), "pitch_m": self.pitch,
                "length_m": self.length, "width_m": self.width, "boundaries": list(self.boundaries),
                "points_xy_m": self.xy.round(5).tolist(), "rowcol": self.rowcol.tolist(),
                "part": self.part.tolist(), "half": self.half.tolist(), "present": self.present.astype(int).tolist(), "block_weights": self.block_w.tolist(),
                "parts": list(PARTS), "halves": ["r (-y)", "l (+y)", "centre line (1/2 to each half)"],
                "blocks": list(BLOCKS),
                "frame": "sole: origin at sole bottom centre, x toe, y left, z up; force on sole from ground"}


# ------------------------------------------------------------------------------------------------- params
@dataclass
class SoleV2Params:
    """Unknown properties of the real sensor; nominal values and per-episode randomization ranges."""
    boundaries: tuple = (1 / 3, 2 / 3)
    mid_clearance: float = 0.001        # m
    compliance: float = 0.0002          # m, mean layer compression under 100 N spread over the load-bearing points
                                        # (0.5 mm in A6/A7; lowered in A8: thin rigid sole, middle off flat ground)
    spread_sigma: float = 0.005         # m, neighbour spread (Gaussian sigma) of the rubber layer
    crosstalk: float = 0.02             # fraction leaked to the same row and column
    gain_std: float = 0.03
    offset_std: float = 0.1             # N
    noise_normal: float = 0.25          # N per point (v1 placeholder)
    noise_shear: float = 0.15           # N per point (v1 placeholder)
    quant: float = 0.05                 # N
    sat_normal: float = 60.0            # N per point
    sat_shear: float = 30.0             # N per point
    nominal: bool = True
    mask: str = "rect"                  # point mask: 'rect' (primary) or 'rect_g1' (foot outline variant)
    view_boundaries: tuple | None = None  # part boundaries used for the part/half/block VIEW (None = boundaries);
                                          # lets a sensor setting assume other blocks without changing the foot
    setting: str = "nominal"            # nominal | shifted | random | (pads / g1 are recorded by the builder)

    RANGES = {  # per-episode randomization (uniform); boundaries span thirds and the K1 foot mesh groove
        "back_end": (0.30, 0.37), "front_start": (0.42, 0.70), "mid_clearance": (0.0005, 0.002),
        "compliance": (0.0001, 0.0004), "spread_sigma": (0.0, 0.010), "crosstalk": (0.0, 0.05),
        "gain_std": (0.01, 0.10), "offset_std": (0.0, 0.3), "noise_scale": (0.5, 2.0), "quant": (0.01, 0.2),
    }

    @classmethod
    def sample(cls, rng) -> "SoleV2Params":
        r = {k: float(rng.uniform(*v)) for k, v in cls.RANGES.items()}
        ns = r.pop("noise_scale")
        return cls(boundaries=(r.pop("back_end"), r.pop("front_start")), noise_normal=0.25 * ns,
                   noise_shear=0.15 * ns, nominal=False, setting="random", **r)

    @classmethod
    def shifted(cls) -> "SoleV2Params":
        """Fixed 'shifted' setting for the sensitivity test. SENSOR-SIDE ONLY (the simulated foot is the nominal
        one, so nominal / shifted / mask readouts can be recorded from the same simulation): blocks assumed at the
        K1-mesh boundaries (16 mm middle groove: 0.344 / 0.433 of the length), twice the neighbour spread (10 mm),
        twice the noise, twice the layer compliance (0.4 mm per 100 N), crosstalk 0.05, gain std 0.06,
        quantization 0.1 N."""
        mesh = ((-0.002 + 0.064) / 0.18, (0.014 + 0.064) / 0.18)
        return cls(view_boundaries=mesh, spread_sigma=0.010, noise_normal=0.5, noise_shear=0.3, compliance=0.0004,
                   crosstalk=0.05, gain_std=0.06, quant=0.1, nominal=False, setting="shifted")

    def to_dict(self):
        d = asdict(self)
        d["boundaries"] = list(self.boundaries)
        d["view_boundaries"] = list(self.view_boundaries) if self.view_boundaries else None
        return d


# ------------------------------------------------------------------------------------------------- ray cast
class TerrainRaycaster:
    """Analytic ray casts against the terrain geoms (planes and boxes, group 2), vectorized over points.
    Boxes are pre-filtered by distance to the query centre."""

    def __init__(self, model, info):
        self.planes, self.boxes = [], []
        for g in info["terrain_geoms"]:
            t = model.geom_type[g]
            if t == mujoco.mjtGeom.mjGEOM_PLANE:
                self.planes.append(g)
            elif t == mujoco.mjtGeom.mjGEOM_BOX:
                self.boxes.append(g)
            else:
                raise ValueError(f"terrain geom type {t} not supported by the analytic ray cast")
        self.boxes = np.array(self.boxes, int)
        self.box_r = np.linalg.norm(model.geom_size[self.boxes], axis=1) if len(self.boxes) else np.zeros(0)
        self.box_half = model.geom_size[self.boxes] if len(self.boxes) else np.zeros((0, 3))

    def cast(self, data, origins, d, center, radius=0.25):
        """Distance t >= 0 along unit direction d from each origin to the first terrain hit (inf if none)."""
        n = len(origins)
        t = np.full(n, np.inf)
        for g in self.planes:
            N = data.geom_xmat[g].reshape(3, 3)[:, 2]
            dn = d @ N
            if abs(dn) < 1e-9:
                continue
            tt = ((data.geom_xpos[g] - origins) @ N) / dn
            t = np.where(tt >= 0, np.minimum(t, tt), t)
        if len(self.boxes):
            near = np.linalg.norm(data.geom_xpos[self.boxes] - center, axis=1) < radius + self.box_r
            for bi in np.nonzero(near)[0]:
                g = self.boxes[bi]
                R = data.geom_xmat[g].reshape(3, 3)
                o = (origins - data.geom_xpos[g]) @ R          # box frame
                dd = d @ R
                h = self.box_half[bi]
                with np.errstate(divide="ignore", invalid="ignore"):
                    inv = 1.0 / np.where(np.abs(dd) < 1e-12, 1e-12, dd)
                    t1 = (-h - o) * inv
                    t2 = (h - o) * inv
                tmin = np.minimum(t1, t2).max(1)
                tmax = np.maximum(t1, t2).min(1)
                hit = (tmax >= np.maximum(tmin, 0.0))
                th = np.where(tmin >= 0, tmin, tmax)  # origin inside the box: exit distance (rare)
                t = np.where(hit & (tmin >= 0), np.minimum(t, th), t)
        return t


# ------------------------------------------------------------------------------------------------- reconstruction
def solve_layer(xy, g, Fz, cop, k, iters=40, tol_cop=1e-5, lam0=None, return_lam=False):
    """Thin elastic layer on a rigid plane: p_i = k max(0, lam . (1, x_i, y_i) - g_i), with sum p = Fz and CoP(p) = cop.

    This p is the solution of the convex QP  min sum(p_i^2 / 2k + g_i p_i)  s.t.  sum p = Fz, sum p x = Fz cx,
    sum p y = Fz cy, p >= 0  (KKT conditions give the formula). Solved through its 3-dim concave dual by Newton with
    Armijo backtracking (globally convergent when the CoP is inside the hull of the points). If the CoP is on the
    hull boundary (e.g. exactly at a corner point) the dual is unbounded; then the remaining error is removed by a
    bounded least-squares projection onto {p >= 0, A p = t} starting from the Winkler map. Returns p [P] (N)."""
    P = len(g)
    if Fz <= 0:
        return np.zeros(P)
    gg = np.where(np.isfinite(g), g, 1e3)
    X = np.column_stack([np.ones(P), xy])                  # [P, 3]
    t = np.array([Fz, Fz * cop[0], Fz * cop[1]])

    def dual(lam):
        d = X @ lam - gg
        pp = k * np.maximum(d, 0.0)
        return lam @ t - 0.5 * (pp * np.maximum(d, 0.0)).sum(), pp

    # start: flat plane carrying Fz
    order = np.sort(gg)
    cs = np.cumsum(order)
    h = order[0] + Fz / k
    for m in range(1, P + 1):
        hm = (Fz / k + cs[m - 1]) / m
        if m == P or hm <= order[m]:
            h = hm
            break
    lam = np.array([h, 0.0, 0.0])
    D, p = dual(lam)
    if lam0 is not None:  # warm start (previous physics step of the same foot) if it is better
        D0, p0 = dual(np.asarray(lam0, float))
        if D0 > D:
            lam, D, p = np.asarray(lam0, float), D0, p0
    for _ in range(iters):
        grad = t - X.T @ p
        if abs(grad[0]) < 1e-6 * Fz and np.hypot(grad[1], grad[2]) < tol_cop * Fz:
            break
        act = p > 0
        H = k * X[act].T @ X[act] + 1e-12 * k * np.eye(3) if act.any() else k * np.eye(3) * 1e-6
        step = np.linalg.solve(H + 1e-9 * np.trace(H) * np.eye(3), grad)
        s_ = 1.0
        for _ in range(30):
            D2, p2 = dual(lam + s_ * step)
            if D2 >= D + 1e-4 * s_ * (grad @ step):
                break
            s_ *= 0.5
        lam, D, p = lam + s_ * step, D2, p2
    tot = p.sum()
    if tot <= 0:
        p = np.zeros(P)
    err = np.abs(X.T @ p - t)
    if err[0] > 1e-4 * Fz or np.hypot(err[1], err[2]) > 1e-4 * Fz:  # CoP on/near the hull boundary
        from scipy.optimize import lsq_linear
        w = 1e3
        A = np.vstack([np.eye(P), w * X.T / max(Fz, 1e-9)])
        b = np.concatenate([p, w * t / max(Fz, 1e-9)])
        p = lsq_linear(A, b, bounds=(0, np.inf), method="bvls").x
    return (p, lam) if return_lam else p


def distribute_shear(xy, p, Fxy, cop, Mz):
    """Per-point shear: translation part proportional to p, plus a rotation part matching the torsion Mz about the
    CoP (sum of the rotation part is zero because the CoP of p is cop)."""
    tot = p.sum()
    if tot <= 0:
        return np.zeros((len(p), 2))
    S = np.outer(p / tot, Fxy)
    r = xy - cop
    J = (p * (r ** 2).sum(1)).sum()
    if J > 1e-12 and abs(Mz) > 0:
        kap = Mz / J
        S += kap * p[:, None] * np.column_stack([-r[:, 1], r[:, 0]])
    return S


class PointMapReconstructor:
    """Per physics step, per foot: foot wrench (from the SoleSensor contacts) + gaps -> per-point force [P, 3]."""

    def __init__(self, model, info, layout: PointLayout, params: SoleV2Params):
        self.m, self.info, self.L, self.prm = model, info, layout, params
        self.ray = TerrainRaycaster(model, info)
        self.o_b = info["sole_cfg"].sole_origin_body()
        self.feet = info["foot_bodies"]
        self.raise_mid = np.where(layout.part == 1, params.mid_clearance, 0.0)
        n_bear = int((layout.part != 1).sum())
        self.k = 100.0 / (n_bear * params.compliance)       # N/m per point
        self.start = 0.05                                    # rays start 5 cm above the sole surface
        self.lam = {"L": None, "R": None}                    # warm starts
        self.pres = layout.present
        self.hull = None
        if not self.pres.all():
            from scipy.spatial import ConvexHull
            xy = layout.xy[self.pres]
            self.hull = xy[ConvexHull(xy).vertices]           # counter-clockwise polygon

    def project_cop(self, c):
        """Closest point of the present points' hull to c (c itself if inside)."""
        H = self.hull
        e = np.roll(H, -1, 0) - H
        inside = np.all(e[:, 0] * (c[1] - H[:, 1]) - e[:, 1] * (c[0] - H[:, 0]) >= 0)
        if inside:
            return c
        t = np.clip(((c - H) * e).sum(1) / (e ** 2).sum(1), 0, 1)
        q = H + t[:, None] * e
        return q[np.argmin(((q - c) ** 2).sum(1))]

    def gaps(self, data, side):
        b = self.feet[side]
        R = data.xmat[b].reshape(3, 3)
        o = data.xpos[b] + R @ self.o_b
        n = R[:, 2]
        pts = o + self.L.xy @ R[:, :2].T + self.raise_mid[:, None] * n
        t = self.ray.cast(data, pts + self.start * n, -n, o)
        return t - self.start                                 # gap along -n from the sole surface (neg = penetration)

    def step(self, data, wrench):
        """wrench: {side: (F[3] sole frame, cop[2], Mz) or None}. Returns {side: [P, 3]}."""
        out = {}
        for s in "LR":
            w = wrench.get(s)
            if w is None or w[0][2] <= 0:
                out[s] = np.zeros((self.L.P, 3))
                self.lam[s] = None
                continue
            F, cop, Mz = w
            g = self.gaps(data, s)
            if self.hull is not None:  # outline variant: only present points, CoP projected into their hull
                c2 = self.project_cop(cop)
                Mz = Mz + (cop[0] - c2[0]) * F[1] - (cop[1] - c2[1]) * F[0]
                cop = c2
                g = np.where(self.pres, g, np.inf)
            p, self.lam[s] = solve_layer(self.L.xy, g, F[2], cop, self.k, lam0=self.lam[s], return_lam=True)
            p = np.where(self.pres, p, 0.0)
            S = distribute_shear(self.L.xy, p, F[:2], cop, Mz)
            out[s] = np.column_stack([S, p])
        return out


# ------------------------------------------------------------------------------------------------- sensor model
class PointSensorModel:
    def __init__(self, layout: PointLayout, params: SoleV2Params, rng):
        self.L, self.prm, self.rng = layout, params, rng
        P = layout.P
        d2 = ((layout.xy[:, None, :] - layout.xy[None, :, :]) ** 2).sum(-1)
        pres = layout.present
        self.pres = pres
        d2[~pres] = np.inf                                     # nothing spreads into absent points
        if params.spread_sigma > 1e-6:
            K = np.exp(-d2 / (2 * params.spread_sigma ** 2))
            K /= K.sum(0, keepdims=True)                      # column j spreads point j's force; mass preserving
        else:
            K = np.eye(P)
        same = ((layout.rowcol[:, None, 0] == layout.rowcol[None, :, 0]) |
                (layout.rowcol[:, None, 1] == layout.rowcol[None, :, 1])) & ~np.eye(P, dtype=bool)
        same &= pres[:, None] & pres[None, :]
        C = same / np.maximum(same.sum(0, keepdims=True), 1)  # column j leaks equally to its row/column
        self.M = ((1 - params.crosstalk) * np.eye(P) + params.crosstalk * C) @ K
        self.gain = 1 + params.gain_std * rng.standard_normal((2, P, 3))   # per foot, point, axis
        self.offset = params.offset_std * rng.standard_normal((2, P, 3))
        self.sigma = np.array([params.noise_shear, params.noise_shear, params.noise_normal])

    def apply(self, f, foot: int):
        """f: true bin-averaged point forces [P, 3] -> measured sample [P, 3]."""
        y = self.M @ f
        y = y * self.gain[foot] + self.offset[foot] + self.sigma * self.rng.standard_normal(f.shape)
        y[:, :2] = np.clip(y[:, :2], -self.prm.sat_shear, self.prm.sat_shear)
        y[:, 2] = np.clip(y[:, 2], -self.prm.sat_normal, self.prm.sat_normal)
        if self.prm.quant > 0:
            y = np.round(y / self.prm.quant) * self.prm.quant
        y[~self.pres] = 0.0
        return y


# ------------------------------------------------------------------------------------------------- views
def views(points, layout_table: dict) -> dict:
    """points [..., P, 3] -> N_real [..., P], S_real [..., P, 2], N_blk [..., 4], S_blk [..., 4, 2]
    (blocks in BLOCK4 order: front_l, front_r, back_l, back_r; weighted sums over the block's points, centre-line
    points count 1/2 to each half; the middle part is not in the 4-block views)."""
    W = np.asarray(layout_table["block_weights"], float)       # [P, 5]
    names = layout_table["blocks"]
    W4 = W[:, [names.index(b) for b in BLOCK4]]                 # [P, 4]
    N = points[..., 2]
    S = points[..., :2]
    Nb = N @ W4
    Sb = np.einsum("...pc,pb->...bc", S, W4)
    return {"N_real": N, "S_real": S, "N_blk": Nb, "S_blk": Sb}


SOLE_SETTINGS = ("random", "nominal", "shifted", "pads", "g1", "nosat")


def sole_setting(name: str, rng):
    """(SoleV2Params, pads) for a named setting: random (per-episode randomization, the default for the dataset),
    nominal, shifted (sensitivity), pads (cross-check sole, nominal parameters), g1 (foot-outline mask, nominal)."""
    if name == "random":
        return SoleV2Params.sample(rng), False
    if name == "nominal":
        return SoleV2Params(), False
    if name == "shifted":
        return SoleV2Params.shifted(), False
    if name == "pads":
        return SoleV2Params(setting="pads"), True
    if name == "g1":
        return SoleV2Params(mask="rect_g1", setting="g1"), False
    if name == "nosat":  # D19 fifth sensitivity arm: nominal without per-point saturation (normal and shear)
        return SoleV2Params(sat_normal=float("inf"), sat_shear=float("inf"), setting="nosat"), False
    raise ValueError(name)
