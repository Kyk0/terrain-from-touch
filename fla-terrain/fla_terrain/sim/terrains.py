"""Terrain suite (brief section 5) and parameter randomization with split-separated bands.

``sample_terrain(cls, rng, range_split, variant=None) -> terrain_cfg`` for ``build_model``.

Classes: hard_grip, hard_slip, soft, rough, slope, edge, plus strips ``strip_<A>_<B>`` where the
surface changes from A to B at x = x_change (A, B in grip/slip/soft/rough).

Train / val / test bands (``BANDS``) for friction and soft-ground stiffness are disjoint with gaps;
test has one band inside the train span (``test`` = test_in) and one outside it (test_out), both
labelled range_split='test' (the sub-band is kept in terrain_params.band). Non-slip classes use grip
friction, so low friction is what defines hard-slip. Bump height, slope angle/direction and edge
geometry are randomized identically in all splits.

Soft ground: terrain solref timeconst tc with solimp (0.1, 0.95, 0.1, 0.5, 2). prop_stiffness is the
whole-robot vertical effective stiffness K = m g / static sink of the K1 standing on it (N/m), from a
calibration of THIS model (2x4 zones per foot; MuJoCo soft-contact stiffness depends on the number of
contacts, so these numbers differ from the independent evaluation's single-box feet). Rigid ground: nominal K below.
"""
from __future__ import annotations

import numpy as np

from ..schema import TERRAIN_NAME_TO_ID

SOFT_SOLIMP = (0.1, 0.95, 0.1, 0.5, 2.0)
# calibration (scripts in NOTES_A T4): tc -> standing static sink of the K1 with 2x4 zones
_CAL_TC = np.array([0.005, 0.008, 0.012, 0.016, 0.020, 0.025, 0.030, 0.040])
_CAL_K = np.array([729.0, 258.3, 107.8, 56.7, 30.7, 16.9, 11.4, 8.3]) * 1e3  # N/m
RIGID_STIFFNESS = 7.7e8  # N/m, default contact (solref 0.001 clamped to 2 dt): sink ~0.25 um (nominal)


def stiffness_from_tc(tc: float) -> float:
    return float(np.exp(np.interp(np.log(tc), np.log(_CAL_TC), np.log(_CAL_K))))


# disjoint bands; each split may have several sub-bands (name, lo, hi)
BANDS = {
    "mu_grip": {
        "train": [("train_a", 0.70, 0.80), ("train_b", 0.86, 0.95), ("train_c", 1.01, 1.10)],
        "val": [("val", 0.82, 0.84)],
        "test": [("test_in", 0.97, 0.99), ("test_out", 1.20, 1.30)],
    },
    "mu_slip": {
        "train": [("train_a", 0.14, 0.19), ("train_b", 0.23, 0.27), ("train_c", 0.31, 0.36)],
        "val": [("val", 0.205, 0.215)],
        "test": [("test_in", 0.285, 0.295), ("test_out", 0.08, 0.12)],
    },
    # hard_mid (v2, D14): test only, no terrain label; friction between the slip and the grip bands
    "mu_mid": {"test": [("test_mid", 0.36, 0.70)]},
    "soft_tc": {  # larger tc = softer
        "train": [("train_a", 0.012, 0.016), ("train_b", 0.020, 0.024), ("train_c", 0.028, 0.031)],
        "val": [("val", 0.0175, 0.0185)],
        "test": [("test_in", 0.0255, 0.0265), ("test_out", 0.034, 0.040)],
    },
}
SPLITS = ("train", "val", "test")


def sample_band(name: str, split: str, rng) -> tuple[float, str]:
    bands = BANDS[name][split]
    if split == "test":  # test_in and test_out equally often (by width test_out would dominate)
        i = rng.integers(len(bands))
    else:  # uniform over the union of the split's bands
        widths = np.array([hi - lo for _, lo, hi in bands])
        i = rng.choice(len(bands), p=widths / widths.sum())
    b, lo, hi = bands[i]
    return float(rng.uniform(lo, hi)), b


def band_table() -> list[tuple]:
    rows = []
    for p, d in BANDS.items():
        for split, bands in d.items():
            for b, lo, hi in bands:
                rows.append((p, split, b, lo, hi))
    return rows


def check_band_overlap() -> list[str]:
    """Errors if any two bands of different splits of one parameter overlap or touch."""
    errs = []
    for p, d in BANDS.items():
        if p == "mu_mid":  # test-only set; checked to lie strictly between the slip and grip bands
            lo, hi = d["test"][0][1:]
            sl = max(b[2] for bb in BANDS["mu_slip"].values() for b in bb)
            gl = min(b[1] for bb in BANDS["mu_grip"].values() for b in bb)
            if not (sl <= lo and hi <= gl):  # the brief's 0.36-0.70 touches slip (0.36) and grip (0.70) ends
                errs.append("mu_mid overlaps the slip or grip bands")
            continue
        ivs = [(lo, hi, s, b) for s, bands in d.items() for b, lo, hi in bands]
        for i in range(len(ivs)):
            for j in range(i + 1, len(ivs)):
                a, c = ivs[i], ivs[j]
                if a[2] != c[2] and a[0] <= c[1] and c[0] <= a[1]:
                    errs.append(f"{p}: {a[2]}/{a[3]} [{a[0]},{a[1]}] overlaps {c[2]}/{c[3]} [{c[0]},{c[1]}]")
        train = [(lo, hi) for _, lo, hi in d["train"]]
        span = (min(t[0] for t in train), max(t[1] for t in train))
        tin = [b for b in d["test"] if span[0] < b[1] < span[1]]
        tout = [b for b in d["test"] if b[2] < span[0] or b[1] > span[1]]
        if not tin or not tout:
            errs.append(f"{p}: test needs one band inside and one outside the train span {span}")
    return errs


# ---------------------------------------------------------------------------------------------
def _region(cls, mu, stiffness=RIGID_STIFFNESS, slope=0.0):
    return {"label": TERRAIN_NAME_TO_ID[cls], "mu": float(mu), "stiffness": float(stiffness), "slope": float(slope),
            "class": cls}


def _surface(kind: str, rng, split: str, params: dict):
    """Material of a flat surface of kind grip/slip/soft/rough: (region dict, geom props dict)."""
    if kind == "slip":
        mu, b = sample_band("mu_slip", split, rng)
        params["mu_band"] = b
        return _region("hard_slip", mu), {"mu": mu}
    mu, b = sample_band("mu_grip", split, rng)
    params["mu_band"] = b
    if kind == "soft":
        tc, bs = sample_band("soft_tc", split, rng)
        params.update(soft_tc=tc, stiffness_band=bs)
        return _region("soft", mu, stiffness_from_tc(tc)), {"mu": mu, "solref": [tc, 1.0], "solimp": list(SOFT_SOLIMP)}
    return _region("hard_grip" if kind == "grip" else "rough", mu), {"mu": mu}


FIELD = 7.0  # half size of the square terrain field around the spawn [m] (20 s at <= 0.5 m/s stays inside)


def _box(center, half, region, props):
    return {"type": "box", "pos": list(map(float, center)), "size": list(map(float, half)), "region": region,
            **props}


def _ring(a_in, a_out, top, th, region, props):
    """Square ring a_in < max(|x|,|y|) < a_out around the origin, top surface at z=top (4 boxes)."""
    w = a_out - a_in
    c = (a_in + a_out) / 2
    z = top - th / 2
    return [_box((sx * c, 0, z), (w / 2, a_out, th / 2), region, props) for sx in (-1, 1)] + \
           [_box((0, sy * c, z), (a_in, w / 2, th / 2), region, props) for sy in (-1, 1)]


def _bumps(rng, region: int, mu: float, a_in: float, a_out: float, h: float, density=12.0):
    """Rigid bumps: scattered boxes 4-10 cm wide, heights U(0.5 h, h) on z=0, uniformly in the square ring
    a_in < max(|x|,|y|) < a_out."""
    n = int(density * ((2 * a_out) ** 2 - (2 * a_in) ** 2))
    geoms = []
    while len(geoms) < n:
        x, y = rng.uniform(-a_out, a_out, 2)
        if max(abs(x), abs(y)) < a_in:
            continue
        hh = rng.uniform(0.5, 1.0) * h
        sx, sy = rng.uniform(0.02, 0.05, 2)
        yaw = rng.uniform(0, np.pi)
        geoms.append({"type": "box", "pos": [x, y, hh / 2], "size": [sx, sy, hh / 2],
                      "quat": [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)], "mu": mu, "region": region})
    return geoms


def _flat_cfg(kind, rng, split, params):
    reg, props = _surface(kind, rng, split, params)
    geoms = [{"type": "plane", "region": 0, **props}]
    if kind == "rough":
        h = float(rng.uniform(0.005, 0.02))
        params["bump_height"] = h
        geoms += _bumps(rng, 0, props["mu"], 0.3, FIELD, h)
    return [reg], geoms


def _edge_cfg(variant, rng, split, params):
    """Edge terrains as concentric square rings around the spawn, so any walking direction crosses them.
    All frames are labelled 'edge'; partial support comes from the support probe."""
    mu, b = sample_band("mu_grip", split, rng)
    params["mu_band"] = b
    reg = [_region("edge", mu)]
    pr = {"mu": mu}
    if variant == "planks":  # rings 10-25 cm wide with 3-8 cm gaps over a 15 cm drop
        geoms = [{"type": "plane", "pos": [0, 0, -0.15], "region": 0, "mu": mu},
                 _box((0, 0, -0.05), (0.4, 0.4, 0.05), 0, pr)]
        a, n = 0.4, 0
        while a < FIELD:
            gap, w = float(rng.uniform(0.03, 0.08)), float(rng.uniform(0.10, 0.25))
            geoms += _ring(a + gap, a + gap + w, 0.0, 0.1, 0, pr)
            a += gap + w
            n += 1
        params["rings"] = n
    elif variant == "steps":  # rings of low platforms (3-6 cm) separated by flat rings
        geoms = [{"type": "plane", "region": 0, "mu": mu}]
        a, n = float(rng.uniform(0.4, 0.8)), 0
        while a < FIELD:
            ln, h = float(rng.uniform(0.3, 0.8)), float(rng.uniform(0.03, 0.06))
            geoms += _ring(a, a + ln, h, h, 0, pr)
            a += ln + float(rng.uniform(0.3, 0.8))
            n += 1
        params["rings"] = n
    else:
        raise ValueError(variant)
    params["variant"] = variant
    return reg, geoms


def _strip_cfg(a, b, rng, split, params):
    """Surface A in a square of half size r_c around the spawn, surface B outside (any heading crosses it)."""
    pa, pb = {}, {}
    ra, ga = _surface(a, rng, split, pa)
    rb, gb = _surface(b, rng, split, pb)
    rc = float(rng.uniform(1.5, 3.0))
    params.update(change_halfsize=rc, before=pa, after=pb, surfaces=[a, b])
    geoms = [_box((0, 0, -0.05), (rc, rc, 0.05), 0, ga)] + _ring(rc, FIELD + 3.0, 0.0, 0.1, 1, gb) + \
            [{"type": "plane", "pos": [0, 0, -0.5], "region": 0, "mu": 1.0}]  # catch plane far below
    for side, kind, g, reg in ((0, a, ga, 0), (1, b, gb, 1)):
        if kind == "rough":
            h = float(rng.uniform(0.005, 0.02))
            params["bump_height"] = h
            geoms += _bumps(rng, reg, g["mu"], 0.3 if side == 0 else rc, rc if side == 0 else FIELD, h)
    return [ra, rb], geoms


STRIPS = ("strip_grip_slip", "strip_slip_grip", "strip_grip_soft", "strip_grip_rough")
EDGE_VARIANTS = ("planks", "steps")
CLASSES = ("hard_grip", "hard_slip", "soft", "rough", "slope", "edge") + STRIPS
TEST_ONLY = ("hard_mid",)


def sample_terrain(cls: str, rng, range_split: str = "train", variant: str | None = None,
                   stand: bool = False) -> dict:
    """Return terrain_cfg (build_model input) with regions, geoms, spawn pose, command limits, params."""
    if range_split not in SPLITS:
        raise ValueError(range_split)
    params = {"class": cls, "range_split": range_split}
    spawn_z = 0.0
    yaw = float(rng.uniform(-np.pi, np.pi))
    wz_max = 0.1
    if cls == "hard_mid":
        if range_split != "test":
            raise ValueError("hard_mid is a test-only set (D14)")
        mu, b = sample_band("mu_mid", "test", rng)
        params["mu_band"] = b
        regions = [{"label": -1, "mu": float(mu), "stiffness": RIGID_STIFFNESS, "slope": 0.0, "class": "hard_mid"}]
        geoms = [{"type": "plane", "region": 0, "mu": mu}]
    elif cls in ("hard_grip", "hard_slip", "soft", "rough"):
        kind = {"hard_grip": "grip", "hard_slip": "slip", "soft": "soft", "rough": "rough"}[cls]
        regions, geoms = _flat_cfg(kind, rng, range_split, params)
        if cls == "rough":
            wz_max = 0.0
    elif cls == "slope":
        mu, b = sample_band("mu_grip", range_split, rng)
        deg = float(rng.uniform(5.0, 10.0))
        rel = float(rng.uniform(-np.pi, np.pi))  # uphill direction relative to the robot's spawn heading
        if stand:  # PD stand cannot level its feet on a lateral slope: uphill/downhill +-0.2 rad only
            rel = float(rng.choice([0.0, np.pi]) + rng.uniform(-0.2, 0.2))
        from .build import slope_normal
        params.update(mu_band=b, slope_deg=deg, uphill_rel_heading=rel)
        regions = [_region("slope", mu, slope=np.deg2rad(deg))]
        geoms = [{"type": "plane", "normal": list(slope_normal(deg, yaw + rel)), "mu": mu, "region": 0}]
    elif cls == "edge":
        variant = variant or str(rng.choice(EDGE_VARIANTS))
        regions, geoms = _edge_cfg(variant, rng, range_split, params)
        wz_max = 0.0
    elif cls in STRIPS:
        _, a, b = cls.split("_")
        regions, geoms = _strip_cfg(a, b, rng, range_split, params)
        wz_max = 0.0
    else:
        raise ValueError(f"unknown terrain class {cls!r}")
    if stand and cls == "edge":  # stand somewhere on the ring field so the feet straddle edges
        spawn_xy = rng.uniform(-2.0, 2.0, 2)
        params["stand_spawn"] = spawn_xy.tolist()
        return {"class": cls, "regions": regions, "geoms": geoms, "spawn": [*spawn_xy, spawn_z], "yaw": yaw,
                "wz_max": wz_max, "params": params}
    for r in regions:  # all props recorded
        r.setdefault("slope", 0.0)
    return {"class": cls, "regions": regions, "geoms": geoms, "spawn": [0.0, 0.0, spawn_z], "yaw": yaw,
            "wz_max": wz_max, "params": params}
