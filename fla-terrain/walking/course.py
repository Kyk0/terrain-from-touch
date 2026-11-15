"""Mixed-terrain course (FLA-0 v2, brief section 5): a straight course of terrain chunks in random order.

    cfg = make_course(seed, split="test")        # terrain_cfg for fla_terrain.sim.build_model
    model, info = build_model(cfg)
    chunk = chunk_at(cfg, x)                     # chunk dict under base x position

Design
- The robot walks along +x (spawn yaw 0). A 1.5 m hard-grip start pad (region "start"), then chunks, then a 6 m
  hard-grip run-out pad (region "end"). Each chunk spans the full course width (|y| <= HALF_WIDTH = 4 m), so lateral
  drift does not take the robot off its chunk; beyond the width and below the course there is a catch plane at
  z = -0.5 m (stepping off the side counts as leaving the course).
- Chunk classes: hard_grip, hard_slip, soft, rough, slope, edge. A run has every class once in random order, then
  extra random chunks (never the same class twice in a row) until the planned crossing time reaches ~60 s.
  Chunk length = U(6, 10) s x v_nominal (default 0.35 m/s, the commanded speed of the velocity policies), i.e.
  2.1-3.5 m. Materials and bands are those of fla_terrain.sim.terrains (sample_band with the given split).
- Every chunk is its own surface at top z = 0 (box slab, 0.1 m thick), so friction / softness change exactly at
  the boundary x. Rough = grip slab + rigid bumps (heights U(0.5h, h), h ~ U(0.5, 2) cm, as in v1).
- Slope chunk = a ridge: up over the first half, down over the second half, at theta ~ U(5, 10) deg, so both
  neighbours join at z = 0 with no step. Its region records the slope angle; both halves are class slope.
- Edge chunk, one of two variants (as v1 edge terrains, laid across the course):
  planks: 10-25 cm wide planks running along y with 3-8 cm gaps over a 15 cm pit (pit floor = same region);
  steps:  low platforms 3-6 cm high, 0.3-0.8 m long in x, separated by 0.3-0.8 m of flat ground.
  Both start and end with flat ground at z = 0 at the chunk boundaries.
- Metadata: cfg["chunks"] = [{index, x0, x1, class, label, region, mu, stiffness, slope_deg, variant, length_m,
  planned_s}], cfg["boundaries"] = x positions of every chunk start. Courses are test only by default.
"""
from __future__ import annotations

import math

import numpy as np

from fla_terrain.sim.terrains import (RIGID_STIFFNESS, SOFT_SOLIMP, sample_band, stiffness_from_tc)
from fla_terrain.schema import TERRAIN_NAME_TO_ID

CLASSES = ("hard_grip", "hard_slip", "soft", "rough", "slope", "edge")
HALF_WIDTH = 4.0
START_PAD = 1.5
END_PAD = 6.0
SLAB = 0.1          # slab thickness [m]
PIT = 0.15          # plank pit depth [m]


def _slab(x0, x1, region, props, top=0.0, half_w=HALF_WIDTH):
    return {"type": "box", "pos": [(x0 + x1) / 2, 0.0, top - SLAB / 2], "size": [(x1 - x0) / 2, half_w, SLAB / 2],
            "region": region, **props}


def _region(cls, mu, stiffness=RIGID_STIFFNESS, slope=0.0):
    return {"label": TERRAIN_NAME_TO_ID[cls], "mu": float(mu), "stiffness": float(stiffness),
            "slope": float(slope), "class": cls}


def _order(rng, target_s, dur):
    order = list(rng.permutation(CLASSES))
    total = sum(dur(c) for c in order)
    while total < target_s:
        c = str(rng.choice([k for k in CLASSES if k != order[-1]]))
        order.append(c)
        total += dur(c)
    return [str(c) for c in order]


def make_course(seed: int, split: str = "test", v_nominal: float = 0.35, target_s: float = 60.0,
                edge_variant: str | None = None) -> dict:
    rng = np.random.default_rng(seed)
    durs = {}

    def dur(c):  # planned crossing time of the next chunk of class c
        d = float(rng.uniform(6.0, 10.0))
        durs.setdefault(c, []).append(d)
        return d

    order = _order(rng, target_s - 0.0, dur)
    # flatten planned durations in the order they were drawn
    drawn = {c: list(v) for c, v in durs.items()}
    regions, geoms, chunks = [], [], []
    # start pad (region 0)
    mu0, _ = sample_band("mu_grip", split, rng)
    regions.append({**_region("hard_grip", mu0), "name": "start"})
    geoms.append(_slab(-2.0, START_PAD, 0, {"mu": mu0}))
    x = START_PAD
    for i, cls in enumerate(order):
        planned = drawn[cls].pop(0)
        L = planned * v_nominal
        x0, x1 = x, x + L
        reg = len(regions)
        info = {"index": i, "x0": x0, "x1": x1, "class": cls, "label": TERRAIN_NAME_TO_ID[cls], "region": reg,
                "length_m": L, "planned_s": planned, "slope_deg": 0.0, "variant": None}
        if cls in ("hard_grip", "rough", "edge", "slope"):
            mu, band = sample_band("mu_grip", split, rng)
        elif cls == "hard_slip":
            mu, band = sample_band("mu_slip", split, rng)
        else:  # soft: grip friction + soft contact
            mu, band = sample_band("mu_grip", split, rng)
        info.update(mu=float(mu), mu_band=band, stiffness=RIGID_STIFFNESS)
        props = {"mu": float(mu)}
        if cls == "soft":
            tc, sb = sample_band("soft_tc", split, rng)
            k = stiffness_from_tc(tc)
            props.update(solref=[tc, 1.0], solimp=list(SOFT_SOLIMP))
            info.update(stiffness=float(k), soft_tc=float(tc), stiffness_band=sb)
            regions.append(_region("soft", mu, k))
            geoms.append(_slab(x0, x1, reg, props))
        elif cls in ("hard_grip", "hard_slip"):
            regions.append(_region(cls, mu))
            geoms.append(_slab(x0, x1, reg, props))
        elif cls == "rough":
            h = float(rng.uniform(0.005, 0.02))
            info["bump_height"] = h
            regions.append(_region("rough", mu))
            geoms.append(_slab(x0, x1, reg, props))
            n = int(12.0 * L * 2 * HALF_WIDTH)
            for _ in range(n):
                bx, by = rng.uniform(x0 + 0.05, x1 - 0.05), rng.uniform(-HALF_WIDTH, HALF_WIDTH)
                hh = rng.uniform(0.5, 1.0) * h
                sx, sy = rng.uniform(0.02, 0.05, 2)
                yaw = rng.uniform(0, np.pi)
                geoms.append({"type": "box", "pos": [bx, by, hh / 2], "size": [sx, sy, hh / 2],
                              "quat": [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], "mu": float(mu), "region": reg})
        elif cls == "slope":
            deg = float(rng.uniform(5.0, 10.0))
            th = math.radians(deg)
            info["slope_deg"] = deg
            regions.append(_region("slope", mu, slope=th))
            half = L / 2
            for sgn, xa in ((1, x0), (-1, x0 + half)):
                # slab whose top surface goes from (xa, z_a) to (xa + half, z_b): up (sgn=+1) then down
                za = 0.0 if sgn > 0 else half * math.tan(th)
                zb = za + sgn * half * math.tan(th)
                cx, cz = xa + half / 2, (za + zb) / 2
                lh = half / 2 / math.cos(th) + 0.02           # half length along the incline (+2 cm overlap)
                ang = -sgn * th                               # rotation about +y
                # centre of the slab = top-surface midpoint shifted along the slab normal by -SLAB/2
                nx, nz = math.sin(sgn * th) * -1.0, math.cos(th)
                pos = [cx - nx * SLAB / 2, 0.0, cz - nz * SLAB / 2]
                geoms.append({"type": "box", "pos": pos, "size": [lh, HALF_WIDTH, SLAB / 2],
                              "quat": [math.cos(ang / 2), 0.0, math.sin(ang / 2), 0.0], "region": reg, **props})
            # flat underlay so the ridge feet meet z = 0 cleanly
            geoms.append(_slab(x0, x1, reg, props, top=-0.001))
        elif cls == "edge":
            variant = edge_variant or str(rng.choice(["planks", "steps"]))
            info["variant"] = variant
            regions.append(_region("edge", mu))
            if variant == "planks":
                lead = 0.3   # flat ground at both ends of the chunk
                geoms.append(_slab(x0, x0 + lead, reg, props))
                geoms.append(_slab(x1 - lead, x1, reg, props))
                geoms.append(_slab(x0 + lead, x1 - lead, reg, props, top=-PIT))       # pit floor
                a, n = x0 + lead, 0
                while True:
                    gap, w = float(rng.uniform(0.03, 0.08)), float(rng.uniform(0.10, 0.25))
                    if a + gap + w > x1 - lead - 0.03:
                        break
                    geoms.append(_slab(a + gap, a + gap + w, reg, props))
                    a += gap + w
                    n += 1
                info["planks"] = n
            else:
                geoms.append(_slab(x0, x1, reg, props))
                a, n = x0 + float(rng.uniform(0.3, 0.8)), 0
                while True:
                    ln, hp = float(rng.uniform(0.3, 0.8)), float(rng.uniform(0.03, 0.06))
                    if a + ln > x1 - 0.3:
                        break
                    geoms.append(_slab(a, a + ln, reg, props, top=hp))
                    a += ln + float(rng.uniform(0.3, 0.8))
                    n += 1
                info["platforms"] = n
        chunks.append(info)
        x = x1
    # run-out pad and catch plane
    mu_e, _ = sample_band("mu_grip", split, rng)
    regions.append({**_region("hard_grip", mu_e), "name": "end"})
    geoms.append(_slab(x, x + END_PAD, len(regions) - 1, {"mu": mu_e}))
    geoms.append({"type": "plane", "pos": [0, 0, -0.5], "region": 0, "mu": 1.0})
    for r in regions:
        r.setdefault("slope", 0.0)
    return {"class": "course", "regions": regions, "geoms": geoms, "spawn": [0.3, 0.0, 0.0], "yaw": 0.0,
            "wz_max": 0.0, "v_nominal": v_nominal, "chunks": chunks,
            "boundaries": [c["x0"] for c in chunks] + [x], "course_end_x": x,
            "params": {"class": "course", "range_split": split, "seed": int(seed), "n_chunks": len(chunks),
                       "order": [c["class"] for c in chunks], "planned_s": float(sum(c["planned_s"] for c in chunks))}}


def chunk_at(cfg: dict, x: float):
    for c in cfg["chunks"]:
        if c["x0"] <= x < c["x1"]:
            return c
    return None
