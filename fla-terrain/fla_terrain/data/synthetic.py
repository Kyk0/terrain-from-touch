"""Synthetic schema-v0.2 episodes for tests and pipeline smoke runs.

Not physics: a crude gait-like pattern (alternating stance, heel-to-toe roll),
shear proportional to load with a terrain-dependent ratio. Only meant to have
the right shapes, dtypes, labels and some learnable structure.
"""
from __future__ import annotations

import numpy as np

from ..schema import (
    DEFAULT_J,
    DEFAULT_ZONE_ACROSS,
    DEFAULT_ZONE_ALONG,
    MODEL_RATE_HZ,
    SCHEMA_VERSION,
    SENSOR_RATE_HZ,
    TERRAIN_NAME_TO_ID,
    zone_layout,
)

_TERRAIN_PROPS = {  # mu, stiffness [N/m], slope [rad]
    "hard_grip": (0.9, 1e6, 0.0),
    "hard_slip": (0.25, 1e6, 0.0),
    "soft": (0.7, 2e4, 0.0),
    "rough": (0.8, 1e6, 0.0),
    "slope": (0.8, 1e6, np.deg2rad(8.0)),
    "edge": (0.8, 1e6, 0.0),
}


def _foot(stance, phase, mu, rng, n_across, n_along, weight, support_frac):
    """Zone forces [T, Z, 3] for one foot given a stance mask and stance phase in [0, 1]."""
    T = stance.shape[0]
    Z = n_across * n_along
    along = (np.arange(n_along) + 0.5) / n_along  # 0 heel .. 1 toe
    cop = np.clip(phase, 0, 1)[:, None]  # rolls heel -> toe through stance
    w_along = np.exp(-((along[None, :] - cop) ** 2) / 0.08)
    w = np.repeat(w_along, n_across, axis=1)  # z = i_along * n_across + i_across
    if support_frac is not None:  # zones off the support carry nothing (toes over the edge)
        n_on = np.clip(np.round(support_frac * n_along).astype(int), 0, n_along)
        mask = (np.arange(n_along)[None, :] < n_on[:, None])
        w = w * np.repeat(mask, n_across, axis=1)
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-9)
    fz_tot = stance * weight * (0.6 + 0.4 * np.sin(np.pi * np.clip(phase, 0, 1)))
    fz = w * fz_tot[:, None]
    ratio = mu * np.clip(0.3 + 0.5 * np.cos(np.pi * np.clip(phase, 0, 1)), -1, 1)  # braking then push-off
    fx = fz * ratio[:, None] * 0.5
    fy = fz * rng.normal(0, 0.05, size=(T, 1))
    f = np.stack([fx, fy, fz], axis=-1).astype(np.float32)
    f += rng.normal(0, 0.05, size=f.shape).astype(np.float32) * (fz[..., None] > 0)
    return f


def make_synthetic_episode(
    terrain: str = "hard_grip",
    mode: str = "walk",
    duration_s: float = 10.0,
    rate_hz: float = MODEL_RATE_HZ,
    n_across: int = DEFAULT_ZONE_ACROSS,
    n_along: int = DEFAULT_ZONE_ALONG,
    n_joints: int = DEFAULT_J,
    terrain_after: str | None = None,
    split_key: str = "synthetic_000",
    seed: int = 0,
):
    """Return ``(arrays, meta)`` for one synthetic episode (not yet cast; use the writer)."""
    rng = np.random.default_rng(seed)
    T = int(round(duration_s * rate_hz))
    t = np.arange(T) / rate_hz
    weight = 193.0
    t_switch = duration_s / 2 if terrain_after else np.inf
    after = t >= t_switch

    tl = np.where(after, TERRAIN_NAME_TO_ID[terrain_after] if terrain_after else 0, TERRAIN_NAME_TO_ID[terrain])
    mu_a, k_a, s_a = _TERRAIN_PROPS[terrain]
    mu_b, k_b, s_b = _TERRAIN_PROPS[terrain_after] if terrain_after else _TERRAIN_PROPS[terrain]
    mu = np.where(after, mu_b, mu_a) * (1 + 0.05 * rng.normal())
    stiff = np.where(after, k_b, k_a)
    slope = np.where(after, s_b, s_a)
    # mark 0.3 s around the switch as transition
    tl = np.where(np.abs(t - t_switch) < 0.15, -1, tl)

    if mode == "walk":
        f_gait = 1.6
        ph = (t * f_gait + rng.uniform()) % 1.0
        st_L = ph < 0.6
        st_R = ((ph + 0.5) % 1.0) < 0.6
        phase_L = ph / 0.6
        phase_R = ((ph + 0.5) % 1.0) / 0.6
        wL = wR = weight
    elif mode in ("stand", "rig"):
        st_L = st_R = np.ones(T, bool)
        phase_L = phase_R = 0.5 + 0.1 * np.sin(2 * np.pi * 0.3 * t)
        wL = wR = weight / 2
    elif mode == "lifted":
        st_L = st_R = np.zeros(T, bool)
        phase_L = phase_R = np.zeros(T)
        wL = wR = 0.0
    else:
        raise ValueError(f"unknown mode {mode!r}")

    sf = np.ones(T)
    if terrain == "edge" or terrain_after == "edge":
        is_edge = tl == TERRAIN_NAME_TO_ID["edge"]
        sf = np.where(is_edge, rng.uniform(0.4, 0.8), 1.0)
    sf_L = np.where(st_L, sf, 0.0)
    sf_R = np.where(st_R, sf, 0.0)
    fL = _foot(st_L.astype(float), phase_L, mu, rng, n_across, n_along, wL, sf_L if (sf < 1).any() else None)
    fR = _foot(st_R.astype(float), phase_R, mu, rng, n_across, n_along, wR, sf_R if (sf < 1).any() else None)

    def sup(st, frac):
        return np.where(~st, 0, np.where(frac >= 0.95, 2, 1)).astype(np.int8)

    jp = 0.3 * np.sin(2 * np.pi * 1.6 * t[:, None] + rng.uniform(0, 2 * np.pi, n_joints)) * (mode == "walk")
    jp = jp + rng.normal(0, 0.01, (T, n_joints))
    jv = np.gradient(jp, 1 / rate_hz, axis=0)
    grav = np.tile([0.0, 0.0, -1.0], (T, 1)) + rng.normal(0, 0.01, (T, 3))
    grav[:, 0] += np.sin(slope)
    arrays = {
        "t": t,
        "foot_zone_L": fL,
        "foot_zone_R": fR,
        "foot_valid_L": np.ones(T, bool),
        "foot_valid_R": np.ones(T, bool),
        "joint_pos": jp,
        "joint_vel": jv,
        "imu_gyro": rng.normal(0, 0.05, (T, 3)),
        "imu_gravity": grav,
        "cmd_vel": np.tile([0.4 if mode == "walk" else 0.0, 0.0, 0.0], (T, 1)),
        "label_terrain": tl.astype(np.int8),
        "label_support_L": sup(st_L, sf_L),
        "label_support_R": sup(st_R, sf_R),
        "prop_mu": mu,
        "prop_stiffness": stiff,
        "prop_slope": slope,
        "prop_support_frac_L": sf_L,
        "prop_support_frac_R": sf_R,
        "contact_gt_L": st_L,
        "contact_gt_R": st_R,
    }
    meta = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": 0,
        "source": "sim",
        "robot": "booster_k1_synthetic",
        "motion_source": "synthetic",
        "sensor_model": {"name": "synthetic", "noise_n": 0.05},
        "zone_layout": zone_layout(n_across, n_along),
        "rate_hz": float(rate_hz),
        "sensor_native_rate_hz": SENSOR_RATE_HZ,
        "terrain_params": {"terrain": terrain, "terrain_after": terrain_after, "t_switch": None if not terrain_after else t_switch},
        "split_key": split_key,
        "mode": mode,
        "notes": "synthetic episode, not physics",
    }
    return arrays, meta
