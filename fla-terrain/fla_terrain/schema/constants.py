"""Schema v0.2 constants: label enums, array specs, metadata keys.

Units: seconds, newtons, radians. Sole frame: x forward (toe), y left, z up.
Fz >= 0 is load on the sole; Fx, Fy are shear. Left/right stored separately.
"""
from __future__ import annotations

import numpy as np

SCHEMA_VERSION = "0.2"      # v1 (zone arrays)
SCHEMA_VERSION_V3 = "0.3"   # v2 sole (per-point arrays)
SCHEMA_VERSIONS = (SCHEMA_VERSION, SCHEMA_VERSION_V3)

# ---- labels ---------------------------------------------------------------
TERRAIN_LABELS = {
    -1: "unknown",  # unknown / transition
    0: "hard_grip",
    1: "hard_slip",
    2: "soft",
    3: "rough",
    4: "slope",
    5: "edge",
}
TERRAIN_NAME_TO_ID = {v: k for k, v in TERRAIN_LABELS.items()}
N_TERRAIN_CLASSES = 6  # classes 0..5; -1 is ignored in training

SUPPORT_LABELS = {0: "airborne", 1: "partial", 2: "full"}
N_SUPPORT_CLASSES = 3

MODES = ("walk", "stand", "rig", "lifted")
SOURCES = ("sim", "real")

# ---- default sizes ----------------------------------------------------------
DEFAULT_ZONE_ACROSS = 2  # zones across the sole (y)
DEFAULT_ZONE_ALONG = 4   # zones along the sole (x)
DEFAULT_Z = DEFAULT_ZONE_ACROSS * DEFAULT_ZONE_ALONG
DEFAULT_J = 22           # Booster K1 22-DoF model
MODEL_RATE_HZ = 50.0
SENSOR_RATE_HZ = 150.0
WINDOW_S = 1.0
STRIDE_S = 0.2

# ---- arrays -----------------------------------------------------------------
# key -> (shape with symbols "T", "Z", "J" or ints, canonical dtype)
F32 = np.dtype(np.float32)
F64 = np.dtype(np.float64)
I8 = np.dtype(np.int8)
BOOL = np.dtype(np.bool_)

ARRAY_SPECS: dict[str, tuple[tuple, np.dtype]] = {
    "t": (("T",), F64),
    "foot_zone_L": (("T", "Z", 3), F32),
    "foot_zone_R": (("T", "Z", 3), F32),
    "foot_valid_L": (("T",), BOOL),
    "foot_valid_R": (("T",), BOOL),
    "joint_pos": (("T", "J"), F32),
    "joint_vel": (("T", "J"), F32),
    "imu_gyro": (("T", 3), F32),
    "imu_gravity": (("T", 3), F32),
    "cmd_vel": (("T", 3), F32),
    "label_terrain": (("T",), I8),
    "label_support_L": (("T",), I8),
    "label_support_R": (("T",), I8),
    "prop_mu": (("T",), F32),
    "prop_stiffness": (("T",), F32),
    "prop_slope": (("T",), F32),
    "prop_support_frac_L": (("T",), F32),
    "prop_support_frac_R": (("T",), F32),
    "contact_gt_L": (("T",), BOOL),
    "contact_gt_R": (("T",), BOOL),
}
ARRAY_KEYS = tuple(ARRAY_SPECS)

# v0.3: the zone arrays are replaced by per-point arrays; the point-coordinate table, part / half index and block
# weights per point live in metadata zone_layout (P = number of points per foot).
ARRAY_SPECS_V03: dict[str, tuple[tuple, np.dtype]] = {
    ("foot_points_" + k[-1] if k.startswith("foot_zone_") else k): ((("T", "P", 3), F32) if k.startswith("foot_zone_") else v)
    for k, v in ARRAY_SPECS.items()
}


ARRAY_SPECS_V03["label_ground_L"] = (("T",), I8)
ARRAY_SPECS_V03["label_ground_R"] = (("T",), I8)
# optional v0.3 arrays (walking sources): PD position target sent by the motion source, actuator torque; joint order
# = joint_pos order. Only for the rule-based detector (signals a controller already has); no learned model reads them.
OPTIONAL_V03: dict[str, tuple[tuple, np.dtype]] = {"joint_target": (("T", "J"), F32), "joint_torque": (("T", "J"), F32)}
GROUND_LABELS = {0: "airborne", 1: "void", 2: "flat", 3: "bump", 4: "partial"}
# per foot per frame: contact -> partial (support probe partial) > bump (touched a bump geom in the frame) > flat;
# no contact -> void (less than half of the sole over support (drop-off rule of the support probe), or terrain under
# the sole centre > 20 cm below the sole, or none) else airborne.


def array_specs(version: str) -> dict:
    return ARRAY_SPECS_V03 if version == SCHEMA_VERSION_V3 else ARRAY_SPECS

# ---- metadata ---------------------------------------------------------------
META_KEYS = (
    "schema_version",
    "episode_id",
    "source",
    "robot",
    "motion_source",
    "sensor_model",
    "zone_layout",
    "rate_hz",
    "sensor_native_rate_hz",
    "terrain_params",
    "split_key",
    "mode",
    "notes",
)


def zone_layout(n_across: int = DEFAULT_ZONE_ACROSS, n_along: int = DEFAULT_ZONE_ALONG) -> dict:
    """Zone layout metadata. Zone index z = i_along * n_across + i_across,
    with i_along = 0 at the heel and i_across = 0 at the -y side of the sole."""
    return {
        "name": f"grid_{n_across}x{n_along}",
        "n_across": int(n_across),
        "n_along": int(n_along),
        "Z": int(n_across * n_along),
        "index": "z = i_along * n_across + i_across; i_along=0 heel, i_across=0 at -y",
    }


# ---- input sets for the ablation -------------------------------------------
INPUT_SETS = ("P", "P+N", "P+N+S")
