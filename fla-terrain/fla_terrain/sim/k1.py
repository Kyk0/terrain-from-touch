"""Booster K1 constants for the MuJoCo model (k1_play_kit MJCF, 22 DoF + free base).

Default pose and PD gains are from NOTES_B.md (G113 policy interface). Joint order of
``joint_pos``/``joint_vel`` in the schema is the MJCF qpos order (JOINT_NAMES below).
"""
from __future__ import annotations

import os

K1_XML = os.environ.get(
    "FLA_K1_XML", "/path/to/fla-terrain/b_ext/k1_play_kit/assets/K1_22dof.xml"
)
BASE_BODY = "trunk"
FOOT_BODIES = {"L": "left_ankle_roll_link", "R": "right_ankle_roll_link"}

# Collision box of each foot in the base MJCF (ankle_roll_link frame): pos, half sizes.
FOOT_BOX_POS = (0.026, 0.0, -0.02)
FOOT_BOX_HALF = (0.09, 0.035, 0.018)

JOINT_NAMES = (
    "aahead_yaw_joint", "aahead_pitch_joint",
    "aaleft_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_elbow_pitch_joint", "left_elbow_yaw_joint",
    "aaright_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_elbow_pitch_joint", "right_elbow_yaw_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_pitch_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_pitch_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
)

DEFAULT_POSE = {
    "left_hip_pitch_joint": -0.2, "right_hip_pitch_joint": -0.2,
    "left_knee_pitch_joint": 0.4, "right_knee_pitch_joint": 0.4,
    "left_ankle_pitch_joint": -0.2, "right_ankle_pitch_joint": -0.2,
    "aaleft_shoulder_pitch_joint": -0.1, "aaright_shoulder_pitch_joint": -0.1,
    "left_shoulder_roll_joint": -1.371, "right_shoulder_roll_joint": 1.371,
    "left_elbow_yaw_joint": -0.35, "right_elbow_yaw_joint": 0.35,
}


def pd_gains(name: str) -> tuple[float, float]:
    """(kp [Nm/rad], kd [Nms/rad]) per joint, from NOTES_B (hip 140/3.5, knee 170/4.25,
    ankle 30/0.6, shoulder 20/1.0). Elbows and head (held, not actioned) use 20/1.0."""
    if "hip" in name:
        return 140.0, 3.5
    if "knee" in name:
        return 170.0, 4.25
    if "ankle" in name:
        return 30.0, 0.6
    return 20.0, 1.0
