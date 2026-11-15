"""MuJoCo simulation: K1 with virtual sole, terrains, sensor model, episode recorder.

Public interface for the independent evaluation: build_model(terrain_cfg, sole_cfg=None) -> (mjModel, info),
SoleSensor(model).read(data) -> {'L': [Z,3], 'R': [Z,3]}, EpisodeRecorder.
"""
from .build import TERRAIN_GROUP, add_terrain, build_model, build_model_v2  # noqa: F401
from .motion import PDController, StandMotionSource, set_pose  # noqa: F401
from .recorder import EpisodeRecorder, ProprioNoise  # noqa: F401
from .sensor_model import SensorModel, SensorModelConfig  # noqa: F401
from .sole import SoleConfig, SoleSensor  # noqa: F401
from .support import SupportProbe  # noqa: F401
