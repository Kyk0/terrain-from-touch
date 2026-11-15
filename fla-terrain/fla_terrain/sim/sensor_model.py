"""Sensor model for the virtual sole: native 150 Hz from the physics rate, then 50 Hz.

Pipeline per foot and zone:
  true contact force at every physics step
  -> 150 Hz samples = mean over each 1/150 s bin (integrating sensor; bins hold 3 or 4 steps at 500 Hz)
  -> per-episode gain and offset per zone/axis, white noise, quantization, saturation
  -> 50 Hz = mean of 3 consecutive 150 Hz samples.
Numbers are placeholders (PX-FOOTRIX noise is not published); all are recorded in metadata.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class SensorModelConfig:
    native_rate_hz: float = 150.0
    noise_shear_n: float = 0.15     # white noise std per zone, Fx/Fy [N]
    noise_normal_n: float = 0.25    # white noise std per zone, Fz [N]
    gain_std: float = 0.03          # per-episode multiplicative gain error per zone/axis
    offset_std_n: float = 0.1       # per-episode offset per zone/axis [N]
    quant_n: float = 0.05           # quantization step [N]
    sat_normal_n: float = 300.0     # per-zone Fz saturation [N]
    sat_shear_n: float = 150.0      # per-zone |Fx|,|Fy| saturation [N]
    enabled: bool = True            # False = ideal sensor (true forces, still bin-averaged)

    def to_dict(self):
        return {"name": "fla_sole_v1", **asdict(self)}


class SensorModel:
    def __init__(self, cfg: SensorModelConfig, n_zones_total: int, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        shape = (n_zones_total, 3)
        self.gain = 1 + cfg.gain_std * rng.standard_normal(shape) if cfg.enabled else np.ones(shape)
        self.offset = cfg.offset_std_n * rng.standard_normal(shape) if cfg.enabled else np.zeros(shape)
        self.sigma = np.array([cfg.noise_shear_n, cfg.noise_shear_n, cfg.noise_normal_n])

    def apply(self, f: np.ndarray) -> np.ndarray:
        """f: true bin-averaged force [N_zones_total, 3] -> measured sample."""
        c = self.cfg
        if not c.enabled:
            return f.copy()
        y = f * self.gain + self.offset + self.sigma * self.rng.standard_normal(f.shape)
        y[:, :2] = np.clip(y[:, :2], -c.sat_shear_n, c.sat_shear_n)
        y[:, 2] = np.clip(y[:, 2], -c.sat_normal_n, c.sat_normal_n)
        if c.quant_n > 0:
            y = np.round(y / c.quant_n) * c.quant_n
        return y
