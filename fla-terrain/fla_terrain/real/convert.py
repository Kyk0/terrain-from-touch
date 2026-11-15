"""Real sole -> schema zones (stub for PaXini PX-FOOTRIX-style taxel grids).

    cfg = TaxelGridConfig(H=18, W=7, row0_at="heel", col0_at="-y")      # one per foot (mirror for R)
    zones150 = taxels_to_zones(taxels, cfg)                                # [T, H, W, 3] -> [T, Z, 3]
    t50, zones50, valid50 = resample_to_rate(t150, zones150, rate_hz=50)   # same averaging as the sim

Conventions (same as fla_terrain.sim.sole / sensor_model, schema v0.2):
  * sole frame: origin at the centre of the sole bottom, x forward (toe), y left, z up; force ON the sole FROM the
    ground, so Fz >= 0 is load and (Fx, Fy) is shear;
  * zones: rectangle sole_length x sole_width tiled n_along x n_across, index z = i_along * n_across + i_across,
    i_along = 0 at the heel, i_across = 0 at -y (defaults = SoleConfig: 0.18 x 0.07 m, 4 x 2);
  * rates: the sim integrates to 150 Hz samples, then a 50 Hz frame = mean of the 3 samples in it, stamped at the
    end of its interval, t_k = (k + 1) / 50. resample_to_rate bins real timestamps the same way.
Left and right are stored separately and NOT mirrored: give each foot its own TaxelGridConfig (axis flips).
Unknowns to fill in from the real sensor: grid size, which corner is taxel (0, 0), taxel force axis signs, grid
offset from the sole centre, sole outline mask. Approximation: each taxel's force is assigned whole to the zone
containing its centre (no area splitting across zone borders).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TaxelGridConfig:
    H: int = 18                      # rows (along the foot)
    W: int = 7                       # columns (across the foot)
    pitch_m: float = 0.01
    row0_at: str = "heel"            # "heel" | "toe": where row 0 is
    col0_at: str = "-y"              # "-y" | "+y": where column 0 is (mirror this between L and R)
    offset_m: tuple = (0.0, 0.0)     # grid centre in the sole frame (x, y)
    force_sign: tuple = (1.0, 1.0, 1.0)   # extra sign per taxel axis (u along rows, v along cols, n normal)
    mask: np.ndarray | None = None   # [H, W] bool sole outline; False taxels are ignored
    outside: str = "nearest"         # taxels outside the zone rectangle: "nearest" zone or "drop"
    sole_length_m: float = 0.18
    sole_width_m: float = 0.07
    n_along: int = 4
    n_across: int = 2
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def Z(self) -> int:
        return self.n_along * self.n_across

    @property
    def signs(self) -> tuple:
        sx = 1.0 if self.row0_at == "heel" else -1.0
        sy = 1.0 if self.col0_at == "-y" else -1.0
        return sx, sy

    def taxel_xy(self) -> np.ndarray:
        """[H, W, 2] taxel centres in the sole frame."""
        sx, sy = self.signs
        r = (np.arange(self.H) - (self.H - 1) / 2) * self.pitch_m * sx + self.offset_m[0]
        c = (np.arange(self.W) - (self.W - 1) / 2) * self.pitch_m * sy + self.offset_m[1]
        return np.stack(np.meshgrid(r, c, indexing="ij"), -1)

    def zone_of_taxel(self) -> np.ndarray:
        """[H, W] zone index per taxel, -1 = ignored."""
        if "zone" not in self._cache:
            xy = self.taxel_xy()
            L, Wd = self.sole_length_m, self.sole_width_m
            fa = (xy[..., 0] + L / 2) / (L / self.n_along)
            fc = (xy[..., 1] + Wd / 2) / (Wd / self.n_across)
            ia, ic = np.floor(fa).astype(int), np.floor(fc).astype(int)
            inside = (ia >= 0) & (ia < self.n_along) & (ic >= 0) & (ic < self.n_across)
            z = np.clip(ia, 0, self.n_along - 1) * self.n_across + np.clip(ic, 0, self.n_across - 1)
            if self.outside == "drop":
                z = np.where(inside, z, -1)
            if self.mask is not None:
                z = np.where(np.asarray(self.mask, bool), z, -1)
            self._cache["zone"] = z
        return self._cache["zone"]

    def zone_centers(self) -> np.ndarray:
        L, Wd = self.sole_length_m, self.sole_width_m
        dx, dy = L / self.n_along, Wd / self.n_across
        return np.array([(-L / 2 + dx * (i + 0.5), -Wd / 2 + dy * (j + 0.5))
                         for i in range(self.n_along) for j in range(self.n_across)])


def taxels_to_zones(taxels: np.ndarray, cfg: TaxelGridConfig) -> np.ndarray:
    """[T, H, W, 3] taxel forces (u along rows, v along columns, n normal; N) -> [T, Z, 3] (Fx, Fy, Fz) sole frame."""
    taxels = np.asarray(taxels, dtype=np.float64)
    if taxels.shape[1:] != (cfg.H, cfg.W, 3):
        raise ValueError(f"expected [T, {cfg.H}, {cfg.W}, 3], got {taxels.shape}")
    sx, sy = cfg.signs
    su, sv, sn = cfg.force_sign
    f = np.stack([taxels[..., 0] * su * sx, taxels[..., 1] * sv * sy, taxels[..., 2] * sn], -1)  # sole-frame axes
    z = cfg.zone_of_taxel().reshape(-1)
    keep = z >= 0
    onehot = np.zeros((cfg.Z, z.size))
    onehot[z[keep], np.nonzero(keep)[0]] = 1.0
    return np.einsum("zk,tkc->tzc", onehot, f.reshape(f.shape[0], -1, 3))


def resample_to_rate(t: np.ndarray, x: np.ndarray, rate_hz: float = 50.0, t0: float | None = None):
    """Bin samples (timestamps t [T], seconds) into frames of 1/rate_hz: frame k covers (t0 + k/r, t0 + (k+1)/r],
    value = mean of its samples, stamped t0 + (k+1)/r (as the sim recorder). Returns (t_frames, x_frames, valid);
    frames without samples are zero and valid=False. For 150 Hz input with t = t0 + (i+1)/150 this is exactly
    the mean of 3 consecutive samples."""
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if t0 is None:  # first sample closes its own sample interval, like the sim's 150 Hz bins
        t0 = t[0] - (float(np.median(np.diff(t))) if t.size > 1 else 0.0)
    k = np.ceil((t - t0) * rate_hz - 1e-9).astype(int) - 1
    n = int(k.max()) + 1
    k = np.clip(k, 0, None)
    out = np.zeros((n,) + x.shape[1:])
    cnt = np.zeros(n)
    np.add.at(out, k, x)
    np.add.at(cnt, k, 1)
    valid = cnt > 0
    out[valid] /= cnt[valid].reshape((-1,) + (1,) * (x.ndim - 1))
    return t0 + (np.arange(n) + 1) / rate_hz, out, valid
