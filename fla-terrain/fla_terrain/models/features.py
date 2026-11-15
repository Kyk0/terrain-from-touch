"""Hand features per window for the classical baseline.

P: per channel mean/std/last/min/max (joints, joint velocities, gyro, gravity, command).
N (+Fz per zone): per zone mean/std/max Fz; per foot total Fz mean/std/max, CoP x/y mean/std, CoP
   velocity std, max Fz rise rate (touchdown), mean contact area (zones above 2 N), loaded fraction.
S (+Fx, Fy): per zone mean |shear|/Fz over loaded frames; per foot |shear|/Fz mean/median/p90/max over
   loaded frames, Fx/Fy mean/std.
"""
from __future__ import annotations

import numpy as np

LOAD_N = 5.0  # N, frame counts as loaded for a foot / zone ratio


def _stats(x):  # x [N, W, C] -> [N, 5C]
    return np.concatenate([x.mean(1), x.std(1), x[:, -1], x.min(1), x.max(1)], 1)


def window_features(X: np.ndarray, channels: list[str], zone_centers: np.ndarray, rate_hz: float = 50.0):
    ci = {c: i for i, c in enumerate(channels)}
    pidx = [i for i, c in enumerate(channels) if not c.startswith("F")]
    feats = [_stats(X[:, :, pidx])]
    names = [f"{s}_{channels[i]}" for s in ("mean", "std", "last", "min", "max") for i in pidx]
    Z = len(zone_centers)
    has_n = f"Fz_L0" in ci
    has_s = f"Fx_L0" in ci
    for side in "LR":
        if not has_n:
            break
        fz = X[:, :, [ci[f"Fz_{side}{z}"] for z in range(Z)]]  # [N, W, Z]
        tot = fz.sum(2)
        loaded = tot > LOAD_N
        cop = (np.clip(fz, 0, None)[..., None] * zone_centers[None, None]).sum(2) / np.maximum(
            np.clip(fz, 0, None).sum(2), 1e-6)[..., None]  # [N, W, 2]
        copv = np.diff(cop, axis=1) * rate_hz
        f = [fz.mean(1), fz.std(1), fz.max(1),
             np.stack([tot.mean(1), tot.std(1), tot.max(1), np.diff(tot, axis=1).max(1) * rate_hz,
                       (fz > 2.0).sum(2).mean(1), loaded.mean(1)], 1),
             _masked_stats(cop, loaded), (copv * loaded[:, 1:, None]).std(1)]
        feats += f
        names += ([f"FzMean_{side}{z}" for z in range(Z)] + [f"FzStd_{side}{z}" for z in range(Z)] +
                  [f"FzMax_{side}{z}" for z in range(Z)] +
                  [f"{side}_{n}" for n in ("totMean", "totStd", "totMax", "riseMax", "area", "loadedFrac")] +
                  [f"{side}_cop{a}_{s}" for s in ("mean", "std") for a in "xy"] + [f"{side}_copv{a}_std" for a in "xy"])
        if has_s:
            fx = X[:, :, [ci[f"Fx_{side}{z}"] for z in range(Z)]]
            fy = X[:, :, [ci[f"Fy_{side}{z}"] for z in range(Z)]]
            sh = np.hypot(fx, fy)
            zl = fz > LOAD_N / Z
            zr = np.where(zl, sh / np.maximum(fz, 1e-6), np.nan)
            with np.errstate(all="ignore"):
                zr_mean = np.nan_to_num(np.nanmean(zr, 1), nan=-1)
            Fx, Fy = fx.sum(2), fy.sum(2)
            rr = np.where(loaded, np.hypot(Fx, Fy) / np.maximum(tot, 1e-6), np.nan)
            with np.errstate(all="ignore"):
                rs = np.nan_to_num(np.stack([np.nanmean(rr, 1), np.nanmedian(rr, 1), np.nanpercentile(rr, 90, 1),
                                             np.nanmax(rr, 1)], 1), nan=-1)
            feats += [zr_mean, rs, np.stack([Fx.mean(1), Fx.std(1), Fy.mean(1), Fy.std(1)], 1)]
            names += ([f"zoneRatio_{side}{z}" for z in range(Z)] +
                      [f"{side}_ratio_{s}" for s in ("mean", "median", "p90", "max")] +
                      [f"{side}_{s}" for s in ("FxMean", "FxStd", "FyMean", "FyStd")])
    F = np.concatenate(feats, 1).astype(np.float32)
    return F, names


def _masked_stats(v, mask):  # v [N, W, 2], mask [N, W] -> mean x,y and std x,y over masked frames
    m = mask[..., None].astype(float)
    n = np.maximum(m.sum(1), 1)
    mean = (v * m).sum(1) / n
    std = np.sqrt((((v - mean[:, None]) ** 2) * m).sum(1) / n)
    return np.concatenate([mean, std], 1)
