"""Windowing loader for schema v0.2 episodes.

Windows are 1.0 s long with 0.2 s stride at the model rate (50 Hz -> 50 frames,
stride 10). Labels/targets are taken at the last frame of each window (causal).
Splits are assigned per ``split_key`` only (deterministic hash), so all windows
of one split_key land in the same split.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..schema import INPUT_SETS, MODEL_RATE_HZ, STRIDE_S, WINDOW_S, validate_episode

SPLITS = ("train", "val", "test")


def channel_names(input_set: str, J: int, Z: int) -> list[str]:
    if input_set not in INPUT_SETS:
        raise ValueError(f"input_set {input_set!r} not in {INPUT_SETS}")
    names = [f"q{j}" for j in range(J)] + [f"qd{j}" for j in range(J)]
    names += [f"gyro_{a}" for a in "xyz"] + [f"grav_{a}" for a in "xyz"] + [f"cmd_{a}" for a in ("vx", "vy", "wz")]
    if "N" in input_set:
        names += [f"Fz_{s}{z}" for s in "LR" for z in range(Z)]
    if "S" in input_set:
        names += [f"F{a}_{s}{z}" for s in "LR" for a in "xy" for z in range(Z)]
    return names


def episode_features(arrays: dict, input_set: str) -> np.ndarray:
    """Per-frame feature matrix [T, C] for an input set. Invalid sensor frames are zeroed."""
    parts = [arrays["joint_pos"], arrays["joint_vel"], arrays["imu_gyro"], arrays["imu_gravity"], arrays["cmd_vel"]]
    feet = {}
    for s in "LR":
        f = np.nan_to_num(np.asarray(arrays[f"foot_zone_{s}"], np.float32))
        f[~np.asarray(arrays[f"foot_valid_{s}"], bool)] = 0.0
        feet[s] = f
    if "N" in input_set:
        parts += [feet[s][..., 2] for s in "LR"]
    if "S" in input_set:
        parts += [feet[s][..., a] for s in "LR" for a in (0, 1)]
    return np.concatenate([np.asarray(p, np.float32).reshape(len(arrays["t"]), -1) for p in parts], axis=1)


@dataclass
class WindowSet:
    input_set: str
    X: np.ndarray            # [N, W, C] float32
    terrain: np.ndarray      # [N] int8, label at last frame (-1 unknown)
    support: np.ndarray      # [N, 2] int8 (L, R)
    mu: np.ndarray           # [N] float32 (NaN unknown)
    stiffness: np.ndarray    # [N]
    slope: np.ndarray        # [N]
    split_key: np.ndarray    # [N] str
    episode: np.ndarray      # [N] episode_id str
    mode: np.ndarray         # [N] str
    t_end: np.ndarray        # [N] time of last frame
    range_split: np.ndarray  # [N] str: split fixed at generation (terrain_params.range_split) or ''
    channels: list[str] = field(default_factory=list)

    def __len__(self):
        return len(self.X)

    def subset(self, mask) -> "WindowSet":
        kw = {k: getattr(self, k)[mask] for k in ("X", "terrain", "support", "mu", "stiffness", "slope",
                                                  "split_key", "episode", "mode", "t_end", "range_split")}
        return WindowSet(self.input_set, channels=self.channels, **kw)


def _episode_paths(root_or_paths) -> list[Path]:
    if isinstance(root_or_paths, (str, Path)):
        p = Path(root_or_paths)
        return sorted(p.rglob("episode_*.npz")) if p.is_dir() else [p]
    return [Path(p) for p in root_or_paths]


def window_starts(T: int, W: int, S: int) -> np.ndarray:
    return np.arange(0, T - W + 1, S) if T >= W else np.zeros(0, int)


def load_windows(root_or_paths, input_set: str = "P+N+S", window_s: float = WINDOW_S, stride_s: float = STRIDE_S,
                 rate_hz: float = MODEL_RATE_HZ, validate: bool = True, drop_after_fall: bool = True) -> WindowSet:
    """Load episodes and cut them into windows for one input set.
    drop_after_fall: drop windows ending after the episode's fall_time (falls are kept out of training)."""
    W = int(round(window_s * rate_hz))
    S = int(round(stride_s * rate_hz))
    acc = {k: [] for k in ("X", "terrain", "support", "mu", "stiffness", "slope", "split_key", "episode", "mode", "t_end", "range_split")}
    channels = None
    for p in _episode_paths(root_or_paths):
        if validate:
            validate_episode(p)
        meta = json.loads(p.with_suffix(".json").read_text())
        if abs(float(meta["rate_hz"]) - rate_hz) > 1e-6:
            raise ValueError(f"{p.name}: rate_hz {meta['rate_hz']} != loader rate {rate_hz}")
        with np.load(p) as z:
            a = {k: z[k] for k in z.files}
        T = len(a["t"])
        st = window_starts(T, W, S)
        if len(st) == 0:
            continue
        F = episode_features(a, input_set)
        ch = channel_names(input_set, a["joint_pos"].shape[1], a["foot_zone_L"].shape[1])
        if channels is None:
            channels = ch
        elif ch != channels:
            raise ValueError(f"{p.name}: channel layout differs from earlier episodes")
        last = st + W - 1
        ft = meta.get("fall_time")
        if drop_after_fall and ft is not None:
            keep = a["t"][last] <= ft
            st, last = st[keep], last[keep]
            if len(st) == 0:
                continue
        idx = st[:, None] + np.arange(W)[None, :]
        acc["X"].append(F[idx])
        acc["terrain"].append(a["label_terrain"][last])
        acc["support"].append(np.stack([a["label_support_L"][last], a["label_support_R"][last]], 1))
        acc["mu"].append(a["prop_mu"][last])
        acc["stiffness"].append(a["prop_stiffness"][last])
        acc["slope"].append(a["prop_slope"][last])
        n = len(st)
        acc["split_key"].append(np.full(n, meta["split_key"], dtype=object))
        acc["episode"].append(np.full(n, str(meta["episode_id"]), dtype=object))
        acc["mode"].append(np.full(n, meta["mode"], dtype=object))
        acc["t_end"].append(a["t"][last])
        rs = (meta.get("terrain_params") or {}).get("range_split") or ""
        acc["range_split"].append(np.full(n, rs, dtype=object))
    if channels is None:
        raise ValueError("no windows: no episodes found or all shorter than one window")
    out = {k: np.concatenate(v) for k, v in acc.items()}
    return WindowSet(input_set, channels=channels, **out)


def assign_splits(split_keys, fractions=(0.7, 0.15, 0.15), salt: str = "fla0") -> dict:
    """Deterministic split per unique split_key by hashing. Returns {split_key: split}."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    edges = np.cumsum(fractions)
    out = {}
    for k in sorted(set(split_keys)):
        u = int(hashlib.sha256(f"{salt}:{k}".encode()).hexdigest()[:12], 16) / float(16 ** 12)
        out[k] = SPLITS[int(np.searchsorted(edges, u, side="right").clip(0, 2))]
    return out


def make_split_mapping(ws: WindowSet, mode: str = "auto", fractions=(0.7, 0.15, 0.15), salt: str = "fla0") -> dict:
    """split_key -> split. mode 'range': use the generation-time range_split (must be set and consistent per
    key); 'hash': hash of split_key; 'auto': 'range' if every window has a range_split and more than one
    split value occurs, else 'hash'."""
    keys = np.asarray(ws.split_key, dtype=object)
    rs = np.asarray(ws.range_split, dtype=object)
    if mode == "auto":
        mode = "range" if all(rs) and len(set(rs.tolist())) > 1 else "hash"
    if mode == "hash":
        return assign_splits(keys, fractions, salt)
    out = {}
    for k, r in zip(keys.tolist(), rs.tolist()):
        if r not in SPLITS:
            raise ValueError(f"split_key {k!r} has no valid range_split ({r!r})")
        if out.setdefault(k, r) != r:
            raise AssertionError(f"split_key {k!r} has inconsistent range_split")
    return out


def split_windows(ws: WindowSet, mapping: dict) -> dict:
    """{split: WindowSet} using a split_key -> split mapping; checks leakage."""
    sp = np.array([mapping[k] for k in ws.split_key], dtype=object)
    parts = {s: ws.subset(sp == s) for s in SPLITS}
    check_no_leakage({s: w.split_key for s, w in parts.items()})
    return parts


def check_no_leakage(keys_by_split: dict) -> None:
    """Raise AssertionError if any split_key appears in more than one split."""
    seen: dict = {}
    for split, keys in keys_by_split.items():
        for k in set(np.asarray(keys, dtype=object).tolist()):
            if k in seen and seen[k] != split:
                raise AssertionError(f"leakage: split_key {k!r} in both {seen[k]} and {split}")
            seen[k] = split
