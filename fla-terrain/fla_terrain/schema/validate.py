"""Episode validator for schema v0.2."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .constants import (
    ARRAY_SPECS,
    SCHEMA_VERSIONS,
    array_specs,
    META_KEYS,
    MODES,
    SCHEMA_VERSION,
    SOURCES,
    SUPPORT_LABELS,
    TERRAIN_LABELS,
)


def format_episode_id(eid) -> str:
    """Integers become zero-padded 6 digits; strings are used as given."""
    return f"{eid:06d}" if isinstance(eid, (int, np.integer)) else str(eid)


class SchemaError(ValueError):
    """Raised when an episode does not conform to the schema."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def validate_arrays(arrays: dict, meta: dict) -> list[str]:
    """Return a list of error strings (empty if valid)."""
    errs: list[str] = []
    ARRAY_SPECS = array_specs(meta.get("schema_version", SCHEMA_VERSION))  # noqa: N806
    if meta.get("schema_version") == "0.3":
        from .constants import OPTIONAL_V03
        ARRAY_SPECS = {**ARRAY_SPECS, **{k: v for k, v in OPTIONAL_V03.items() if k in arrays}}  # noqa: N806
    missing = [k for k in ARRAY_SPECS if k not in arrays]
    if missing:
        errs.append(f"missing arrays: {missing}")
    extra = [k for k in arrays if k not in ARRAY_SPECS]
    if extra:
        errs.append(f"unexpected arrays: {extra}")

    dims: dict[str, int] = {}
    if isinstance(meta.get("zone_layout"), dict) and "Z" in meta["zone_layout"]:
        dims["Z"] = int(meta["zone_layout"]["Z"])
    if isinstance(meta.get("zone_layout"), dict) and "P" in meta["zone_layout"]:
        dims["P"] = int(meta["zone_layout"]["P"])
    for key, (shape, dtype) in ARRAY_SPECS.items():
        if key not in arrays:
            continue
        a = np.asarray(arrays[key])
        if a.dtype != dtype:
            errs.append(f"{key}: dtype {a.dtype}, expected {dtype}")
        if a.ndim != len(shape):
            errs.append(f"{key}: ndim {a.ndim}, expected {len(shape)} {shape}")
            continue
        for ax, (sym, n) in enumerate(zip(shape, a.shape)):
            if isinstance(sym, int):
                if n != sym:
                    errs.append(f"{key}: axis {ax} is {n}, expected {sym}")
            else:
                if sym in dims and dims[sym] != n:
                    errs.append(f"{key}: {sym}={n} mismatches {sym}={dims[sym]}")
                dims.setdefault(sym, n)
    if errs:
        return errs

    T = dims["T"]
    if T < 1:
        errs.append("T must be >= 1")
    t = arrays["t"]
    if not np.all(np.isfinite(t)):
        errs.append("t: non-finite values")
    elif T > 1 and np.any(np.diff(t) <= 0):
        errs.append("t: not strictly increasing")
    rate = meta.get("rate_hz")
    if isinstance(rate, (int, float)) and rate > 0 and T > 1 and np.all(np.isfinite(t)):
        dt = float(np.median(np.diff(t)))
        if abs(dt * rate - 1.0) > 0.01:
            errs.append(f"t: median dt {dt:.5f} inconsistent with rate_hz {rate}")

    bad = set(np.unique(arrays["label_terrain"]).tolist()) - set(TERRAIN_LABELS)
    if bad:
        errs.append(f"label_terrain: invalid values {sorted(bad)}")
    if "label_ground_L" in arrays:
        from .constants import GROUND_LABELS
        for side in ("L", "R"):
            bad = set(np.unique(arrays[f"label_ground_{side}"]).tolist()) - set(GROUND_LABELS)
            if bad:
                errs.append(f"label_ground_{side}: invalid values {sorted(bad)}")
    for side in ("L", "R"):
        k = f"label_support_{side}"
        bad = set(np.unique(arrays[k]).tolist()) - set(SUPPORT_LABELS)
        if bad:
            errs.append(f"{k}: invalid values {sorted(bad)}")
        f = arrays[f"prop_support_frac_{side}"]
        fin = f[np.isfinite(f)]
        if fin.size and (fin.min() < 0 or fin.max() > 1):
            errs.append(f"prop_support_frac_{side}: values outside [0, 1]")

    for k in ("foot_zone_L", "foot_zone_R", "foot_points_L", "foot_points_R", "joint_pos", "joint_vel", "imu_gyro", "imu_gravity", "cmd_vel"):
        valid = None
        if k not in arrays:
            continue
        if k.startswith("foot_"):
            valid = arrays["foot_valid_" + k[-1]]
        a = arrays[k] if valid is None else arrays[k][valid]
        if not np.all(np.isfinite(a)):
            errs.append(f"{k}: non-finite values in valid frames")
    return errs


def validate_meta(meta: dict) -> list[str]:
    errs: list[str] = []
    missing = [k for k in META_KEYS if k not in meta]
    if missing:
        errs.append(f"missing metadata: {missing}")
    if meta.get("schema_version", SCHEMA_VERSION) not in SCHEMA_VERSIONS:
        errs.append(f"schema_version {meta.get('schema_version')!r} not in {SCHEMA_VERSIONS}")
    sk = meta.get("split_key", None)
    if "split_key" in meta and (not isinstance(sk, str) or not sk.strip()):
        errs.append("split_key must be a non-empty string")
    if "source" in meta and meta["source"] not in SOURCES:
        errs.append(f"source {meta['source']!r} not in {SOURCES}")
    if "mode" in meta and meta["mode"] not in MODES:
        errs.append(f"mode {meta['mode']!r} not in {MODES}")
    for k in ("rate_hz", "sensor_native_rate_hz"):
        if k in meta and not (isinstance(meta[k], (int, float)) and meta[k] > 0):
            errs.append(f"{k} must be a positive number")
    zl = meta.get("zone_layout")
    if "zone_layout" in meta:
        if not isinstance(zl, dict) or "Z" not in zl:
            errs.append("zone_layout must be a dict with Z (and n_across, n_along for grid layouts)")
        elif {"n_across", "n_along"} <= set(zl) and zl["n_across"] * zl["n_along"] != zl["Z"]:
            errs.append("zone_layout: n_across * n_along != Z")
        elif not {"n_across", "n_along"} <= set(zl) and "blocks" not in zl and "points" not in zl:
            errs.append("zone_layout without a grid needs 'blocks' or 'points'")
    if "terrain_params" in meta and not isinstance(meta["terrain_params"], dict):
        errs.append("terrain_params must be a dict")
    return errs


def validate_episode(npz_path, json_path=None, raise_on_error: bool = True) -> list[str]:
    """Validate one episode (npz + json). Returns the list of errors.

    If ``raise_on_error`` (default), raises SchemaError when any error is found.
    ``json_path`` defaults to the npz path with a .json suffix.
    """
    npz_path = Path(npz_path)
    json_path = Path(json_path) if json_path is not None else npz_path.with_suffix(".json")
    errs: list[str] = []
    try:
        meta = json.loads(json_path.read_text())
        if not isinstance(meta, dict):
            raise ValueError("metadata is not a JSON object")
    except Exception as e:  # noqa: BLE001
        errs.append(f"cannot read metadata {json_path.name}: {e}")
        meta = {}
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    except Exception as e:  # noqa: BLE001
        errs.append(f"cannot read arrays {npz_path.name}: {e}")
        arrays = None
    errs += validate_meta(meta)
    if arrays is not None:
        errs += validate_arrays(arrays, meta)
    if meta.get("episode_id") is not None and npz_path.stem != f"episode_{format_episode_id(meta['episode_id'])}":
        errs.append(f"episode_id {meta['episode_id']!r} does not match file name {npz_path.name}")
    if errs and raise_on_error:
        raise SchemaError(errs)
    return errs
