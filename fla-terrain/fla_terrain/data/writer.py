"""Write one schema v0.2 episode (npz + json), validated before and after writing."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..schema import SCHEMA_VERSION, array_specs, SchemaError, format_episode_id, validate_arrays, validate_episode, validate_meta


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def cast_arrays(arrays: dict, version: str = SCHEMA_VERSION) -> dict:
    """Cast every schema array to its canonical dtype (other keys are passed through)."""
    specs = array_specs(version)
    if version == "0.3":
        from ..schema.constants import OPTIONAL_V03
        specs = {**specs, **OPTIONAL_V03}
    out = {}
    for k, v in arrays.items():
        out[k] = np.ascontiguousarray(np.asarray(v).astype(specs[k][1], copy=False)) if k in specs else v
    return out


def write_episode(out_dir, episode_id, arrays: dict, meta: dict, compress: bool = True) -> Path:
    """Cast, validate and write ``episode_<id>.npz`` / ``.json``. Returns the npz path.

    Raises SchemaError (and writes nothing) if the episode is invalid.
    """
    out_dir = Path(out_dir)
    meta = _jsonable(dict(meta))
    meta.setdefault("schema_version", SCHEMA_VERSION)
    meta["episode_id"] = format_episode_id(episode_id)
    arrays = cast_arrays(arrays, meta["schema_version"])
    errs = validate_meta(meta) + validate_arrays(arrays, meta)
    if errs:
        raise SchemaError(errs)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"episode_{meta['episode_id']}"
    npz = out_dir / f"{stem}.npz"
    js = out_dir / f"{stem}.json"
    tmp = out_dir / f".{stem}.tmp.npz"
    (np.savez_compressed if compress else np.savez)(tmp, **arrays)
    tmp.replace(npz)
    js.write_text(json.dumps(meta, indent=1, sort_keys=True))
    validate_episode(npz, js)
    return npz
