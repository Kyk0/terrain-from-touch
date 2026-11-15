import json

import numpy as np
import pytest

from fla_terrain.data import cast_arrays, make_synthetic_episode, write_episode
from fla_terrain.schema import SchemaError, validate_arrays, validate_episode, validate_meta


@pytest.fixture
def ep():
    a, m = make_synthetic_episode("hard_grip", "walk", duration_s=4.0, seed=1)
    return cast_arrays(a), m


def test_valid_episode_passes(ep):
    a, m = ep
    assert validate_meta(m) == []
    assert validate_arrays(a, m) == []


@pytest.mark.parametrize("terrain,mode,after", [("hard_slip", "stand", None), ("edge", "walk", None),
                                                ("soft", "lifted", None), ("hard_grip", "walk", "hard_slip"),
                                                ("slope", "rig", None), ("rough", "walk", "edge")])
def test_all_modes_and_terrains_valid(terrain, mode, after):
    a, m = make_synthetic_episode(terrain, mode, duration_s=3.0, terrain_after=after, seed=2)
    assert validate_meta(m) + validate_arrays(cast_arrays(a), m) == []


def test_write_roundtrip(tmp_path, ep):
    a, m = ep
    p = write_episode(tmp_path, 7, a, m)
    assert p.name == "episode_000007.npz"
    assert validate_episode(p) == []
    with np.load(p) as z:
        assert z["foot_zone_L"].dtype == np.float32 and z["t"].dtype == np.float64
        np.testing.assert_array_equal(z["label_terrain"], a["label_terrain"])


@pytest.mark.parametrize("key,bad", [
    ("foot_zone_L", lambda x: x[:, :, :2]),          # wrong last axis
    ("foot_zone_R", lambda x: x[:, :4]),             # Z mismatch vs layout
    ("joint_pos", lambda x: x[:-1]),                 # T mismatch
    ("imu_gyro", lambda x: x[:, 0]),                 # wrong ndim
    ("joint_vel", lambda x: x[:, :5]),               # J mismatch between joint arrays
])
def test_rejects_wrong_shapes(ep, key, bad):
    a, m = ep
    a = dict(a)
    a[key] = bad(a[key])
    errs = validate_arrays(a, m)
    assert errs and any(key in e for e in errs)


def test_rejects_wrong_dtype(ep):
    a, m = ep
    a = dict(a)
    a["foot_zone_L"] = a["foot_zone_L"].astype(np.float64)
    assert any("dtype" in e for e in validate_arrays(a, m))


def test_rejects_missing_array(ep):
    a, m = ep
    a = dict(a)
    del a["contact_gt_R"]
    assert any("missing arrays" in e for e in validate_arrays(a, m))


@pytest.mark.parametrize("value", [None, "", "  ", 3])
def test_rejects_bad_split_key(ep, value):
    _, m = ep
    m = dict(m, split_key=value)
    assert any("split_key" in e for e in validate_meta(m))


def test_rejects_missing_split_key(tmp_path, ep):
    a, m = ep
    m = {k: v for k, v in m.items() if k != "split_key"}
    assert any("split_key" in e for e in validate_meta(m))
    with pytest.raises(SchemaError):
        write_episode(tmp_path, 1, a, m)
    assert not list(tmp_path.glob("*"))  # nothing written
    # also on disk: a valid episode whose json loses split_key
    m_ok = dict(m, split_key="k")
    p = write_episode(tmp_path, 2, a, m_ok)
    js = p.with_suffix(".json")
    d = json.loads(js.read_text())
    del d["split_key"]
    js.write_text(json.dumps(d))
    with pytest.raises(SchemaError, match="split_key"):
        validate_episode(p)


def test_rejects_bad_labels_and_meta(ep):
    a, m = ep
    a = dict(a)
    a["label_terrain"] = a["label_terrain"].copy()
    a["label_terrain"][0] = 9
    a["label_support_L"] = a["label_support_L"].copy()
    a["label_support_L"][0] = 3
    errs = validate_arrays(a, m)
    assert any("label_terrain" in e for e in errs) and any("label_support_L" in e for e in errs)
    assert validate_meta(dict(m, mode="fly"))
    assert validate_meta(dict(m, source="cloud"))
    assert validate_meta(dict(m, schema_version="0.1"))
    assert validate_meta(dict(m, zone_layout={"n_across": 2, "n_along": 4, "Z": 9}))


def test_rejects_time_and_rate(ep):
    a, m = ep
    b = dict(a, t=a["t"][::-1].copy())
    assert any("increasing" in e for e in validate_arrays(b, m))
    assert any("rate_hz" in e for e in validate_arrays(a, dict(m, rate_hz=150.0)))


def test_rejects_nonfinite_valid_forces_but_allows_invalid_frames(ep):
    a, m = ep
    a = dict(a)
    f = a["foot_zone_L"].copy()
    f[3] = np.nan
    a["foot_zone_L"] = f
    assert any("non-finite" in e for e in validate_arrays(a, m))
    v = a["foot_valid_L"].copy()
    v[3] = False
    a["foot_valid_L"] = v
    assert validate_arrays(a, m) == []


def test_episode_id_must_match_filename(tmp_path, ep):
    a, m = ep
    p = write_episode(tmp_path, 3, a, m)
    p2 = p.with_name("episode_000004.npz")
    p.rename(p2)
    p.with_suffix(".json").rename(p2.with_suffix(".json"))
    assert any("episode_id" in e for e in validate_episode(p2, raise_on_error=False))
