import numpy as np
import pytest

from fla_terrain.data import (
    assign_splits,
    channel_names,
    check_no_leakage,
    load_windows,
    make_synthetic_episode,
    split_windows,
    write_episode,
)

J, Z = 22, 8


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    d = tmp_path_factory.mktemp("eps")
    i = 0
    for k in range(12):
        for terrain, mode in (("hard_grip", "walk"), ("hard_slip", "stand")):
            a, m = make_synthetic_episode(terrain, mode, duration_s=5.0, split_key=f"key_{k:02d}", seed=i)
            write_episode(d, i, a, m)
            i += 1
    return d


@pytest.mark.parametrize("input_set,C", [("P", 2 * J + 9), ("P+N", 2 * J + 9 + 2 * Z), ("P+N+S", 2 * J + 9 + 6 * Z)])
def test_window_shapes(root, input_set, C):
    ws = load_windows(root, input_set)
    # 5 s at 50 Hz = 250 frames; W=50, stride 10 -> (250-50)//10 + 1 = 21 windows per episode
    assert ws.X.shape == (24 * 21, 50, C)
    assert ws.X.dtype == np.float32
    assert len(ws.channels) == C == len(channel_names(input_set, J, Z))
    assert ws.support.shape == (len(ws), 2)
    assert np.isfinite(ws.X).all()


def test_sets_are_nested(root):
    p, pn, pns = (load_windows(root, s) for s in ("P", "P+N", "P+N+S"))
    np.testing.assert_array_equal(pn.X[..., : p.X.shape[2]], p.X)
    np.testing.assert_array_equal(pns.X[..., : pn.X.shape[2]], pn.X)
    # P has no force channel at all
    assert not any(c.startswith("F") for c in p.channels)
    assert all(c.startswith("Fz") for c in pn.channels[p.X.shape[2]:])
    assert all(c[:2] in ("Fx", "Fy") for c in pns.channels[pn.X.shape[2]:])


def test_labels_at_last_frame(tmp_path):
    a, m = make_synthetic_episode("hard_grip", "walk", duration_s=6.0, terrain_after="hard_slip", seed=3)
    write_episode(tmp_path, 0, a, m)
    ws = load_windows(tmp_path, "P")
    last = np.round(ws.t_end * 50).astype(int)
    np.testing.assert_array_equal(ws.terrain, a["label_terrain"][last].astype(np.int8))
    np.testing.assert_allclose(ws.t_end[:3], [0.98, 1.18, 1.38])
    # window content ends at t_end: last frame of the cmd channel equals the episode at t_end
    np.testing.assert_allclose(ws.X[:, -1, 0], a["joint_pos"][last, 0], rtol=1e-6)


def test_splits_by_key_no_leakage(root):
    ws = load_windows(root, "P+N")
    mapping = assign_splits(ws.split_key)
    assert mapping == assign_splits(list(ws.split_key)[::-1])  # deterministic, order independent
    parts = split_windows(ws, mapping)
    assert sum(len(p) for p in parts.values()) == len(ws)
    sets = [set(p.split_key) for p in parts.values()]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not (sets[i] & sets[j])
    assert len(parts["train"]) > 0


def test_leakage_check_detects():
    with pytest.raises(AssertionError, match="leakage"):
        check_no_leakage({"train": ["a", "b"], "val": ["c"], "test": ["b"]})
    check_no_leakage({"train": ["a", "a"], "val": ["c"], "test": ["d"]})


def test_rate_mismatch_rejected(tmp_path):
    a, m = make_synthetic_episode(duration_s=3.0, rate_hz=150.0, seed=0)
    write_episode(tmp_path, 0, a, m)
    with pytest.raises(ValueError, match="rate_hz"):
        load_windows(tmp_path, "P")


def test_invalid_frames_zeroed(tmp_path):
    a, m = make_synthetic_episode("hard_grip", "stand", duration_s=2.0, seed=0)
    a["foot_valid_L"][:] = False
    a["foot_zone_L"][:] = np.nan
    write_episode(tmp_path, 0, a, m)
    ws = load_windows(tmp_path, "P+N+S")
    iL = [i for i, c in enumerate(ws.channels) if c.endswith(tuple(f"_L{z}" for z in range(Z)))]
    assert len(iL) == 3 * Z and np.all(ws.X[..., iL] == 0)
