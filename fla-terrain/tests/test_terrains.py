import copy

import mujoco
import numpy as np
import pytest

from fla_terrain.data import load_windows, make_split_mapping, make_synthetic_episode, split_windows, write_episode
from fla_terrain.sim import build_model
from fla_terrain.sim import terrains as T


def test_bands_disjoint():
    assert T.check_band_overlap() == []


def test_overlap_detected(monkeypatch):
    bands = copy.deepcopy(T.BANDS)
    bands["mu_slip"]["val"] = [("val", 0.18, 0.21)]  # overlaps train_a
    monkeypatch.setattr(T, "BANDS", bands)
    assert any("overlaps" in e for e in T.check_band_overlap())


@pytest.mark.parametrize("cls", T.CLASSES)
def test_every_class_builds(cls):
    rng = np.random.default_rng(1)
    cfg = T.sample_terrain(cls, rng, "test")
    m, info = build_model(cfg)
    assert len(info["terrain_geoms"]) == len(cfg["geoms"])
    assert cfg["params"]["range_split"] == "test"
    mujoco.mj_forward(m, mujoco.MjData(m))


def test_slip_defined_by_low_friction():
    rng = np.random.default_rng(0)
    for s in T.SPLITS:
        for _ in range(200):
            slip = T.sample_terrain("hard_slip", rng, s)["regions"][0]["mu"]
            grip = T.sample_terrain("rough", rng, s)["regions"][0]["mu"]
            assert slip < 0.4 < 0.65 < grip


def test_range_split_honoured(tmp_path):
    for i, rs in enumerate(["train", "train", "val", "test"]):
        a, m = make_synthetic_episode(duration_s=2.0, split_key=f"k{i}", seed=i)
        m["terrain_params"]["range_split"] = rs
        write_episode(tmp_path, i, a, m)
    ws = load_windows(tmp_path, "P")
    mp = make_split_mapping(ws)
    assert mp == {"k0": "train", "k1": "train", "k2": "val", "k3": "test"}
    parts = split_windows(ws, mp)
    assert set(parts["test"].split_key) == {"k3"}
