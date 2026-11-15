import numpy as np
import pytest

from fla_terrain.schema import SCHEMA_VERSION_V3, validate_arrays, validate_meta
from fla_terrain.sim.pointmap import (PointLayout, PointSensorModel, SoleV2Params, distribute_shear, solve_layer,
                                      views)


@pytest.fixture(scope="module")
def L():
    return PointLayout()


def test_layout(L):
    assert L.P == 19 * 8
    assert np.allclose(L.block_w.sum(1), 1.0)
    assert (L.part == 1).sum() > 0 and set(np.unique(L.part)) == {0, 1, 2}
    assert np.isclose(np.abs(L.xy[:, 0]).max(), 0.09) and np.isclose(np.abs(L.xy[:, 1]).max(), 0.035)
    assert PointLayout(mask="g1").P < L.P


@pytest.mark.parametrize("cop", [(0.0, 0.0), (0.05, -0.02), (-0.09, -0.035), (0.09, 0.035), (-0.089, 0.0)])
def test_solve_layer_exact(L, cop):
    rng = np.random.default_rng(0)
    g = np.abs(rng.normal(0, 3e-4, L.P))
    p = solve_layer(L.xy, g, 100.0, np.array(cop), 2e5)
    assert (p >= 0).all()
    assert np.isclose(p.sum(), 100.0, rtol=1e-4)
    assert np.allclose((p @ L.xy) / p.sum(), cop, atol=1e-5)


def test_void_points_zero(L):
    g = np.where(L.xy[:, 1] > 0, 0.0, 0.1)  # -y half 10 cm above ground
    p = solve_layer(L.xy, g, 80.0, np.array([0.0, 0.02]), 2e5)
    assert np.all(p[L.xy[:, 1] < 0] == 0)


def test_shear_sum_and_torsion(L):
    p = solve_layer(L.xy, np.zeros(L.P), 100.0, np.array([0.01, 0.0]), 2e5)
    S = distribute_shear(L.xy, p, np.array([20.0, -5.0]), np.array([0.01, 0.0]), 1.5)
    assert np.allclose(S.sum(0), [20.0, -5.0])
    r = L.xy - [0.01, 0.0]
    assert np.isclose((r[:, 0] * S[:, 1] - r[:, 1] * S[:, 0]).sum(), 1.5)


def test_spread_preserves_sum(L):
    prm = SoleV2Params(crosstalk=0.0, gain_std=0.0, offset_std=0.0, noise_normal=0.0, noise_shear=0.0, quant=0.0)
    sm = PointSensorModel(L, prm, np.random.default_rng(0))
    f = np.zeros((L.P, 3))
    f[10, 2] = 50.0
    y = sm.apply(f, 0)
    assert np.isclose(y[:, 2].sum(), 50.0) and (y[:, 2] > 0).sum() > 1


def test_views_and_schema(L):
    T = 5
    pts = np.random.default_rng(0).random((T, L.P, 3)).astype(np.float32)
    tab = L.table()
    v = views(pts, tab)
    assert v["N_blk"].shape == (T, 4) and v["S_blk"].shape == (T, 4, 2)
    mid = L.part == 1
    assert np.allclose(v["N_blk"].sum(1), pts[:, ~mid, 2].sum(1), rtol=1e-5)
    meta = {"schema_version": SCHEMA_VERSION_V3, "zone_layout": tab, "rate_hz": 50.0}
    arr = {"t": np.arange(1, T + 1) / 50.0, "foot_points_L": pts, "foot_points_R": pts,
           "foot_valid_L": np.ones(T, bool), "foot_valid_R": np.ones(T, bool),
           "joint_pos": np.zeros((T, 22), np.float32), "joint_vel": np.zeros((T, 22), np.float32),
           "imu_gyro": np.zeros((T, 3), np.float32), "imu_gravity": np.zeros((T, 3), np.float32),
           "cmd_vel": np.zeros((T, 3), np.float32), "label_terrain": np.zeros(T, np.int8),
           "label_support_L": np.zeros(T, np.int8), "label_support_R": np.zeros(T, np.int8),
           "prop_mu": np.ones(T, np.float32), "prop_stiffness": np.ones(T, np.float32),
           "prop_slope": np.zeros(T, np.float32), "prop_support_frac_L": np.ones(T, np.float32),
           "prop_support_frac_R": np.ones(T, np.float32), "contact_gt_L": np.ones(T, bool),
           "contact_gt_R": np.ones(T, bool), "label_ground_L": np.full(T, 2, np.int8),
           "label_ground_R": np.full(T, 2, np.int8)}
    arr["t"] = arr["t"].astype(np.float64)
    assert validate_arrays(arr, meta) == []
    assert validate_arrays(dict(arr, joint_target=np.zeros((T, 22), np.float32)), meta) == []
    assert any("label_ground_L" in e for e in validate_arrays(dict(arr, label_ground_L=np.full(T, 9, np.int8)), meta))
    bad = dict(arr, foot_points_L=pts[:, :10])
    assert any("foot_points_L" in e for e in validate_arrays(bad, meta))
    assert not [e for e in validate_meta({**meta}) if "zone_layout" in e or "schema_version" in e]


def test_param_sampling():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = SoleV2Params.sample(rng)
        lo, hi = SoleV2Params.RANGES["compliance"]
        assert lo <= p.compliance <= hi and p.boundaries[0] < p.boundaries[1] and not p.nominal
