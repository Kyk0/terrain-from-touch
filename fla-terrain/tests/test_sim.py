import mujoco
import numpy as np
import pytest

from fla_terrain.schema import validate_episode
from fla_terrain.sim import EpisodeRecorder, SoleConfig, SoleSensor, StandMotionSource, build_model, k1


@pytest.fixture(scope="module")
def flat():
    return build_model({"type": "flat", "mu": 0.7})


def test_build_keeps_robot(flat):
    m, info = flat
    base = mujoco.MjModel.from_xml_path(k1.K1_XML)
    assert m.nq == base.nq and m.nv == base.nv and m.nu == base.nu
    assert np.isclose(info["mass"], base.body_subtreemass[1])
    np.testing.assert_allclose(m.body_mass, base.body_mass)
    assert [m.joint(i).name for i in range(m.njnt)] == [base.joint(i).name for i in range(base.njnt)]
    assert [m.actuator(i).name for i in range(m.nu)] == [base.actuator(i).name for i in range(base.nu)]
    assert all(len(info["zone_geoms"][s]) == 8 for s in "LR")
    # terrain wins the contact parameter mixing
    t = info["terrain_geoms"][0]
    assert m.geom_priority[t] > m.geom_priority[info["zone_geoms"]["L"][0]]
    assert np.isclose(m.geom_friction[t, 0], 0.7)


def test_zones_tile_footprint():
    cfg = SoleConfig(n_across=3, n_along=5)
    m, info = build_model(sole_cfg=cfg)
    gz = info["zone_geoms"]["L"]
    assert len(gz) == 15
    area = sum(4 * m.geom_size[g][0] * m.geom_size[g][1] for g in gz)
    hx, hy, _ = k1.FOOT_BOX_HALF
    assert np.isclose(area, 4 * hx * hy)
    assert cfg.layout()["Z"] == 15


def test_stand_and_lifted_forces(flat):
    m, info = flat
    d = mujoco.MjData(m)
    src = StandMotionSource(info)
    src.reset(m, d)
    S = SoleSensor(m)
    for _ in range(750):
        src.pre_step(m, d, None)
        mujoco.mj_step(m, d)
    F = S.read(d)
    fz = F["L"][:, 2].sum() + F["R"][:, 2].sum()
    assert abs(fz / (info["mass"] * 9.81) - 1) < 0.05
    assert (F["L"][:, 2] >= 0).all()
    # lift the robot far above the ground: all zero
    d.qpos[2] += 1.0
    mujoco.mj_forward(m, d)
    F = S.read(d)
    assert all(np.all(F[s] == 0) for s in "LR")


def test_recorder_writes_valid_episode(flat, tmp_path):
    m, info = flat
    d = mujoco.MjData(m)
    src = StandMotionSource(info)
    src.reset(m, d)
    rec = EpisodeRecorder(m, info, motion_source=src.name, mode="stand", split_key="unit", rng=np.random.default_rng(0))
    rec.reset(d)
    for _ in range(500):  # 1 s
        src.pre_step(m, d, None)
        mujoco.mj_step(m, d)
        rec.step(d)
    p = rec.write(tmp_path, 1)
    assert validate_episode(p) == []
    with np.load(p) as z:
        assert z["foot_zone_L"].shape == (50, 8, 3)
        assert z["joint_pos"].shape == (50, 22)
        np.testing.assert_allclose(z["t"][:2], [0.02, 0.04])
        assert np.isclose(z["prop_mu"][0], 0.7)
