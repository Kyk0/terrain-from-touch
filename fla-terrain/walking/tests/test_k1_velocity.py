import mujoco
import numpy as np

from walking import K1VelocityPolicy, load_k1_model


def test_walks_forward_2s():
    model = load_k1_model(friction=1.0)
    data = mujoco.MjData(model)
    src = K1VelocityPolicy("G113_m2200")
    assert src.physics_dt == model.opt.timestep
    src.reset(model, data, np.random.default_rng(0))
    x0 = data.qpos[0]
    for _ in range(int(2.0 / model.opt.timestep)):
        src.pre_step(model, data, (0.4, 0.0, 0.0))
        mujoco.mj_step(model, data)
    assert not src.fallen(model, data)
    assert src.base_height_above_feet(data) > 0.4  # upright (standing ~0.49)
    assert data.qpos[0] - x0 > 0.3                 # moved forward (~0.6 m expected)


def test_fallen_detects_lying_robot():
    model = load_k1_model()
    data = mujoco.MjData(model)
    src = K1VelocityPolicy()
    src.reset(model, data, np.random.default_rng(0), base_pos=(0, 0, 0.15))
    data.qpos[3:7] = (np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0)  # rolled 90 deg
    mujoco.mj_forward(model, data)
    assert src.fallen(model, data)
