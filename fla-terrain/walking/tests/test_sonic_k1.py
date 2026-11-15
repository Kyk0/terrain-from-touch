import os

import mujoco
import numpy as np
import pytest

from walking.sonic_k1 import CLIPS, WEIGHTS, SONIC_DIR

pytestmark = pytest.mark.skipif(not (os.path.exists(WEIGHTS) and os.path.isdir(CLIPS)),
                                reason="SONIC export missing: run scripts/walk_sonic_export.py")


def test_actor_matches_torch():
    from walking.sonic_k1 import SonicActor
    p = np.load(os.path.join(SONIC_DIR, "actor_parity.npz"))
    assert np.abs(SonicActor()(p["prop"], p["tok"]) - p["action"]).max() < 1e-4


def test_sonic_walks_3s_with_clip_switch():
    from walking import load_k1_model
    from walking.sonic_k1 import SonicK1
    model = load_k1_model()
    data = mujoco.MjData(model)
    src = SonicK1()
    src.reset(model, data, np.random.default_rng(0), clip="neutral_walk_ff_180_R_002__A535")
    x0 = data.qpos[:2].copy()
    for _ in range(int(3.0 / model.opt.timestep)):
        src.pre_step(model, data)
        mujoco.mj_step(model, data)
    assert not src.fallen(model, data)
    assert np.linalg.norm(data.qpos[:2] - x0) > 0.3          # ~1.1 m expected
    # chaining: at the end of a clip playback jumps to another clip, heading re-anchored to the robot
    from walking.sonic_k1 import yaw_of
    src.frame = int(src.clip["T"]) - 2 - src.F * src.skip
    src._advance(data)
    assert len(src.clip_log) == 2 and src.lib.names[src.ci] != "neutral_walk_ff_180_R_002__A535"
    dy = yaw_of(src.ref_quat(src.frame)) - yaw_of(data.qpos[3:7])
    assert abs((dy + np.pi) % (2 * np.pi) - np.pi) < 1e-6
