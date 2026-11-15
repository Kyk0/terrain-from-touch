import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from gen_data import run_episode
    d = tmp_path_factory.mktemp("v2")
    for i, split in enumerate(("train", "train", "val", "test")):
        run_episode({"terrain": "hard_grip" if i % 2 == 0 else "hard_slip", "variant": None, "source": "walk_g113",
                     "split": split, "seconds": 3.0, "seed": 1, "index": i, "out": str(d), "episode_id": f"t{i}",
                     "split_key": f"t{i}", "sole": "v2", "sole_setting": "nominal"})
    from fla_terrain.models.dataset_v2 import FrameDataV2
    return d, FrameDataV2.build(str(d), workers=2)


@pytest.mark.parametrize("view", ["P+Nreal+Sreal", "P+Nblk+Sblk", "Sreal"])
def test_saved_model_reproduces_predictions(tiny, tmp_path, view):
    from fla_terrain.models.v2 import predict_saved, train_v2
    d, fd = tiny
    sp = fd.split_indices()
    assert all(len(v) for v in sp.values())
    _, pred, _ = train_v2(fd, view, 0, sp, "m", str(d), epochs=1, device="cpu", models_root=str(tmp_path),
                          log=lambda *a: None)
    mdir = tmp_path / "m"
    assert (mdir / "model.pt").exists() and (mdir / "config.json").exists()
    stored = np.load(mdir / "test_pred.npz")
    again = predict_saved(str(mdir), fd)
    for k in ("terrain", "sup_L", "sup_R"):
        np.testing.assert_array_equal(again[k], stored[k])
    for k in ("mu", "logk", "slope"):
        np.testing.assert_allclose(again[k], stored[k], rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(stored["idx"], sp["test"])


def test_views_shapes(tiny):
    from fla_terrain.models.v2 import ViewInputs
    _, fd = tiny
    sp = fd.split_indices()
    vi = ViewInputs(fd, "P+Nreal+Sreal+Nblk", "cpu", train_idx=sp["train"])
    vec, maps = vi.batch(torch.as_tensor(sp["test"][:3]))
    assert vec.shape == (3, 50, fd.n_p + 8)
    assert maps.shape == (3, 50, 2, 4, 19, 8)  # Fz, Fx, Fy + mask
    vi2 = ViewInputs(fd, "Pcmd", "cpu", train_idx=sp["train"])
    assert vi2.c_vec == fd.n_p + 3 and vi2.c_map == 0
