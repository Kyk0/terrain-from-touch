"""Tests for fla_terrain.real.convert (the independent evaluation, B12)."""
import numpy as np
import pytest

from fla_terrain.real.convert import TaxelGridConfig, resample_to_rate, taxels_to_zones


def _patch(cfg, x, y, f, T=6):
    """taxel array with force f=(Fx, Fy, Fz) in the SOLE frame on the taxel nearest to sole point (x, y)."""
    xy = cfg.taxel_xy()
    r, c = np.unravel_index(np.argmin(((xy - [x, y]) ** 2).sum(-1)), xy.shape[:2])
    sx, sy = cfg.signs
    a = np.zeros((T, cfg.H, cfg.W, 3))
    a[:, r, c] = [f[0] * sx, f[1] * sy, f[2]]          # taxel axes = sole axes up to the grid flips
    return a


@pytest.mark.parametrize("row0,col0", [("heel", "-y"), ("toe", "+y"), ("toe", "-y")])
def test_patch_lands_in_right_zone_and_frame(row0, col0):
    cfg = TaxelGridConfig(row0_at=row0, col0_at=col0)
    # toe-left corner -> i_along 3, i_across 1 -> zone 7; heel-right -> zone 0
    for (x, y), zone in (((0.07, 0.02), 7), ((-0.07, -0.02), 0), ((0.02, -0.02), 4)):
        out = taxels_to_zones(_patch(cfg, x, y, (3.0, -2.0, 50.0)), cfg)
        assert out.shape == (6, 8, 3)
        assert np.argmax(out[0, :, 2]) == zone
        np.testing.assert_allclose(out[0, zone], [3.0, -2.0, 50.0])


def test_totals_conserved_and_zone_centres_match_sim():
    cfg = TaxelGridConfig()
    rng = np.random.default_rng(0)
    tax = rng.normal(0, 1, (10, cfg.H, cfg.W, 3))
    tax[..., 2] = np.abs(tax[..., 2])
    out = taxels_to_zones(tax, cfg)
    sx, sy = cfg.signs
    np.testing.assert_allclose(out.sum(1), np.stack([tax[..., 0].sum((1, 2)) * sx, tax[..., 1].sum((1, 2)) * sy,
                                                      tax[..., 2].sum((1, 2))], -1), atol=1e-9)
    # 18 x 7 taxels at 1 cm tile the 0.18 x 0.07 sole: 9 x 3.5 -> every zone gets taxels
    assert set(np.unique(cfg.zone_of_taxel())) == set(range(8))
    from fla_terrain.sim.sole import SoleConfig
    np.testing.assert_allclose(cfg.zone_centers(), SoleConfig().zone_centers(), atol=1e-12)


def test_mask_and_drop():
    m = np.ones((18, 7), bool)
    m[:, 0] = False
    cfg = TaxelGridConfig(mask=m)
    tax = np.zeros((1, 18, 7, 3))
    tax[0, :, 0, 2] = 10.0
    assert taxels_to_zones(tax, cfg).sum() == 0
    big = TaxelGridConfig(H=22, W=9, outside="drop")
    tax = np.ones((1, 22, 9, 3))
    assert taxels_to_zones(tax, big)[0, :, 2].sum() == 18 * 7       # only taxels inside 0.18 x 0.07 kept


def test_resample_150_to_50_matches_sim_averaging():
    t = (np.arange(30) + 1) / 150.0
    x = np.arange(30, dtype=float)[:, None, None] * np.ones((1, 8, 3))
    t50, x50, valid = resample_to_rate(t, x, 50.0)
    np.testing.assert_allclose(t50, (np.arange(10) + 1) / 50.0)
    np.testing.assert_allclose(x50[:, 0, 0], x[:, 0, 0].reshape(10, 3).mean(1))
    assert valid.all()
    # a dropped burst of samples -> invalid frame, not interpolated
    keep = np.r_[0:6, 9:30]
    t50, x50, valid = resample_to_rate(t[keep], x[keep], 50.0, t0=0.0)
    assert valid.tolist() == [True, True, False] + [True] * 7
