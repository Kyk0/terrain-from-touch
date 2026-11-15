import numpy as np
import pytest

from walking.scenarios import KINDS, make_schedule


@pytest.mark.parametrize("kind", KINDS)
def test_schedule_ranges_and_determinism(kind):
    a = make_schedule(kind, np.random.default_rng(3))
    b = make_schedule(kind, np.random.default_rng(3))
    ts = np.arange(0, 20, 0.02)
    ca, cb = np.array([a(t) for t in ts]), np.array([b(t) for t in ts])
    assert np.array_equal(ca, cb)
    assert np.all(ca[ts < 1.0] == 0)                 # stand first
    assert np.all(ca[:, 1] == 0)                     # no sideways commands
    assert ca[:, 0].max() <= 0.5 + 1e-9 and ca[:, 0].min() >= 0
    assert np.abs(ca[:, 2]).max() <= 0.5 + 1e-9
    assert ca[ts > 2.0, 0].max() >= 0.2              # actually walks


def test_stop_go_stands_two_seconds():
    s = make_schedule("stop_go", np.random.default_rng(0))
    ts = np.arange(1.0, 20, 0.02)
    still = np.array([np.abs(s(t)).max() < 0.05 for t in ts])
    runs, n = [], 0
    for x in still:
        n = n + 1 if x else 0
        runs.append(n)
    assert max(runs) * 0.02 >= 1.5                   # >= 1.5 s of standing (2 s minus ramp)


def test_turn_has_yaw_and_straight_has_none():
    ts = np.arange(0, 20, 0.1)
    assert max(abs(make_schedule("turn", np.random.default_rng(1))(t)[2]) for t in ts) >= 0.2
    assert all(make_schedule("straight", np.random.default_rng(1))(t)[2] == 0 for t in ts)
