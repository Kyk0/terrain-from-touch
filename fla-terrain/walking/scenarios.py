"""Command schedules for walking data generation.

    sched = make_schedule("speed_change", rng, duration=20.0)
    cmd_vel = sched(t)            # (vx m/s, vy m/s, wz rad/s), body frame; t = time since episode start

All random parameters are drawn once from `rng` at construction, so a schedule is a deterministic
function of t (reproducible from the episode seed). `sched.params` holds them for metadata.
Every schedule starts with `stand_s` (default 1 s) of zero command so the robot settles first.

Kinds: steady, speed_change, turn, stop_go, straight (alias heading_hold).
Sideways (vy) commands are not generated: both policies under-track them (B1: vy 0.09-0.15 for 0.2).
"""

from __future__ import annotations

import numpy as np

V_RANGE = (0.2, 0.5)        # m/s forward, tracked within ~5 % by G113 and G80 on flat ground
WZ_RANGE = (0.2, 0.5)       # rad/s yaw magnitude


class Schedule:
    def __init__(self, kind, knots, params, stand_s, ramp_s=0.5):
        self.kind = kind
        self.params = dict(params, kind=kind, stand_s=stand_s)
        # knots: list of (t_start, (vx, vy, wz)); command ramps linearly over ramp_s after each knot
        self.t = np.array([k[0] for k in knots], float)
        self.c = np.array([k[1] for k in knots], float)
        self.ramp = ramp_s
        self.stand_s = stand_s

    def __call__(self, t, rng=None):
        if t < self.stand_s:
            return np.zeros(3)
        i = int(np.searchsorted(self.t, t, side="right")) - 1
        if i < 0:
            return np.zeros(3)
        prev = self.c[i - 1] if i > 0 else np.zeros(3)
        a = min(1.0, (t - self.t[i]) / self.ramp) if self.ramp > 0 else 1.0
        return prev + a * (self.c[i] - prev)  # policy stands (clock off) once |cmd| < 0.05


def make_schedule(kind, rng, duration=20.0, stand_s=1.0):
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    t0 = stand_s
    if kind == "steady":
        v = rng.uniform(*V_RANGE)
        return Schedule(kind, [(t0, (v, 0, 0))], {"vx": v}, stand_s)
    if kind in ("straight", "heading_hold"):
        # zero yaw: the policy holds the initial heading (heading-error observation), so the path is
        # a straight line along the spawn yaw. Place the feature (strip, edge) across that line.
        v = rng.uniform(0.25, 0.45)
        return Schedule("straight", [(t0, (v, 0, 0))], {"vx": v}, stand_s)
    if kind == "speed_change":
        knots, t = [], t0
        while t < duration:
            knots.append((t, (rng.uniform(*V_RANGE), 0, 0)))
            t += rng.uniform(3.0, 6.0)
        return Schedule(kind, knots, {"vx": [k[1][0] for k in knots], "t": [k[0] for k in knots]}, stand_s)
    if kind == "turn":
        v = rng.uniform(0.25, 0.4)
        knots, t, turning = [], t0, False
        while t < duration:
            wz = rng.choice([-1, 1]) * rng.uniform(*WZ_RANGE) if turning else 0.0
            knots.append((t, (v, 0, wz)))
            t += rng.uniform(2.0, 4.0)
            turning = not turning
        return Schedule(kind, knots, {"vx": v, "wz": [k[1][2] for k in knots]}, stand_s)
    if kind == "stop_go":
        knots, t = [], t0
        while t < duration:
            knots.append((t, (rng.uniform(*V_RANGE), 0, 0)))
            t += rng.uniform(3.0, 5.0)
            knots.append((t, (0, 0, 0)))
            t += 2.0
        return Schedule(kind, knots, {"vx": [k[1][0] for k in knots if k[1][0]]}, stand_s)
    raise ValueError(f"unknown schedule kind {kind!r}")


KINDS = ("steady", "speed_change", "turn", "stop_go", "straight")
