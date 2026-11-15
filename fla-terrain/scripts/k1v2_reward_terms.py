# B14: compare per-episode reward terms (tensorboard summaries on adder): run4 end vs run5 epochs 50-70 vs last 30. Run on adder with env_isaaclab python.
import glob, numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
r4, r5 = sorted(glob.glob("/path/to/fla-terrain/k1v2/k1_gym/logs/*/summaries"))[-2:]
def load(r):
    ea = EventAccumulator(r, size_guidance={"scalars": 0}); ea.Reload()
    return {t: np.array([(e.step, e.value) for e in ea.Scalars(t)]) for t in ea.Tags()["scalars"]}
A, B = load(r4), load(r5)
def win(D, t, lo, hi):
    x = D[t]; m = (x[:, 0] >= lo) & (x[:, 0] < hi); return x[m, 1].mean() if m.any() else np.nan
n4 = int(A["reward"][:, 0].max()); n5 = int(B["reward"][:, 0].max())
print("run4 last step", n4, "run5 last step", n5)
rows = [(win(B, t, n5 - 30, n5 + 1) - win(B, t, 50, 70), t, win(A, t, n4 - 30, n4 + 1), win(B, t, 50, 70), win(B, t, n5 - 30, n5 + 1)) for t in sorted(A)]
print("%-32s %9s %11s %11s" % ("term", "run4 end", "run5 50-70", "run5 last30"))
for d, t, a, b0, b1 in sorted(rows)[:10] + [r for r in rows if r[1] in ("steps", "reward", "episode/survival", "episode/tracking_lin_vel_x")]:
    print("%-32s %9.3f %11.3f %11.3f" % (t, a, b0, b1))
