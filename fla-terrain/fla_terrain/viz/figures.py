"""Slide figures for FLA-0 sim results (results/v1).

Figures 1-5 read every number from metrics.json / lofo.json at run time, through the same
accessors that fla_terrain.models.report uses for tables.md (``col``, ``seeds_of``, ``_get``,
``MAIN``): the bar height is the mean of the per-seed values, as in ``report._ms``; the
confusion matrix is the per-seed matrices summed, as in ``report.render``.
Figure 6 takes the sole geometry from SoleConfig / k1.

Every ``fig*`` function returns (figure, values) where ``values`` is a list of
(key, value) pairs of exactly what was drawn, for ``--print-values``.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle  # noqa: E402

from ..models.report import MAIN, _get, col, seeds_of  # noqa: E402
from ..schema import SUPPORT_LABELS, TERRAIN_LABELS  # noqa: E402

# One identity per input set, used in every figure. Okabe-Ito hues ordered so that
# grayscale lightness falls with richer input (L* 71 / 61 / 46); markers and hatches
# are a second, colour-free channel.
STYLE = {
    "P": dict(color="#E69F00", marker="o", hatch=""),
    "P+N": dict(color="#CC79A7", marker="s", hatch="//"),
    "P+N+S": dict(color="#0072B2", marker="^", hatch=".."),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
W, H, DPI = 16, 9, 100  # 1600 x 900 px

plt.rcParams.update({
    "font.size": 18, "axes.titlesize": 22, "axes.labelsize": 20, "xtick.labelsize": 18,
    "ytick.labelsize": 18, "legend.fontsize": 18, "figure.titlesize": 26,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#898781",
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 1.0,
    "axes.axisbelow": True, "text.color": INK, "axes.labelcolor": INK,
    "hatch.linewidth": 1.2, "savefig.dpi": DPI, "figure.dpi": DPI,
})


def seed_vals(metrics, s, path):
    """Per-seed values (finite only), same filter as report._ms."""
    return np.array([x for x in col(metrics, s, path) if x is not None and np.isfinite(x)], float)


def _bars_with_seeds(ax, x, vals, s, width, values, key, label=True, value_fmt="{:.3f}", value_y=0.02, value_size=17):
    st = STYLE[s]
    m = vals.mean()
    ax.bar(x, m, width, color=st["color"], hatch=st["hatch"], edgecolor="white", linewidth=0,
           label=s if label else None, zorder=2)
    jit = np.linspace(-width * 0.22, width * 0.22, len(vals)) if len(vals) > 1 else [0]
    ax.scatter(x + np.asarray(jit), vals, s=70, marker=st["marker"], facecolor="white", edgecolor=INK,
               linewidth=1.6, zorder=3)
    values.append((key, m))
    if value_fmt:
        ax.text(x, value_y, value_fmt.format(m), ha="center", va="bottom", fontsize=value_size, color=INK,
                fontweight="bold", bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.2"), zorder=4)
    for i, v in enumerate(vals):
        values.append((f"{key} [seed {i}]", v))
    return m


def _seed_legend_handle():
    return Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK, markersize=10,
                  markeredgewidth=1.6, label="one seed")


def _set_handles():
    return [Patch(facecolor=STYLE[s]["color"], hatch=STYLE[s]["hatch"], edgecolor="white", label=s) for s in MAIN]


# ---------------------------------------------------------------- figure 1
def fig1_ablation(metrics):
    vals = []
    fig, ax = plt.subplots(figsize=(W, H))
    groups = [("terrain macro-F1", ("headline_walk", "terrain_f1")), ("support-state F1", ("headline_walk", "support_f1"))]
    w = 0.26
    for g, (name, path) in enumerate(groups):
        for k, s in enumerate(MAIN):
            x = g + (k - 1) * w
            _bars_with_seeds(ax, x, seed_vals(metrics, s, path), s, w * 0.92, vals, f"{name} | {s}", label=(g == 0))
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=21)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 (walk, test)")
    ax.set_title("Input-set ablation: terrain and support-state F1 (walk, test)\n"
                 "bar = mean of 3 seeds, markers = individual seeds", pad=16)
    ax.legend(handles=_set_handles() + [_seed_legend_handle()], loc="upper center", bbox_to_anchor=(0.5, -0.1),
              ncol=4, frameon=False)
    fig.tight_layout()
    return fig, vals


# ---------------------------------------------------------------- figure 2
def confusion_summed(metrics, s):
    """Summed over seeds, exactly as report.render builds the tables.md confusion matrices."""
    cms = [np.array(_get(r, ("headline_walk", "confusion"))) for _, r in seeds_of(metrics, s)]
    cms = [c for c in cms if c.ndim == 2]
    return sum(cms), len(cms)


def fig2_confusion(metrics, s="P+N"):
    vals = []
    cm, n = confusion_summed(metrics, s)
    names = [TERRAIN_LABELS[i] for i in range(cm.shape[0])]
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(W, H))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            v = norm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=18,
                    color="white" if v > 0.55 else INK)
            vals.append((f"{s} row-norm [{names[i]} -> {names[j]}]", v))
    ax.set_xticks(range(len(names)), names, rotation=20, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("fraction of the true-class windows")
    seeds = "summed over seeds " + ", ".join(k for k, _ in seeds_of(metrics, s)) if n > 1 else "single seed"
    ax.set_title(f"Terrain confusion matrix, {s} (walk, test)\n{seeds}; each row normalized to 1", pad=14)
    fig.tight_layout()
    return fig, vals


# ---------------------------------------------------------------- figure 3
def fig3_support_per_class(metrics):
    vals = []
    fig, ax = plt.subplots(figsize=(W, H))
    w = 0.26
    for c in range(3):
        for k, s in enumerate(MAIN):
            x = c + (k - 1) * w
            v = seed_vals(metrics, s, ("headline_walk", "support_f1_per_class", c))
            _bars_with_seeds(ax, x, v, s, w * 0.92, vals, f"support F1 {SUPPORT_LABELS[c]} | {s}", label=(c == 0))
    ax.set_xticks(range(3), [SUPPORT_LABELS[c] for c in range(3)], fontsize=21)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 per class (mean of L and R feet)")
    ax.set_title("Support-state F1 per class (walk, test)\nbar = mean of 3 seeds, markers = individual seeds", pad=16)
    ax.legend(handles=_set_handles() + [_seed_legend_handle()], loc="upper center", bbox_to_anchor=(0.5, -0.1),
              ncol=4, frameon=False)
    fig.tight_layout()
    return fig, vals


# ---------------------------------------------------------------- figure 4
DELAY_PANELS = (
    ("Strip surface change:\nmedian latency (s)", ("strip", "median_latency_s"), "s"),
    ("Strip surface change:\nmedian latency (steps)", ("strip", "median_latency_steps"), "steps"),
    ("Edge (walk):\nmedian detection delay (s)", ("edge_delay", "walk", "median_delay_s"), "s"),
    ("Edge (walk):\nmiss rate", ("edge_delay", "walk", "miss_rate"), "fraction of events"),
    ("Lifted:\nmedian detection delay (s)", ("lifted_delay", "median_delay_s"), "s"),
)


def fig4_delays(metrics):
    vals = []
    fig, axes = plt.subplots(2, 3, figsize=(W, H + 1))
    for ax, (title, path, unit) in zip(axes.flat, DELAY_PANELS):
        top = max([1e-9] + [seed_vals(metrics, s, path).max() for s in MAIN])
        for k, s in enumerate(MAIN):
            v = seed_vals(metrics, s, path)
            _bars_with_seeds(ax, k, v, s, 0.7, vals, f"{title.replace(chr(10), ' ')} | {s}",
                             value_fmt="{:.2f}", value_y=top * 0.03, value_size=15)
        ax.set_ylim(0, top * 1.15)
        ax.set_xticks(range(3), MAIN, fontsize=16)
        ax.set_title(title, fontsize=18)
        ax.set_ylabel(unit, fontsize=16)
        ax.tick_params(axis="y", labelsize=15)
    # last panel: legend + lifted miss rate (computed, not typed)
    lx = axes.flat[-1]
    lx.axis("off")
    lm = {s: seed_vals(metrics, s, ("lifted_delay", "miss_rate")) for s in MAIN}
    for s in MAIN:
        for i, v in enumerate(lm[s]):
            vals.append((f"Lifted miss rate | {s} [seed {i}]", v))
    lx.legend(handles=_set_handles() + [_seed_legend_handle()], loc="upper left", frameon=False, fontsize=17)
    allz = all(np.all(v == 0) for v in lm.values())
    note = ("Lifted miss rate: 0 for every seed\nand input set" if allz else
            "Lifted miss rate (mean):\n" + "\n".join(f"{s}: {lm[s].mean():.2f}" for s in MAIN))
    lx.text(0.0, 0.25, note + "\n\nbar = mean of 3 seeds\n(test split)", fontsize=15, color=MUTED,
            transform=lx.transAxes, va="top")
    fig.suptitle("Detection delays by input set (test split; lower is better)", y=0.99)
    fig.tight_layout()
    return fig, vals


# ---------------------------------------------------------------- figure 5
def lofo_vals(lofo, held, s, which):
    """Per-seed values from lofo.json, same extraction as report._exploratory (R5)."""
    r = list(lofo["runs"][held][s].values())
    v = [x.get(which, {}).get("terrain_f1", np.nan) for x in r]
    return np.array([x for x in v if x is not None and np.isfinite(x)], float)


def fig5_gait_transfer(lofo):
    vals = []
    fig, ax = plt.subplots(figsize=(W, H))
    helds = list(lofo["runs"].keys())
    w = 0.27
    has_seen = False
    for g, held in enumerate(helds):
        for k, s in enumerate([x for x in MAIN if x in lofo["runs"][held]]):
            x = g + (k - 1) * w
            st = STYLE[s]
            seen = lofo_vals(lofo, held, s, "seen")
            if len(seen):
                has_seen = True
                ms = seen.mean()
                ax.bar(x, ms, w * 0.92, facecolor="none", edgecolor=st["color"], linewidth=2.5, zorder=1)
                ax.text(x, ms + 0.015, f"{ms:.2f}", ha="center", va="bottom", fontsize=14, color=MUTED)
                vals.append((f"LOFO seen (in-distribution) | {held} | {s}", ms))
                for i, v in enumerate(seen):
                    vals.append((f"LOFO seen | {held} | {s} [seed {i}]", v))
            un = lofo_vals(lofo, held, s, "unseen")
            m = _bars_with_seeds(ax, x, un, s, w * 0.62, vals, f"LOFO unseen (held-out) | {held} | {s}", value_fmt=None)
            ax.text(x, max(m, un.max()) + 0.015, f"{m:.2f}", ha="center", va="bottom", fontsize=16, fontweight="bold")
    ax.set_xticks(range(len(helds)), [f"held out: {h}" for h in helds], fontsize=19)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("terrain macro-F1 on the held-out gait")
    ax.set_title("EXPLORATORY (post hoc, after the G1 verdict): transfer to an unseen gait\n"
                 "leave-one-gait-out, walk test windows of the held-out walker", pad=14)
    h = _set_handles() + [_seed_legend_handle()]
    if has_seen:
        h.append(Patch(facecolor="none", edgecolor=MUTED, linewidth=2.5, label="outline: gait seen in training"))
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5, frameon=False, fontsize=16)
    fig.tight_layout()
    return fig, vals


# ---------------------------------------------------------------- figure 6
def fig6_sole_diagram():
    from ..sim import k1
    from ..sim.sole import SoleConfig
    cfg = SoleConfig()
    hx, hy, hz = (1000 * v for v in cfg.box_half)  # mm
    ox, oy, oz = (1000 * v for v in cfg.sole_origin_body())
    na, nc = cfg.n_along, cfg.n_across
    dx, dy = 2 * hx / na, 2 * hy / nc
    centers = 1000 * cfg.zone_centers()
    vals = [("sole length mm", 2 * hx), ("sole width mm", 2 * hy), ("thickness mm", 2 * hz),
            ("zone x mm", dx), ("zone y mm", dy), ("sole origin in ankle_roll_link (x,y,z) mm", (ox, oy, oz))]

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.0, 0.0, 0.56, 0.90])
    tx = fig.add_axes([0.56, 0.0, 0.44, 0.90])
    tx.axis("off")
    ax.set_aspect("equal")
    ax.axis("off")
    # Page: x (toe) up, y (left) to the page left -> robot seen from above, facing up.
    # The gap between the feet is schematic (not from the model).
    sep = 2 * hy + 150.0
    feet = (("L", "LEFT foot", -sep / 2), ("R", "RIGHT foot", sep / 2))
    blue, orange = "#0072B2", "#D55E00"

    def P(x, y, off):  # sole-frame (x, y) [mm] -> page
        return off - y, x

    for side, label, off in feet:
        ax.add_patch(Rectangle((off - hy, -hx), 2 * hy, 2 * hx, facecolor="#f0efec", edgecolor=INK, linewidth=2.5))
        for z, (cx, cy) in enumerate(centers):
            px, py = P(cx, cy, off)
            ax.add_patch(Rectangle((px - dy / 2, py - dx / 2), dy, dx, facecolor="none", edgecolor=MUTED, linewidth=1.5))
            ax.text(px, py + dx * 0.18, f"{z}", ha="center", va="center", fontsize=26, fontweight="bold")
        ax.text(off, hx + 12, "toe", ha="center", va="bottom", fontsize=16, color=MUTED)
        ax.text(off, -hx - 10, "heel", ha="center", va="top", fontsize=16, color=MUTED)
        ax.text(off, hx + 38, label, ha="center", va="bottom", fontsize=20, fontweight="bold")
        # sole origin (centre of the sole bottom) and ankle_roll_link origin projection
        o = P(0, 0, off)
        ax.plot(*o, marker="o", markersize=16, markerfacecolor="white", markeredgecolor=blue, mew=2.5, zorder=6)
        ax.plot(*o, marker=".", markersize=9, color=blue, zorder=7)
        a = P(-ox, -oy, off)
        ax.plot(*a, marker="+", color=orange, markersize=20, mew=3, zorder=6)
        # sole-frame axes, drawn beside the foot (same orientation as at the origin)
        t0 = (off + hy + 75, -hx + 10)
        L = 50
        for (ux, uy), name, ha, va, d in (((0, L), "x (toe)", "center", "bottom", (0, 6)),
                                          ((-L, 0), "y (left)", "center", "top", (0, -8))):
            ax.add_patch(FancyArrowPatch(t0, (t0[0] + ux, t0[1] + uy), arrowstyle="-|>", mutation_scale=24,
                                         color=blue, linewidth=2.8, zorder=5))
            ax.text(t0[0] + ux + d[0], t0[1] + uy + d[1], name, ha=ha, va=va, fontsize=16, color=blue)
        ax.plot(*t0, marker="o", markersize=16, markerfacecolor="white", markeredgecolor=blue, mew=2.5, zorder=6)
        ax.plot(*t0, marker=".", markersize=9, color=blue, zorder=7)
        ax.text(t0[0] + 12, t0[1] - 6, "z (up)", ha="left", va="top", fontsize=16, color=blue)

    # dimensions: length and width on the left foot, zone size on the right foot
    offL, offR = feet[0][2], feet[1][2]
    xl = offL - hy - 18
    ax.annotate("", (xl, -hx), (xl, hx), arrowprops=dict(arrowstyle="<->", lw=2, color=INK))
    ax.text(xl - 5, 0, f"{2 * hx:.0f} mm", rotation=90, ha="right", va="center", fontsize=18)
    yb = -hx - 38
    ax.annotate("", (offL - hy, yb), (offL + hy, yb), arrowprops=dict(arrowstyle="<->", lw=2, color=INK))
    ax.text(offL, yb - 5, f"{2 * hy:.0f} mm", ha="center", va="top", fontsize=18)
    ax.annotate("", (offR - hy, yb), (offR - hy + dy, yb), arrowprops=dict(arrowstyle="<->", lw=2, color=MUTED))
    ax.text(offR - hy + dy / 2, yb - 5, f"one zone: {dy:.0f} mm across (y)\nx {dx:.0f} mm along (x)", ha="center", va="top",
            fontsize=16, color=MUTED)
    ax.set_xlim(offL - hy - 55, offR + hy + 135)
    ax.set_ylim(-hx - 95, hx + 70)

    foot = k1.FOOT_BODIES["L"].replace("left_", "")
    rows = [
        ("View", "from above, toes up; robot faces up the page"),
        ("Sole frame", "foot body axes: x toe, y left,\nz up out of the ground (towards you)"),
        ("Origin  ⊙", "centre of the sole bottom"),
        ("+", f"{foot} origin; the sole origin is at\nx={ox:.0f}, y={oy:.0f}, z={oz:.0f} mm in that frame"),
        ("Size", f"{2 * hx:.0f} x {2 * hy:.0f} mm, {2 * hz:.0f} mm thick; {nc} x {na} zones"),
        ("Zone index", f"z = i_along x {nc} + i_across\ni_along = 0 at the heel, i_across = 0 at -y"),
        ("Both feet", "same rule, not mirrored: even zones are\nmedial on the left foot, lateral on the right"),
        ("Force", "on the sole FROM the ground:\nFz >= 0 is load; Fx, Fy are shear"),
    ]
    y = 0.95
    for k, v in rows:
        tx.text(0.0, y, k, fontsize=16, fontweight="bold", va="top", color=orange if k == "+" else
                (blue if k.startswith("Origin") else INK))
        tx.text(0.27, y, v, fontsize=16, va="top", linespacing=1.3)
        y -= 0.048 * (1 + v.count("\n")) + 0.022
    tx.text(0.0, 0.02, "Gap between the feet is not to scale.", fontsize=14, color=MUTED)
    fig.suptitle(f"Virtual sole: {nc} x {na} zones per foot (Booster K1)", y=0.97)
    return fig, vals


FIGURES = (
    ("fig1_ablation.png", "metrics", fig1_ablation),
    ("fig2_confusion_PN.png", "metrics", fig2_confusion),
    ("fig3_support_per_class.png", "metrics", fig3_support_per_class),
    ("fig4_delays.png", "metrics", fig4_delays),
    ("fig5_gait_transfer.png", "lofo", fig5_gait_transfer),
    ("fig6_sole_diagram.png", None, fig6_sole_diagram),
)
