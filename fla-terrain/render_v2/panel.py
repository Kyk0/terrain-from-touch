"""Sole panel v2 and timeline strips (matplotlib Agg -> RGB arrays).

Sole panel: for each foot the recorded point map (19 x 8 points at 10 mm, sole frame, x toe up, y left drawn to the
left, i.e. seen from above) coloured by the normal force of each point, nearest-cell display (no interpolation),
FIXED scale 0..FZ_MAX N per point (above: top colour, marked in the legend). The foot outline traced from the PaXini
screenshot (fla_terrain.sim.pointmap.foot_halfwidth_fraction) is drawn on top. The four block shear vectors per
foot (S_blk: front-left, front-right, back-left, back-right; force on the sole from the ground) are drawn at the
block centroids with a FIXED scale: SHEAR_REF newtons = 30 mm arrow.
"""
from __future__ import annotations

import numpy as np

from fla_terrain.sim.pointmap import BLOCK4, PointLayout, foot_halfwidth_fraction, views

FZ_MAX = 15.0       # N per point, top of the colour scale
SHEAR_REF = 50.0    # N of block shear drawn as a 30 mm arrow
ARROW_MM = 30.0
BG, FG, MUTED = "#15171c", "#e8e8e8", "#8a8f98"
CLASS_NAMES = ["hard-grip", "hard-slip", "soft", "rough", "slope", "edge"]
# same colours as the terrain in the 3D view (render_v2.core.CLASS_RGBA)
CLASS_HEX = ["#807f7d", "#6ba8e0", "#b39154", "#857a6b", "#80b37a", "#a87038"]
SUP_HEX = ["#6f7f95", "#f0a030", "#3fbf6f"]


class SolePanel:
    def __init__(self, layout: PointLayout, w=640, h=720, title="", sat=None):
        compact = w < 560
        fs = 0.8 if compact else 1.0
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.L = layout
        self.table = layout.table()
        self.w, self.h = w, h
        self.fig = Figure(figsize=(w / 100, h / 100), dpi=100, facecolor=BG)
        self.canvas = FigureCanvasAgg(self.fig)
        rows = int(layout.rowcol[:, 0].max()) + 1
        cols = int(layout.rowcol[:, 1].max()) + 1
        self.shape = (rows, cols)
        p = layout.pitch * 1000
        Lmm, Wmm = layout.length * 1000, layout.width * 1000
        # cells centred on the points (rect mask: outer points on the box edges)
        self.extent = (-Wmm / 2 - p / 2, Wmm / 2 + p / 2, -Lmm / 2 - p / 2, Lmm / 2 + p / 2)
        self.axes, self.ims, self.arrows, self.texts, self.satmk = [], [], [], [], []
        self.sat = sat              # per-point normal-force limit of this sensor [N]; points at it are marked
        s = np.linspace(0, 1, 200)
        hw = foot_halfwidth_fraction(s) * Wmm / 2
        xo = s * Lmm - Lmm / 2
        outline = (np.r_[hw, -hw[::-1], hw[:1]], np.r_[xo, xo[::-1], xo[:1]])
        # block centroids in (h = -y, x) plot coordinates [mm]
        W4 = np.asarray(self.table["block_weights"])[:, [self.table["blocks"].index(b) for b in BLOCK4]]
        xy = layout.xy * 1000
        self.cent = [((-(W4[:, b] * xy[:, 1]).sum() / W4[:, b].sum()), (W4[:, b] * xy[:, 0]).sum() / W4[:, b].sum())
                     for b in range(4)]
        for k, side in enumerate(("left foot", "right foot")):
            ax = self.fig.add_axes([0.03 + 0.39 * k, 0.15, 0.35, 0.66] if compact else [0.04 + 0.42 * k, 0.15, 0.38, 0.66])
            ax.set_facecolor(BG)
            im = ax.imshow(np.zeros(self.shape), origin="lower", extent=self.extent, cmap="inferno", vmin=0,
                           vmax=FZ_MAX, interpolation="nearest", aspect="equal")
            ax.plot(*outline, color="#4fd1ff", lw=1.2)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(side, color=FG, fontsize=11)
            arr = [ax.annotate("", xy=c, xytext=c, arrowprops=dict(arrowstyle="-|>", color="#00ff9c", lw=2.0,
                                                                   mutation_scale=12)) for c in self.cent]
            ax.set_xlim(self.extent[0] - 20, self.extent[1] + 20)
            ax.set_ylim(self.extent[2] - 10, self.extent[3] + 10)
            ax.text(0, self.extent[3] + 4, "toe", color=MUTED, fontsize=8, ha="center", va="bottom")
            ax.text(0, self.extent[2] - 4, "heel", color=MUTED, fontsize=8, ha="center", va="top")
            self.texts.append(ax.text(0, self.extent[2] - 14, "", color=FG, fontsize=9 * fs, ha="center", va="top"))
            self.satmk.append(ax.scatter([], [], marker="x", c="#00e5ff", s=40, linewidths=2.0, zorder=5))
            self.axes.append(ax)
            self.ims.append(im)
            self.arrows.append(arr)
        cax = self.fig.add_axes([0.80 if compact else 0.89, 0.30, 0.025, 0.5])
        cb = self.fig.colorbar(self.ims[0], cax=cax)
        cb.set_label(f"Fz per point [N] (>= {FZ_MAX:.0f}: top)" if compact else
                     f"normal force per point [N] (>= {FZ_MAX:.0f} N: top colour)", color=FG, fontsize=8 * fs)
        cb.ax.tick_params(colors=FG, labelsize=8)
        lg = self.fig.add_axes([0.04, 0.03, 0.9, 0.06])
        lg.axis("off")
        lg.set_xlim(0, 100)
        lg.set_ylim(0, 10)
        lg.annotate("", xy=(14, 5), xytext=(2, 5), arrowprops=dict(arrowstyle="-|>", color="#00ff9c", lw=2.0))
        lg.text(15, 5, (f"block shear: {ARROW_MM:.0f} mm = {SHEAR_REF:.0f} N" if compact else
                        f"block shear (force on sole from ground): arrow {ARROW_MM:.0f} mm = {SHEAR_REF:.0f} N; "
                        "blocks front-L/R, back-L/R"), color=FG, fontsize=8, va="center")
        self.compact = compact
        if sat is not None:
            lg.scatter([2], [0.5], marker="x", c="#00e5ff", s=30, linewidths=2.0, clip_on=False)
            lg.text(5, 0.5, f"x = point at the sensor limit ({sat:.0f} N): value clipped there, true load higher",
                    color=FG, fontsize=8 if not compact else 7, va="center")
            lg.set_ylim(-2, 10)
        self.title = self.fig.text(0.5, 0.975, title, color=FG, fontsize=12, ha="center", va="top")
        self.sub = self.fig.text(0.5, 0.935, "", color=MUTED, fontsize=9, ha="center", va="top")

    def grid(self, fz):
        A = np.zeros(self.shape)
        A[self.L.rowcol[:, 0], self.L.rowcol[:, 1]] = fz
        return A[:, ::-1]           # column 0 is -y (right side when seen from above): flip so +y is on the left

    def draw(self, F2, subtitle=""):
        """F2: [2, P, 3] recorded point forces (L, R) in the sole frame (raw sensor output)."""
        v = views(F2, self.table)
        for k in range(2):
            self.ims[k].set_data(self.grid(F2[k, :, 2]))
            for b in range(4):
                sx, sy = v["S_blk"][k, b]
                c = self.cent[b]
                tip = (c[0] - sy / SHEAR_REF * ARROW_MM, c[1] + sx / SHEAR_REF * ARROW_MM)
                self.arrows[k][b].xy = tip
                self.arrows[k][b].set_visible(np.hypot(sx, sy) > 0.5)
            if self.sat is not None:
                at = F2[k, :, 2] >= 0.995 * self.sat
                xy = self.L.xy[at] * 1000
                self.satmk[k].set_offsets(np.column_stack([-xy[:, 1], xy[:, 0]]) if at.any() else np.zeros((0, 2)))
            fz = F2[k, :, 2].sum()
            s = F2[k, :, :2].sum(0)
            self.texts[k].set_text(f"Fz {fz:.0f} N\n|S| {np.hypot(*s):.0f} N" if self.compact else
                                   f"Fz {fz:6.1f} N   |shear| {np.hypot(*s):5.1f} N")
        self.sub.set_text(subtitle)
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[:, :, :3].copy()


class StripPanel:
    """Horizontal timeline strips that fill in up to time t: rows of class (or phase) colours per window/frame."""

    def __init__(self, w, h, rows, t_max, title="", legend=None):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.fig = Figure(figsize=(w / 100, h / 100), dpi=100, facecolor=BG)
        self.canvas = FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_axes([0.13, 0.40 if legend else 0.14, 0.85, 0.42 if legend else 0.70])
        self.ax.set_facecolor("#22252c")
        self.ax.set_xlim(0, t_max)
        self.ax.set_ylim(-0.5, len(rows) - 0.5)
        self.ax.set_yticks(range(len(rows)))
        self.ax.set_yticklabels(rows[::-1], color=FG, fontsize=9)
        self.ax.tick_params(colors=MUTED, labelsize=8)
        if not legend:
            self.ax.set_xlabel("time [s]", color=MUTED, fontsize=8)
        for sp in self.ax.spines.values():
            sp.set_color("#444")
        self.nrows = len(rows)
        self.cursor = self.ax.axvline(0, color="#ffffff", lw=1.0)
        from matplotlib.patches import Rectangle
        self.t_max = t_max
        self.mask = Rectangle((0, -0.5), t_max, len(rows), color="#22252c", zorder=5)   # hides the future
        self.ax.add_patch(self.mask)
        if title:
            self.fig.text(0.13, 0.985, title + "   (time axis in s)", color=FG, fontsize=9, va="top")
        if legend:
            x = 0.13
            for name, col in legend:
                self.fig.patches.append(__import__("matplotlib").patches.Rectangle(
                    (x, 0.04), 0.018, 0.09, transform=self.fig.transFigure, color=col))
                self.fig.text(x + 0.022, 0.085, name, color=FG, fontsize=8, va="center")
                x += 0.022 + 0.0075 * len(name) + 0.02
        self.drawn = {}

    def add_segments(self, row, t0, t1, colors):
        """row index from the top; t0/t1/colors arrays of equal length (segments drawn once)."""
        y = self.nrows - 1 - row
        from matplotlib.collections import PolyCollection
        verts = [[(a, y - 0.4), (b, y - 0.4), (b, y + 0.4), (a, y + 0.4)] for a, b in zip(t0, t1)]
        self.ax.add_collection(PolyCollection(verts, facecolors=colors, edgecolors="none"))

    def render(self, t):
        self.cursor.set_xdata([t, t])
        self.mask.set_x(t)
        self.mask.set_width(max(self.t_max - t, 0.0))
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[:, :, :3].copy()
