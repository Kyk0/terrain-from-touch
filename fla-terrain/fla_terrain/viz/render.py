"""Render one bundle (fla_terrain/viz/prepare.py) to an MP4: robot view + data panel.

Needs MuJoCo with an offscreen GL backend (MUJOCO_GL=egl on a GPU machine, or osmesa). Nothing is simulated here:
the robot pose of every video frame is the logged qpos of the dataset episode, and every number in the panel is read
from the bundle (sensor output of that episode, stored test predictions of the P+N seed-0 model for that episode).

Frame layout (1280 x 720): robot view 800 x 720 on the left, panel 480 x 720 on the right.
Panel timing: the sole heat map shows the latest finished 50 Hz sensor frame; the model block shows the latest
finished window (1 s window, a new output every 0.2 s, exactly the windows the report is scored on) next to the
ground truth at that window's end, and is marked WRONG when the two differ.
Scales are fixed for all clips: FZ_MAX newtons per zone for colour, SHEAR_REF newtons per zone = one zone length.
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

VIEW_W, VIEW_H, PANEL_W = 800, 720, 480
TILE_W, TILE_H = 400, 360
FZ_MAX = 150.0     # N per zone, top of the colour scale (values above are drawn at the top colour)
SHEAR_REF = 40.0   # N per zone drawn as an arrow one zone length (45 mm) long
SHEAR_CLIP = 100.0  # arrows longer than this are clipped [N]
WEIGHT_N = 19.666 * 9.81

CLASS_NAMES = ["hard-grip", "hard-slip", "soft", "rough", "slope", "edge"]
SUPPORT_NAMES = ["airborne", "partial", "full"]
# class -> two checker colours of the ground in the 3D view (also the swatch colour in the panel)
GROUND = {
    0: ((0.50, 0.50, 0.49), (0.42, 0.42, 0.41)),   # hard-grip: concrete grey
    1: ((0.42, 0.66, 0.88), (0.33, 0.56, 0.80)),   # hard-slip: ice blue
    2: ((0.70, 0.57, 0.33), (0.61, 0.49, 0.27)),   # soft: sand
    3: ((0.52, 0.48, 0.42), (0.45, 0.41, 0.36)),   # rough: gravel brown (bumps are drawn dark)
    4: ((0.50, 0.70, 0.48), (0.42, 0.61, 0.41)),   # slope: green
    5: ((0.66, 0.44, 0.22), (0.56, 0.36, 0.17)),   # edge: wooden planks (the pit below is black)
}
BUMP_RGBA = (0.16, 0.13, 0.11, 1.0)
PIT_RGBA = (0.03, 0.03, 0.04, 1.0)
BG = "#15171c"
FG = "#e8e8e8"
MUTED = "#8a8f98"
GOOD, BAD, NEUTRAL = "#3fbf6f", "#f0483e", "#6f7f95"
SUPPORT_COL = ["#6f7f95", "#f0a030", "#3fbf6f"]


def _hex(rgb):
    return "#%02x%02x%02x" % tuple(int(round(255 * c)) for c in rgb)


CLASS_HEX = [_hex(GROUND[c][0]) for c in range(6)]


# ------------------------------------------------------------------------------------------------ scene
def build_scene(meta: dict, xml: str):
    """Model with the same bodies/joints as the dataset model (so the logged qpos applies) and readable terrain."""
    import mujoco

    from ..sim.build import add_base_rig, add_terrain, terrain_regions
    from ..sim.sole import SoleConfig, add_sole

    cfg = meta["terrain_cfg"]
    spec = mujoco.MjSpec.from_file(xml)
    for s in "LR":
        add_sole(spec, s, SoleConfig())
    names = add_terrain(spec, cfg)
    if meta.get("base_rig"):
        add_base_rig(spec)
    regions, geom_region = terrain_regions(cfg)
    for c, (a, b) in GROUND.items():
        t = spec.add_texture()
        t.name = f"viz_tex_{c}"
        t.type = mujoco.mjtTexture.mjTEXTURE_2D
        t.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
        t.rgb1, t.rgb2 = list(a), list(b)
        t.width = t.height = 256
        mat = spec.add_material()
        mat.name = f"viz_mat_{c}"
        tex = list(mat.textures)
        tex[mujoco.mjtTextureRole.mjTEXROLE_RGB] = t.name
        mat.textures = tex
        mat.texrepeat = [2.0, 2.0]   # one checker cell = 0.25 m
        mat.texuniform = True
        mat.reflectance = 0.12 if c == 1 else 0.0
        mat.specular = 0.3 if c == 1 else 0.05
        mat.shininess = 0.6 if c == 1 else 0.1
    geoms = cfg.get("geoms")
    for i, name in enumerate(names):
        g = spec.geom(name)
        gd = geoms[i] if geoms else {"type": "plane"}
        cls = int(regions[geom_region[i]]["label"])
        pos = gd.get("pos") or [0, 0, 0]
        if gd["type"] == "plane" and pos[2] < -0.01:
            g.rgba = list(PIT_RGBA)
        elif gd["type"] == "box" and max(gd["size"][0], gd["size"][1]) <= 0.06:
            g.rgba = list(BUMP_RGBA)
        else:
            g.material = f"viz_mat_{cls}"
            g.rgba = [1, 1, 1, 1]
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, VIEW_W)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, VIEW_H)
    spec.visual.quality.shadowsize = 4096
    spec.visual.map.znear = 0.02
    spec.visual.headlight.ambient = [0.30, 0.30, 0.30]
    spec.visual.headlight.diffuse = [0.15, 0.15, 0.15]
    spec.visual.headlight.specular = [0.05, 0.05, 0.05]
    m = spec.compile()
    return m


def _smooth(x, sigma_frames):
    r = int(3 * sigma_frames)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_frames) ** 2)
    k /= k.sum()
    xp = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)])
    return np.stack([np.convolve(xp[:, i], k, mode="valid") for i in range(x.shape[1])], 1)


def camera_path(qpos, fps):
    """Tracking camera, front-left of the robot: smoothed base position and heading. -> lookat [F,3], azimuth [F]."""
    w, x, y, z = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
    yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    p = _smooth(qpos[:, :3], 0.35 * fps)
    yaw_s = _smooth(yaw[:, None], 1.2 * fps)[:, 0]
    look = p.copy()
    look[:, 2] -= 0.12
    return look, np.rad2deg(yaw_s) + 180.0 + 52.0


# ------------------------------------------------------------------------------------------------ panel
class Panel:
    def __init__(self, meta: dict):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import Rectangle

        self.meta = meta
        self.cmap = matplotlib.colormaps["inferno"]
        fig = Figure(figsize=(PANEL_W / 100, VIEW_H / 100), dpi=100, facecolor=BG)
        self.canvas = FigureCanvasAgg(fig)
        self.fig = fig
        tx = dict(color=FG, family="DejaVu Sans")
        fig.text(0.04, 0.982, meta["panel_title"], va="top", fontsize=12.5, weight="bold", **tx)
        self.t_sub = fig.text(0.04, 0.949, "", va="top", fontsize=9.5, color=MUTED)
        self.t_time = fig.text(0.96, 0.982, "", va="top", ha="right", fontsize=12.5, family="DejaVu Sans Mono", color=FG)
        fig.text(0.04, 0.905, "SOLE SENSOR  (50 Hz, top view, toes up)", va="top", fontsize=9.5, weight="bold", color=MUTED)

        # ---- soles
        lay = meta["zone_layout"]
        cen = np.array(lay["zone_centers_xy_m"])
        L, Wd = lay["sole_length_m"], lay["sole_width_m"]
        n_along = len(np.unique(np.round(cen[:, 0], 6)))
        n_across = len(cen) // n_along
        zx, zy = L / n_along, Wd / n_across
        self.zx = zx
        ax = fig.add_axes([0.03, 0.535, 0.74, 0.345])
        ax.set_facecolor(BG)
        ax.set_xlim(-0.14, 0.14)
        ax.set_ylim(-0.118, 0.118)
        ax.set_aspect("equal")
        ax.axis("off")
        self.rects, X, Y = [], [], []
        self.foot_x = {"L": -0.062, "R": 0.062}
        for s in "LR":
            x0 = self.foot_x[s]
            for (cx, cy) in cen:  # plot X = -y (left is left), plot Y = x (toe up)
                r = Rectangle((x0 - cy - zy / 2, cx - zx / 2), zy, zx, facecolor="k", edgecolor="#5a606b", linewidth=1.0)
                ax.add_patch(r)
                self.rects.append(r)
                X.append(x0 - cy)
                Y.append(cx)
            ax.text(x0, L / 2 + 0.006, {"L": "LEFT", "R": "RIGHT"}[s], ha="center", va="bottom", fontsize=10, weight="bold", color=FG)
        self.quiv = ax.quiver(X, Y, np.zeros(len(X)), np.zeros(len(X)), angles="xy", scale_units="xy", scale=1.0,
                              width=0.012, headwidth=3.2, headlength=3.6, headaxislength=3.2, color="#35e0ff",
                              edgecolor="k", linewidth=0.6, zorder=5)
        self.t_fz = {s: ax.text(self.foot_x[s], -L / 2 - 0.006, "", ha="center", va="top", fontsize=10, color=FG,
                                family="DejaVu Sans Mono") for s in "LR"}
        # shear legend arrow
        ax.quiver([-0.124], [-0.010], [0.0], [zx], angles="xy", scale_units="xy", scale=1.0, width=0.012, headwidth=3.2,
                  headlength=3.6, headaxislength=3.2, color="#35e0ff", edgecolor="k", linewidth=0.6)
        ax.text(-0.124, -0.018, f"shear\n{SHEAR_REF:.0f} N", va="top", ha="center", fontsize=8, color=MUTED)
        # colour bar
        cax = fig.add_axes([0.83, 0.575, 0.035, 0.27])
        cax.imshow(np.linspace(1, 0, 128)[:, None], aspect="auto", cmap=self.cmap, extent=(0, 1, 0, FZ_MAX))
        cax.set_xticks([])
        cax.yaxis.tick_right()
        cax.set_yticks([0, 50, 100, 150])
        cax.tick_params(colors=FG, labelsize=8.5, length=3)
        for sp in cax.spines.values():
            sp.set_edgecolor("#5a606b")
        fig.text(0.848, 0.858, "Fz, N\nper zone", ha="center", va="bottom", fontsize=8, color=MUTED)
        self.t_sum = fig.text(0.04, 0.513, "", va="top", fontsize=9.5, color=FG, family="DejaVu Sans Mono")

        # ---- model block
        fig.text(0.04, 0.468, "MODEL OUTPUT  (P+N net, updated every 0.2 s)  vs  TRUTH", va="top", fontsize=9.5, weight="bold",
                 color=MUTED)
        bx = fig.add_axes([0.27, 0.215, 0.50, 0.225])
        bx.set_facecolor(BG)
        bx.set_xlim(0, 1)
        bx.set_ylim(5.6, -0.6)
        for sp in bx.spines.values():
            sp.set_visible(False)
        bx.set_xticks([])
        bx.set_yticks([])
        for c in range(6):
            bx.add_patch(Rectangle((0, c - 0.36), 1.0, 0.72, facecolor="#22262e", edgecolor="none"))
        self.bars = [bx.add_patch(Rectangle((0, c - 0.36), 0.0, 0.72, facecolor=NEUTRAL, edgecolor="none")) for c in range(6)]
        self.bar_lab, self.bar_pct, self.truth_tag = [], [], []
        y0, dy = 0.4215, 0.03635
        for c in range(6):
            yy = y0 - c * dy
            fig.patches.append(Rectangle((0.04, yy - 0.011), 0.022, 0.022, transform=fig.transFigure, facecolor=CLASS_HEX[c],
                                         edgecolor="none"))
            self.bar_lab.append(fig.text(0.072, yy, CLASS_NAMES[c], va="center", fontsize=10, color=FG))
            self.bar_pct.append(fig.text(0.785, yy, "", va="center", fontsize=9, color=MUTED, family="DejaVu Sans Mono"))
            self.truth_tag.append(fig.text(0.868, yy, "", va="center", fontsize=8.5, weight="bold", color=FG))
        self.t_terr = fig.text(0.04, 0.192, "", va="top", fontsize=10.5, color=FG)
        self.t_flag = fig.text(0.96, 0.192, "", va="top", ha="right", fontsize=11.5, weight="bold", color=BAD)
        self.t_mu = fig.text(0.04, 0.160, "", va="top", fontsize=9.5, color=MUTED)
        fig.text(0.04, 0.128, "support state", va="top", fontsize=9.5, weight="bold", color=MUTED)
        self.t_sup = {s: [fig.text(x, yy, "", va="top", fontsize=10, color=FG) for x in (0.04, 0.23, 0.51, 0.80)]
                      for s, yy in (("L", 0.102), ("R", 0.074))}
        self.t_note = fig.text(0.04, 0.043, "", va="top", fontsize=7.5, color=MUTED, linespacing=1.25)
        self.border = Rectangle((0.006, 0.004), 0.988, 0.992, transform=fig.transFigure, fill=False, edgecolor=BAD,
                                linewidth=5, visible=False)
        fig.patches.append(self.border)

    def draw(self, tv, F, lab, pred):
        """F [2, Z, 3] sensor frame (or None), lab = {terrain, mu} now, pred = dict of the latest window or None."""
        m = self.meta
        self.t_time.set_text(f"t = {tv:5.2f} s")
        cls_now = lab["terrain"]
        mu = lab["mu"]
        self.t_sub.set_text(f"{m['gait_name']}   |   ground now: {CLASS_NAMES[cls_now] if cls_now >= 0 else 'not labelled'}"
                            + (f", friction {mu:.2f}" if np.isfinite(mu) else ""))
        if F is None:
            F = np.zeros((2, len(self.rects) // 2, 3))
        fz = F[..., 2].reshape(-1)
        for r, v in zip(self.rects, fz):
            r.set_facecolor(self.cmap(float(np.clip(v / FZ_MAX, 0, 1))))
        sh = F[..., :2].reshape(-1, 2).copy()
        mag = np.hypot(sh[:, 0], sh[:, 1])
        sh *= (np.minimum(mag, SHEAR_CLIP) / np.maximum(mag, 1e-9))[:, None]
        sh[(mag < 1.5) | (fz < 2.0)] = 0.0   # no arrow for an unloaded zone (sensor noise is 0.15-0.25 N)
        k = self.zx / SHEAR_REF
        self.quiv.set_UVC(-sh[:, 1] * k, sh[:, 0] * k)
        tot = F[..., 2].sum(1)
        for i, s in enumerate("LR"):
            self.t_fz[s].set_text(f"Fz {int(round(float(tot[i]))):5d} N")   # int: no "-0"
        self.t_sum.set_text(f"sum Fz both feet {int(round(float(tot.sum()))):5d} N    (robot weight {WEIGHT_N:.0f} N)")
        wrong = False
        if pred is None:
            for b in self.bars:
                b.set_width(0)
            for c in range(6):
                self.bar_pct[c].set_text("")
                self.truth_tag[c].set_text("")
            self.t_terr.set_text("terrain: no output yet (first 1 s window not full)")
            self.t_flag.set_text("")
            self.t_mu.set_text("")
            for s in "LR":
                for t in self.t_sup[s]:
                    t.set_text("")
        else:
            pc, tc = int(pred["terrain"]), int(pred["gt_terrain"])
            ok = pc == tc
            col = NEUTRAL if tc < 0 else (GOOD if ok else BAD)
            for c in range(6):
                self.bars[c].set_width(float(pred["prob"][c]))
                self.bars[c].set_facecolor(col if c == pc else "#4a505b")
                self.bar_pct[c].set_text(f"{100 * float(pred['prob'][c]):3.0f}%")
                self.bar_pct[c].set_color(FG if c == pc else MUTED)
                self.bar_lab[c].set_weight("bold" if c == tc else "normal")
                self.truth_tag[c].set_text("TRUTH" if c == tc else "")
            if tc < 0:
                self.t_terr.set_text(f"terrain: model {CLASS_NAMES[pc]}  |  truth not labelled (not scored)")
                self.t_flag.set_text("")
            else:
                self.t_terr.set_text(f"terrain: model {CLASS_NAMES[pc]}  |  truth {CLASS_NAMES[tc]}")
                self.t_flag.set_text("" if ok else "WRONG")
                wrong |= not ok
            if tc >= 0 and np.isfinite(pred["gt_mu"]) and pred["in_contact"]:
                self.t_mu.set_text(f"friction: model {float(pred['mu']):.2f}  |  truth {float(pred['gt_mu']):.2f}")
            else:
                self.t_mu.set_text("friction: not scored here (no contact or no label)")
            for s in "LR":
                ps, ts = int(pred[f"sup_{s}"]), int(pred[f"gt_sup_{s}"])
                a, b, c, d = self.t_sup[s]
                a.set_text({"L": "left foot", "R": "right foot"}[s])
                b.set_text(f"model {SUPPORT_NAMES[ps]}")
                b.set_color(SUPPORT_COL[ps])
                c.set_text(f"truth {SUPPORT_NAMES[ts]}")
                c.set_color(SUPPORT_COL[ts])
                d.set_text("" if ps == ts else "WRONG")
                d.set_color(BAD)
                d.set_weight("bold")
                wrong |= ps != ts
        self.border.set_visible(wrong)
        self.t_note.set_text(m.get("panel_note", ""))
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[:, :, :3]


# ------------------------------------------------------------------------------------------------ video
class Writer:
    def __init__(self, path, w, h, fps):
        import imageio_ffmpeg
        self.p = subprocess.Popen(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-profile:v", "high", "-crf", "18", "-preset", "medium", "-movflags", "+faststart", path],
            stdin=subprocess.PIPE)

    def write(self, frame):
        self.p.stdin.write(np.ascontiguousarray(frame, np.uint8).tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError("ffmpeg failed")


def _font(size, bold=False):
    import matplotlib
    from PIL import ImageFont
    f = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    return ImageFont.truetype(f, size)


def _banner(img, lines, color, y=14):
    """Text box at the top of the robot view. lines: list of str."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    f1, f2 = _font(21, True), _font(16)
    h = 16 + 28 + 22 * (len(lines) - 1)
    d.rectangle([14, y, VIEW_W - 14, y + h], fill=(12, 13, 16, 215), outline=color, width=3)
    d.text((28, y + 8), lines[0], font=f1, fill=color)
    for i, ln in enumerate(lines[1:]):
        d.text((28, y + 38 + 22 * i), ln, font=f2, fill=(232, 232, 232))
    return np.asarray(im)


def _caption(img, text):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    f = _font(15)
    w = d.textlength(text, font=f)
    d.rectangle([10, VIEW_H - 36, 24 + w, VIEW_H - 10], fill=(12, 13, 16, 190))
    d.text((17, VIEW_H - 32), text, font=f, fill=(232, 232, 232))
    return np.asarray(im)


def _tile(view, cls_true, pred):
    """400 x 360 montage tile: the robot view with a one-line caption (class, model output)."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(view).resize((TILE_W, TILE_H), Image.LANCZOS)
    d = ImageDraw.Draw(im, "RGBA")
    d.rectangle([0, 0, TILE_W, 34], fill=(12, 13, 16, 225))
    d.rectangle([8, 9, 24, 25], fill=tuple(int(255 * c) for c in GROUND[cls_true][0]))
    d.text((32, 6), CLASS_NAMES[cls_true], font=_font(17, True), fill=(232, 232, 232))
    if pred is not None and pred["gt_terrain"] >= 0:
        ok = int(pred["terrain"]) == int(pred["gt_terrain"])
        txt = f"model: {CLASS_NAMES[int(pred['terrain'])]}"
        f = _font(15)
        w = d.textlength(txt, font=f)
        d.text((TILE_W - w - 10, 8), txt, font=f, fill=(63, 191, 111) if ok else (240, 72, 62))
    d.rectangle([0, 0, TILE_W - 1, TILE_H - 1], outline=(12, 13, 16), width=2)
    return np.asarray(im)


def _event_banner(ev, tv):
    """(lines, colour) of the banner shown at video time tv, or None."""
    if not ev:
        return None
    k = ev["kind"]
    if k == "strip":
        new = CLASS_NAMES[ev["new_class"]]
        if ev["t_change"] <= tv < ev["t_detect"]:
            return ([f"SURFACE CHANGE at t = {ev['t_change']:.2f} s", f"first touchdown on {new}; the model has not switched yet"],
                    (240, 160, 48))
        if ev["t_detect"] <= tv < ev["t_detect"] + 2.5:
            return ([f"MODEL SWITCHED TO {new.upper()} at t = {ev['t_detect']:.2f} s",
                     f"{ev['latency_s']:.2f} s and {int(ev['latency_steps'])} step(s) after the first touchdown on {new}"],
                    (63, 191, 111))
    elif k == "lift":
        if ev["t_up"] <= tv < ev["t_down"]:
            d = ev.get("delay_s")
            return ([f"LIFTED by the trunk (t = {ev['t_up']:.2f} to {ev['t_down']:.2f} s)",
                     "both feet airborne from %.2f s; model says airborne %s" % (
                         ev["t_airborne"], f"{d:.2f} s later" if d is not None else "never (miss)")], (240, 160, 48))
    elif k == "edge_miss":
        if ev["t_p"] - 0.5 <= tv < ev["t_stop"] + 1.2:
            side = {"L": "LEFT", "R": "RIGHT"}[ev["side"]]
            return ([f"MISSED: {side} foot lands half on a plank edge",
                     f"truth: partial support from {ev['t_p']:.2f} to {ev['t_stop']:.2f} s; the model never says partial"],
                    (240, 72, 62))
    return None


def render_clip(bundle_base: str, out_dir: str, xml: str, max_frames: int | None = None, tiles: bool = True) -> dict:
    import mujoco
    meta = json.load(open(bundle_base + ".json"))
    z = np.load(bundle_base + ".npz")
    fps = meta["fps"]
    regions = meta["regions"]
    if len(regions) == 1:
        gtxt = f"{CLASS_NAMES[regions[0]['label']]}"
    else:
        gtxt = " -> ".join(CLASS_NAMES[r["label"]] for r in regions)
    meta["panel_title"] = gtxt
    ev = meta.get("events") or {}
    meta["panel_note"] = f"dataset v1 test episode {meta['episode']}\nP+N seed 0, stored test predictions; 1 s window"
    model = build_scene(meta, xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, VIEW_H, VIEW_W)
    opt = mujoco.MjvOption()
    opt.geomgroup[0] = 0   # collision geoms of the robot (the zone boxes) are not drawn
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation = 2.05, -14.0
    look, azim = camera_path(z["qpos"], fps)
    tvid = z["t_video"]
    frames = np.nonzero((tvid > meta["t0"] + 1e-9) & (tvid <= meta["t1"] + 1e-9))[0]
    if max_frames:
        frames = frames[:: max(1, len(frames) // max_frames)][:max_frames]
    panel = Panel(meta)
    os.makedirs(out_dir, exist_ok=True)
    w_main = Writer(os.path.join(out_dir, meta["clip"] + ".mp4"), VIEW_W + PANEL_W, VIEW_H, fps) if not max_frames else None
    w_tile = None
    if tiles and not max_frames and meta["group"] == "terrain":
        os.makedirs(os.path.join(out_dir, "tiles"), exist_ok=True)
        w_tile = Writer(os.path.join(out_dir, "tiles", meta["clip"] + ".mp4"), TILE_W, TILE_H, fps)
    t50 = z["t"]
    FL, FR = z["foot_zone_L"], z["foot_zone_R"]
    lt, lmu = z["label_terrain"], z["prop_mu"]
    sl, sr = z["label_support_L"], z["label_support_R"]
    pt = z["pred_t"]
    log = {"tv": [], "pred_terrain": [], "sum_fz": [], "gt_sup": [], "wrong": []}
    stills = []
    foot_body = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for s, n in
                 (("L", "left_ankle_roll_link"), ("R", "right_ankle_roll_link"))}
    for n, j in enumerate(frames):
        tv = float(tvid[j])
        data.qpos[:] = z["qpos"][j]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = look[j]
        cam.azimuth = float(azim[j])
        renderer.update_scene(data, camera=cam, scene_option=opt)
        if ev.get("kind") == "edge_miss" and ev["t_p"] - 0.5 <= tv < ev["t_stop"] + 1.2:
            sc = renderer.scene
            if sc.ngeom < sc.maxgeom:
                # red disc on the ground around the foot that is on the plank edge
                p = data.xpos[foot_body[ev["side"]]] + np.array([0.0, 0.0, -0.034])
                mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([0.15, 0.003, 0.0]), p,
                                    np.eye(3).flatten(), np.array([0.94, 0.20, 0.16, 0.55], np.float32))
                sc.ngeom += 1
        view = renderer.render().copy()
        i = int(np.floor(tv * 50 + 1e-6)) - 1           # latest finished 50 Hz frame
        F = np.stack([FL[i], FR[i]]) if i >= 0 else None
        k = int(np.searchsorted(pt, tv + 1e-6, side="right")) - 1   # latest finished window
        pred = None
        if k >= 0:
            fi = int(round(float(pt[k]) * 50)) - 1
            pred = {"terrain": z["pred_terrain"][k], "prob": z["pred_prob"][k], "sup_L": z["pred_sup_L"][k],
                    "sup_R": z["pred_sup_R"][k], "mu": z["pred_mu"][k], "gt_terrain": int(lt[fi]), "gt_mu": float(lmu[fi]),
                    "gt_sup_L": int(sl[fi]), "gt_sup_R": int(sr[fi]), "in_contact": bool(sl[fi] > 0 or sr[fi] > 0)}
        lab = {"terrain": int(lt[i]) if i >= 0 else int(lt[0]), "mu": float(lmu[i]) if i >= 0 else float(lmu[0])}
        pimg = panel.draw(tv, F, lab, pred)
        b = _event_banner(ev, tv)
        v2 = _banner(view, *b) if b else view
        v2 = _caption(v2, meta["title"])
        frame = np.concatenate([v2, pimg], 1)
        log["tv"].append(tv)
        log["pred_terrain"].append(int(pred["terrain"]) if pred else -9)
        log["sum_fz"].append(float(F[..., 2].sum()) if F is not None else 0.0)
        log["gt_sup"].append([int(sl[max(i, 0)]), int(sr[max(i, 0)])])
        log["wrong"].append(bool(panel.border.get_visible()))
        if w_main:
            w_main.write(frame)
        else:
            stills.append(frame)
        if w_tile:
            cls_true = int(regions[0]["label"])
            w_tile.write(_tile(view, cls_true, pred))
    for w in (w_main, w_tile):
        if w:
            w.close()
    renderer.close()
    res = {"clip": meta["clip"], "frames": int(len(frames)), "log": log}
    if stills:
        from PIL import Image
        for n, fr in enumerate(stills):
            Image.fromarray(fr).save(os.path.join(out_dir, f"{meta['clip']}_test{n}.png"))
    return res


def clip_checks(bundle_base: str, log: dict) -> dict:
    """Numbers for the acceptance checks of one rendered clip (what was actually drawn, from the render log)."""
    meta = json.load(open(bundle_base + ".json"))
    z = np.load(bundle_base + ".npz")
    ev = meta.get("events") or {}
    tv = np.array(log["tv"])
    pc = np.array(log["pred_terrain"])
    fz = np.array(log["sum_fz"])
    sup = np.array(log["gt_sup"])
    out = {"clip": meta["clip"], "frames": len(tv), "duration_s": round(len(tv) / meta["fps"], 2),
           "frames_marked_wrong": int(np.sum(log["wrong"])),
           "mean_sum_fz_both_full_N": float(fz[(sup == 2).all(1)].mean()) if (sup == 2).all(1).any() else None,
           "mean_sum_fz_all_frames_N": float(fz.mean()), "weight_N": WEIGHT_N}
    # standing start of the episode (command zero during the first second): both feet down, quasi-static
    m = (z["t"] > 0.5) & (z["t"] <= 1.0) & (z["label_support_L"] == 2) & (z["label_support_R"] == 2)
    if m.sum() > 5 and meta.get("gait") in ("g113", "g80"):
        out["standing_sum_fz_N"] = float((z["foot_zone_L"][m][..., 2].sum(1) + z["foot_zone_R"][m][..., 2].sum(1)).mean())
    if ev.get("kind") == "strip":
        after = np.nonzero((tv >= ev["t_change"] - 1e-6) & (pc == ev["new_class"]))[0]
        runs = [a for a in after if np.all(pc[a:a + 7] == ev["new_class"])]  # stays for the next window too (0.2 s = 6 frames)
        out["strip"] = {"t_change": ev["t_change"], "t_detect_reported": ev["t_detect"], "latency_reported_s": ev["latency_s"],
                        "first_video_time_showing_new_class": float(tv[runs[0]]) if runs else None,
                        "pred_before_change_all_old": bool(np.all(pc[(tv < ev["t_change"]) & (pc >= 0)] != ev["new_class"]))}
    if ev.get("kind") == "lift":
        m2 = (tv >= ev["t_airborne"] + 0.06) & (tv <= ev["t_down"] - ev.get("ramp", 0.5) - 0.1)
        out["lift"] = {"t_airborne": ev["t_airborne"], "max_sum_fz_while_airborne_N": float(np.abs(fz[m2]).max()),
                       "frames": int(m2.sum())}
    return out
