"""Video writer, scene styling, camera and a simulation tap (pose + recorded sole frames) for the v2 renders."""
from __future__ import annotations

import os
import subprocess

import mujoco
import numpy as np

W, H, FPS = 1280, 720, 30
NOMINAL_COMPLIANCE = 0.0002     # m per 100 N (DECISIONS: nominal changed 0.5 -> 0.2 mm before the full run)
CLASS_RGBA = {0: (0.50, 0.50, 0.49, 1), 1: (0.42, 0.66, 0.88, 1), 2: (0.70, 0.57, 0.33, 1), 3: (0.52, 0.48, 0.42, 1),
              4: (0.50, 0.70, 0.48, 1), 5: (0.66, 0.44, 0.22, 1)}


def nominal_params():
    from fla_terrain.sim.pointmap import SoleV2Params
    return SoleV2Params(compliance=NOMINAL_COMPLIANCE)


class Writer:
    def __init__(self, path, w=W, h=H, fps=FPS):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.n = 0
        self.p = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
             "-preset", "medium", "-movflags", "+faststart", path], stdin=subprocess.PIPE)

    def write(self, frame):
        self.p.stdin.write(np.ascontiguousarray(frame, np.uint8).tobytes())
        self.n += 1

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError("ffmpeg failed")


def style(m, info):
    """Colour terrain geoms by region class (pit planes black, small boxes = bumps dark), floor-ish lighting."""
    regions = info.get("terrain_regions") or __import__("fla_terrain.sim.build", fromlist=["x"]).terrain_regions(
        info["terrain_cfg"])[0]
    cfg = info["terrain_cfg"]
    geoms = cfg.get("geoms") or [{"type": "plane"}]
    for i, gid in enumerate(info["terrain_geoms"]):
        if m.geom_matid[gid] >= 0:          # textured (rig floor): keep the texture
            m.geom_rgba[gid] = (1, 1, 1, 1)
            continue
        gd = geoms[i] if i < len(geoms) else {"type": "plane"}
        reg = regions[int(gd.get("region", 0))]
        pos = gd.get("pos") or [0, 0, 0]
        if gd["type"] == "plane" and pos[2] < -0.01:
            m.geom_rgba[gid] = (0.03, 0.03, 0.04, 1)
        elif gd["type"] == "box" and max(gd["size"][0], gd["size"][1]) <= 0.06:
            m.geom_rgba[gid] = (0.16, 0.13, 0.11, 1)
        else:
            m.geom_rgba[gid] = CLASS_RGBA.get(int(reg["label"]), (0.5, 0.5, 0.5, 1))
    if info.get("rig"):   # the rig foot is only the sole contact blocks (transparent in the model): show them
        for gid in range(m.ngeom):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if nm.startswith("sole_"):
                m.geom_rgba[gid] = (0.75, 0.80, 0.88, 1) if nm.endswith("_mid") else (0.30, 0.45, 0.70, 1)
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W)
    m.vis.global_.offheight = max(m.vis.global_.offheight, H)
    m.vis.quality.shadowsize = 4096
    m.vis.headlight.ambient[:] = (0.35, 0.35, 0.35)
    m.vis.headlight.diffuse[:] = (0.4, 0.4, 0.4)


def camera(lookat, distance, azimuth, elevation):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = lookat
    c.distance, c.azimuth, c.elevation = distance, azimuth, elevation
    return c


def smooth(x, sigma):
    if sigma <= 0:
        return x
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    xp = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)])
    return np.stack([np.convolve(xp[:, i], k, mode="valid") for i in range(x.shape[1])], 1)


class Tap:
    """Runs alongside a simulation: after every mj_step call .step(); keeps the qpos at every 50 Hz sensor frame
    (the recorder's frame clock) and the recorded sole frames rec.frames['F'] (raw sensor output)."""

    def __init__(self, rec):
        self.rec = rec
        self.qpos, self.t = [], []
        self._n = 0

    def step(self, data, cmd=(0.0, 0.0, 0.0)):
        self.rec.step(data, cmd)
        n = len(self.rec.frames["F"])
        if n > self._n:
            self._n = n
            self.qpos.append(data.qpos.copy())
            self.t.append(n / self.rec.rate)

    def arrays(self):
        F = np.asarray(self.rec.frames["F"])                          # [T, 2, P, 3]
        contact = np.asarray(self.rec.frames["contact"], bool).reshape(len(F), 2)
        return {"t": np.asarray(self.t), "qpos": np.asarray(self.qpos), "F": F, "contact": contact}
