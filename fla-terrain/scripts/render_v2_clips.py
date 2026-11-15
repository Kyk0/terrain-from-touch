#!/usr/bin/env python
"""B10: render the FLA-0 v2 clips (MUJOCO_GL=egl).  python scripts/render_v2_clips.py [rig closeup course] [--only name]
Outputs /path/to/fla-terrain/render/v2/*.mp4 and three check frames per clip in render/v2/frames/."""
import argparse, os, subprocess, sys, time
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from render_v2 import clips as C  # noqa: E402

FR = os.path.join(C.OUT, "frames")


def frames(path, n=3):
    os.makedirs(FR, exist_ok=True)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                               capture_output=True, text=True).stdout)
    outs = []
    for k in range(n):
        t = dur * (k + 0.5) / n
        o = os.path.join(FR, os.path.basename(path)[:-4] + f"_{k}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", path, "-frames:v", "1", o], check=True)
        outs.append(o)
    return dur, outs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="*", default=["rig", "closeup"])
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    jobs = []
    if "rig" in a.groups:
        jobs += [("rig", k) for k in ("press", "drag03", "drag09", "twist", "edge")]
    if "closeup" in a.groups:
        jobs += [("closeup", k) for k in ("hard_grip", "hard_slip", "slope")]
    for g, k in jobs:
        if a.only and k != a.only:
            continue
        t0 = time.time()
        path, _ = C.rig_clip(k) if g == "rig" else C.closeup(k)
        dur, fr = frames(path)
        print(f"[render] {os.path.basename(path)} {dur:.1f} s, {time.time() - t0:.0f} s wall; frames {', '.join(map(os.path.basename, fr))}", flush=True)
