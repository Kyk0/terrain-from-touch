"""Step 2 of the videos (needs a renderer): bundles from scripts/render_prepare.py -> MP4 clips, montage, checks.

  export MUJOCO_GL=egl            # GPU machine; or osmesa for software rendering
  python scripts/render_clips.py --bundles bundles --out videos --xml assets/K1_22dof.xml [--workers 6]
  --test N    render only N evenly spaced frames per clip as PNG (quick look, no video)
  --only a,b  clip names
Writes <out>/<clip>.mp4 (1280x720, 30 fps, H.264), <out>/montage_terrains.mp4 (2x3 grid of the six terrain clips,
same frame index in every tile), <out>/frames/<clip>_{start,middle,end}.png and <out>/checks.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

ROOT = __file__.rsplit("/scripts/", 1)[0]
sys.path.insert(0, ROOT)
MONTAGE_ORDER = ["hard_grip", "hard_slip", "soft", "rough", "slope", "edge"]


def _one(a):
    base, out, xml, test = a
    from fla_terrain.viz.render import clip_checks, render_clip
    r = render_clip(base, out, xml, max_frames=test)
    return clip_checks(base, r["log"])


def montage(out):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    tiles = [os.path.join(out, "tiles", f"terrain_{c}_g113.mp4") for c in MONTAGE_ORDER]
    cmd = [ff, "-y", "-loglevel", "error"]
    for t in tiles:
        cmd += ["-i", t]
    lay = "0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0"
    cmd += ["-filter_complex", f"xstack=inputs=6:layout={lay}[g];[g]pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=0x15171c[v]",
            "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "high", "-crf", "18", "-r", "30",
            "-movflags", "+faststart", os.path.join(out, "montage_terrains.mp4")]
    subprocess.run(cmd, check=True)


def probe_and_stills(out):
    """ffprobe-like facts via ffmpeg, and three frames (start, middle, end) of every MP4."""
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    info = {}
    for mp4 in sorted(glob.glob(os.path.join(out, "*.mp4"))):
        name = os.path.basename(mp4)[:-4]
        n = int(subprocess.run([ff, "-i", mp4, "-map", "0:v:0", "-c", "copy", "-f", "null", "-"], capture_output=True,
                               text=True).stderr.rsplit("frame=", 1)[1].split()[0])
        hdr = subprocess.run([ff, "-i", mp4], capture_output=True, text=True).stderr
        vline = [ln.strip() for ln in hdr.splitlines() if "Video:" in ln][0]
        info[name] = {"frames": n, "stream": vline, "bytes": os.path.getsize(mp4)}
        for tag, k in (("start", 0), ("middle", n // 2), ("end", n - 1)):
            subprocess.run([ff, "-y", "-loglevel", "error", "-i", mp4, "-vf", f"select=eq(n\\,{k})", "-vframes", "1",
                            os.path.join(out, "frames", f"{name}_{tag}.png")], check=True)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--xml", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--test", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    bases = sorted(p[:-5] for p in glob.glob(os.path.join(args.bundles, "*.json")) if not p.endswith("clips.json"))
    if args.only:
        bases = [b for b in bases if os.path.basename(b) in args.only.split(",")]
    os.makedirs(args.out, exist_ok=True)
    jobs = [(b, args.out, os.path.abspath(args.xml), args.test) for b in bases]
    if args.workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
            checks = pool.map(_one, jobs, chunksize=1)
    else:
        checks = [_one(j) for j in jobs]
    for c in checks:
        print(json.dumps(c))
    if args.test:
        return 0
    if not args.only:
        montage(args.out)
    info = probe_and_stills(args.out)
    json.dump({"backend": os.environ.get("MUJOCO_GL", ""), "clips": checks, "files": info},
              open(os.path.join(args.out, "checks.json"), "w"), indent=1)
    print("RENDER_DONE", len(info), "mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
