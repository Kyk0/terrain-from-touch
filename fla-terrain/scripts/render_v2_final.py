#!/usr/bin/env python
"""B18: render the final v2 clips to render/v2/final/ (MUJOCO_GL=egl):
3 course videos (render_v2.final, selected by render_v2.select_final, poses from render_v2.rollout_final which must have
been run first), 3 sole close-ups at 0.25x, 5 foot-rig clips (render_v2.clips). Three still frames per clip in frames/."""
import json, os, subprocess, sys
from multiprocessing import Pool
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = "/path/to/fla-terrain/render/v2/final"


def frames(path, n=3):
    os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                               capture_output=True, text=True).stdout)
    outs = []
    for k in range(n):
        o = os.path.join(OUT, "frames", os.path.basename(path)[:-4] + f"_{k}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{dur * (k + 0.5) / n:.2f}", "-i", path, "-frames:v", "1", o], check=True)
        outs.append(os.path.basename(o))
    return dur, outs


def job(j):
    kind, arg = j
    if kind == "course":
        from render_v2.final import course_video
        r = course_video(*arg)
        path = os.path.join(OUT, r["file"])
    elif kind == "closeup":
        from render_v2 import clips
        path, _ = clips.closeup(arg, out=OUT)
        r = {"file": os.path.basename(path)}
    else:
        from render_v2 import clips
        path, _ = clips.rig_clip(arg, out=OUT)
        r = {"file": os.path.basename(path)}
    r["duration_s"], r["stills"] = frames(path)
    return r


if __name__ == "__main__":
    import json as _j
    from render_v2.select_final import select
    sel = select()
    nofall = {}
    for s in sel:
        nofall[s] = sum(not _j.load(open(f"/path/to/fla-terrain/data/v2_course/episode_course_{s}_seed{k:03d}.json"))["fell"] for k in range(40))
    jobs = [("course", (v["episode"], s, v["crossed_of_40"], nofall[s])) for s, v in sel.items()]
    jobs += [("closeup", c) for c in ("hard_grip", "hard_slip", "slope")]
    jobs += [("rig", k) for k in ("press", "drag03", "drag09", "twist", "edge")]
    with Pool(len(jobs)) as p:
        res = p.map(job, jobs, chunksize=1)
    json.dump({"selection": sel, "clips": res}, open(os.path.join(OUT, "clips.json"), "w"), indent=1, default=float)
    for r in res:
        print(f"[render] {r['file']:34s} {r['duration_s']:6.1f} s  {r.get('caption', '')[:90]}")
