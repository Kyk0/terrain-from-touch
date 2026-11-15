"""B18 selection rule (fixed in code, applied in course-seed order 0..39): for each walking source, the first recorded
course run of data/v2_course that crosses every chunk = no fall and final base x (terrain_params.base_xy_1hz, last
sample) >= course_end_x. Also counts how many of the source's 40 runs cross every chunk."""
import glob
import json
import os

DATA = "/path/to/fla-terrain/data/v2_course"
SOURCES = ("walk_g113", "walk_g80", "walk_sonic")


def crossed(j):
    tp = j["terrain_params"]
    xy = tp["base_xy_1hz"]
    return (not j.get("fell")) and xy[-1][0] >= tp["course_end_x"]


def select():
    out = {}
    for s in SOURCES:
        runs = []
        for seed in range(40):
            p = os.path.join(DATA, f"episode_course_{s}_seed{seed:03d}.json")
            j = json.load(open(p))
            runs.append((seed, j["episode_id"], crossed(j)))
        first = next(r for r in runs if r[2])
        out[s] = {"episode": first[1], "seed": first[0], "crossed_of_40": sum(r[2] for r in runs)}
    return out


if __name__ == "__main__":
    print(json.dumps(select(), indent=1))
