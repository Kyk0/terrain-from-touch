"""Raw v2 point-map snapshots (recorded maps after the sensor model, nominal sole parameters; no smoothing).

Generates short dedicated episodes (nominal setting) and saves PNGs to --out:
  standing (PD stand, hard grip), heel strike / mid-stance / toe-off (G113 walking, hard grip, left foot),
  plank edge (G113 on edge planks, a frame with partial support), bump (G113 on rough, the stance frame with the
  largest middle-part share).
Each PNG: both feet, Fz per point (fixed colour scale, N), absent points grey, part boundaries, the four block shear
vectors (S_blk) and block normal sums (N_blk) printed, CoP marked.
  $PY scripts/v2_map_snapshots.py --out /path/to/fla-terrain/render/pilot_maps
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

ROOT = __file__.rsplit("/scripts/", 1)[0]
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fla_terrain.sim.pointmap import BLOCK4, views  # noqa: E402


def episode(out, terrain, source, variant=None, seconds=8.0, idx=0):
    from gen_data import run_episode
    eid = f"snap_{terrain}_{source}_{idx}"
    run_episode({"terrain": terrain, "variant": variant, "source": source, "split": "train", "seconds": seconds,
                 "seed": 7, "index": idx, "out": out, "episode_id": eid, "split_key": eid, "sole": "v2",
                 "sole_setting": "nominal"})
    p = os.path.join(out, f"episode_{eid}.npz")
    return np.load(p), json.load(open(p[:-4] + ".json"))


def plot(z, meta, t, title, path, vmax=None):
    lay = meta["zone_layout"]
    xy = np.asarray(lay["points_xy_m"]) * 1e3
    pres = np.asarray(lay["present"], bool)
    part = np.asarray(lay["part"])
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 6.2))
    for ax, s in zip(axs, "LR"):
        f = z[f"foot_points_{s}"][t]
        v = views(f[None], lay)
        fz = f[:, 2]
        vm = vmax or max(1.0, float(np.percentile(z[f"foot_points_{s}"][..., 2], 99.5)))
        sc = ax.scatter(xy[pres, 1], xy[pres, 0], c=fz[pres], s=60, marker="s", cmap="viridis", vmin=0, vmax=vm)
        ax.scatter(xy[~pres, 1], xy[~pres, 0], c="0.8", s=60, marker="s")
        for b in (part == 0, part == 2):
            pass
        xb = [xy[part == 0, 0].max() + 5, xy[part == 2, 0].min() - 5]
        for x in xb:
            ax.axhline(x, color="w", lw=0.8, ls="--")
        if fz.sum() > 1:
            c = (fz @ xy) / fz.sum()
            ax.plot(c[1], c[0], "r+", ms=14, mew=2)
        cen = {"front_l": (17.5, 60), "front_r": (-17.5, 60), "back_l": (17.5, -60), "back_r": (-17.5, -60)}
        Sb, Nb = v["S_blk"][0], v["N_blk"][0]
        for i, b in enumerate(BLOCK4):
            yx = cen[b]
            ax.arrow(yx[0], yx[1], Sb[i, 1] * 0.8, Sb[i, 0] * 0.8, color="orange", width=0.6, head_width=3,
                     length_includes_head=True)
            ax.text(yx[0], yx[1] - 9, f"{Nb[i]:.0f} N", ha="center", fontsize=7, color="w")
        ax.set_title(f"{s} foot: Fz {fz.sum():.0f} N, mid {fz[part == 1].sum():.0f} N", fontsize=8)
        ax.set_xlim(45, -45)
        ax.set_ylim(-100, 100)
        ax.set_aspect("equal")
        ax.set_xlabel("y (mm, left ->)" if s == "L" else "y (mm)")
        if s == "L":
            ax.set_ylabel("x (mm, toe up)")
        plt.colorbar(sc, ax=ax, fraction=0.05, label="Fz per point (N)")
    fig.suptitle(f"{title}  (t = {z['t'][t]:.2f} s; arrows = block shear S_blk x0.8 mm/N)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def stance_frames(z, side="L"):
    on = z[f"contact_gt_{side}"].astype(bool)
    runs, s = [], None
    for i, v in enumerate(np.append(on, False)):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if i - s >= 8:
                runs.append((s, i - 1))
            s = None
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/path/to/fla-terrain/render/pilot_maps")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        z, m = episode(td, "hard_grip", "stand", seconds=4.0)
        plot(z, m, len(z["t"]) - 1, "standing, hard grip (PD stand)", os.path.join(args.out, "1_standing.png"))
        z, m = episode(td, "hard_grip", "walk_g113", seconds=8.0)
        a, b = stance_frames(z)[2]
        plot(z, m, a, "heel strike (first stance frame, left foot), G113 hard grip", os.path.join(args.out, "2_heel_strike.png"))
        plot(z, m, (a + b) // 2, "mid-stance, G113 hard grip", os.path.join(args.out, "3_mid_stance.png"))
        plot(z, m, b, "toe-off (last stance frame), G113 hard grip", os.path.join(args.out, "4_toe_off.png"))
        z, m = episode(td, "edge", "walk_g113", variant="planks", seconds=8.0)
        part = np.where((z["label_support_L"] == 1) & (z["foot_points_L"][..., 2].sum(1) > 50))[0]
        plot(z, m, part[len(part) // 2], "left foot on a plank edge (partial support), G113",
             os.path.join(args.out, "5_plank_edge.png"))
        z, m = episode(td, "rough", "walk_g113", seconds=8.0)
        lp = np.asarray(m["zone_layout"]["part"])
        f = z["foot_points_L"][..., 2]
        share = f[:, lp == 1].sum(1) / np.maximum(f.sum(1), 1e-9) * (f.sum(1) > 50)
        plot(z, m, int(np.argmax(share)), f"left foot on a bump (rough, max middle share {100 * share.max():.0f} %), G113",
             os.path.join(args.out, "6_bump.png"))
    print("saved", sorted(os.listdir(args.out)))


if __name__ == "__main__":
    main()
