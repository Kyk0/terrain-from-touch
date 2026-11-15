"""Frame cache for schema v0.3 (sole v2 point maps) and the sensor views.

One per-point recording gives every view (V2_BRIEF section 2):
  Nreal  per-point normal force map            [2 feet, P]     (fed as a 2D map per foot)
  Sreal  per-point shear (Fx, Fy) map          [2 feet, P, 2]  (2D map; APPROXIMATE: see pointmap.py)
  Nblk   4 block normal sums per foot          [2, 4]          (front_l, front_r, back_l, back_r)
  Sblk   4 block shear vectors per foot        [2, 4, 2]
  P      proprioception: joint pos/vel, gyro, projected gravity (no command); 'Pcmd' adds cmd_vel
Input names join tokens with '+', e.g. 'P+Nreal+Sblk', 'Nblk+Sblk', 'Sreal', 'P'.

Arrays held: V [frames, 53] float32 (P + cmd), PTS [frames, 2, P, 3] float16 (sensor output, N), BLK [frames, 2, 4, 3]
float32 (block sums of PTS: Fx, Fy, Fz). Windows: 1.0 s / 0.2 s stride, label at the last frame, windows ending
after a fall dropped (as v1).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from ..data.loader import SPLITS, check_no_leakage, window_starts
from ..sim.pointmap import BLOCK4
from .dataset import STRIDE, W, _band_group, _flat_items

TOKENS = ("P", "Pcmd", "Nreal", "Sreal", "Nblk", "Sblk")
CACHE_VERSION = 4  # v4: course chunk list per episode; v3: v3: per-foot ground labels (ground_L/R, -1 if absent); v2: per-episode part labels and 4-block weights (boundaries are randomized per episode)


def cache_path_v2(data):
    return data.rstrip("/") + f"_cache_v2pts{CACHE_VERSION}.npz"


def parse_view(view: str) -> list[str]:
    toks = view.split("+")
    if not toks or any(t not in TOKENS for t in toks) or ("P" in toks and "Pcmd" in toks):
        raise ValueError(f"bad view {view!r}; tokens {TOKENS}")
    return toks


def _read(p):
    meta = json.load(open(p[:-4] + ".json"))
    if meta.get("schema_version") != "0.3":
        raise ValueError(f"{p}: not a v0.3 episode")
    z = np.load(p)
    T = len(z["t"])
    lay = meta["zone_layout"]
    Wb = np.asarray(lay["block_weights"], float)
    W4 = Wb[:, [lay["blocks"].index(b) for b in BLOCK4]]          # [P, 4]
    pts = []
    for s in "LR":
        f = np.nan_to_num(z[f"foot_points_{s}"].astype(np.float32))
        f[~z[f"foot_valid_{s}"]] = 0.0
        pts.append(f)
    pts = np.stack(pts, 1)                                         # [T, 2, P, 3]
    blk = np.einsum("tfpc,pb->tfbc", pts, W4).astype(np.float32)   # [T, 2, 4, 3]
    V = np.concatenate([z["joint_pos"], z["joint_vel"], z["imu_gyro"], z["imu_gravity"], z["cmd_vel"]], 1).astype(np.float32)
    lab = {"terrain": z["label_terrain"].astype(np.int8), "sup_L": z["label_support_L"].astype(np.int8),
           "sup_R": z["label_support_R"].astype(np.int8), "mu": z["prop_mu"].astype(np.float32),
           "logk": np.log10(z["prop_stiffness"].astype(np.float64)).astype(np.float32),
           "slope": z["prop_slope"].astype(np.float32), "t": z["t"].astype(np.float32),
           "ground_L": (z["label_ground_L"] if "label_ground_L" in z.files else np.full(T, -1)).astype(np.int8),
           "ground_R": (z["label_ground_R"] if "label_ground_R" in z.files else np.full(T, -1)).astype(np.int8)}
    tp = meta["terrain_params"]
    em = {"episode": str(meta["episode_id"]), "split_key": meta["split_key"], "range_split": tp.get("range_split", ""),
          "mode": meta["mode"], "class": tp.get("class", ""), "variant": tp.get("variant"),
          "source": tp.get("source", meta["motion_source"]), "band": _band_group(tp), "fall_time": meta.get("fall_time"),
          "touchdowns": meta.get("touchdowns", []), "regions": [r["label"] for r in tp.get("regions", [])],
          "lift": tp.get("lift"), "unknown_intervals": meta.get("unknown_intervals", []), "T": T,
          "J": z["joint_pos"].shape[1], "P": pts.shape[2], "present": lay.get("present", [1] * pts.shape[2]),
          "sole": tp.get("sole"), "sole_params": meta["sensor_model"].get("params"),
          "part": lay["part"], "block_w4": W4.round(3).tolist(),
          "chunks": [{k: c[k] for k in ("index", "x0", "x1", "class", "label", "region")} for c in tp.get("chunks") or []],
          "bands": {k: v for k, v in _flat_items(tp) if k.endswith("band")},
          "mu_values": [r["mu"] for r in tp.get("regions", [])], "k_values": [r["stiffness"] for r in tp.get("regions", [])]}
    # float16 cache: clip at +-60000 N (fp16 max 65504); only reached by unsaturated impact spikes (D19 'nosat' arm)
    return V, np.clip(pts, -6e4, 6e4).astype(np.float16), blk, lab, em, {k: lay[k] for k in ("rowcol", "points_xy_m", "part", "blocks")}


class FrameDataV2:
    def __init__(self, V, PTS, BLK, lab, eps, starts, ep_of_win, layout):
        self.V, self.PTS, self.BLK, self.lab, self.eps = V, PTS, BLK, lab, eps
        self.starts, self.ep_of_win = starts, ep_of_win
        self.last = starts + W - 1
        self.layout = layout  # grid / point coordinates (shared); part and block weights are PER EPISODE (eps[i])
        rc = np.asarray(layout["rowcol"])
        self.grid = (int(rc[:, 0].max()) + 1, int(rc[:, 1].max()) + 1)
        assert self.grid[0] * self.grid[1] == len(rc) and np.all(rc[:, 0] * self.grid[1] + rc[:, 1] == np.arange(len(rc))), \
            "point order must be the row-major full grid (rect mask)"
        J = eps[0]["J"]
        self.n_p = 2 * J + 6                       # joints, gyro, gravity (no command)

    @classmethod
    def build(cls, root, cache=None, workers=32):
        if cache and os.path.exists(cache):
            d = np.load(cache, allow_pickle=False)
            ex = json.load(open(cache + ".json"))
            lab = {k[4:]: d[k] for k in d.files if k.startswith("lab_")}
            return cls(d["V"], d["PTS"], d["BLK"], lab, ex["eps"], d["starts"], d["ep_of_win"], ex["layout"])
        paths = sorted(glob.glob(f"{root}/**/episode_*.npz", recursive=True))
        from multiprocessing import Pool
        with Pool(workers) as pool:
            res = pool.map(_read, paths, chunksize=4)
        Vs, Ps, Bs, labs, eps, starts, epw = [], [], [], {k: [] for k in res[0][3]}, [], [], []
        off = 0
        for i, (V, pts, blk, lab, em, lay) in enumerate(res):
            T = len(V)
            st = window_starts(T, W, STRIDE)
            if em["fall_time"] is not None:
                st = st[lab["t"][st + W - 1] <= em["fall_time"]]
            em["offset"] = off
            Vs.append(V)
            Ps.append(pts)
            Bs.append(blk)
            for k in labs:
                labs[k].append(lab[k])
            starts.append(st + off)
            epw.append(np.full(len(st), i, np.int32))
            eps.append(em)
            off += T
        lab = {k: np.concatenate(v) for k, v in labs.items()}
        out = cls(np.concatenate(Vs), np.concatenate(Ps), np.concatenate(Bs), lab, eps, np.concatenate(starts),
                  np.concatenate(epw), res[0][5])
        if cache:
            np.savez(cache, V=out.V, PTS=out.PTS, BLK=out.BLK, starts=out.starts, ep_of_win=out.ep_of_win,
                     **{f"lab_{k}": v for k, v in lab.items()})
            json.dump({"eps": eps, "layout": out.layout}, open(cache + ".json", "w"))
        return out

    # same interface as the v1 FrameData for evaluate.py
    def win_labels(self, idx=None):
        last = self.last if idx is None else self.last[idx]
        return {k: v[last] for k, v in self.lab.items()}

    def split_indices(self) -> dict:
        sp = np.array([e["range_split"] for e in self.eps])[self.ep_of_win]
        out = {s: np.nonzero(sp == s)[0] for s in SPLITS}
        check_no_leakage({s: [self.eps[e]["split_key"] for e in np.unique(self.ep_of_win[i])] for s, i in out.items()})
        return out

    def view_dims(self, view):
        toks = parse_view(view)
        c_vec = (self.n_p if "P" in toks else 0) + (self.n_p + 3 if "Pcmd" in toks else 0) + \
            (8 if "Nblk" in toks else 0) + (16 if "Sblk" in toks else 0)
        c_map = (1 if "Nreal" in toks else 0) + (2 if "Sreal" in toks else 0)
        return c_vec, c_map
