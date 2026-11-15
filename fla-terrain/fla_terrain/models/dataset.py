"""Frame cache for training/evaluation at scale.

All episodes are read once into a frame matrix ALL [total_frames, C_all] holding every channel group; an input
set is a column selection. Windows (1.0 s, stride 0.2 s, label at the last frame, windows ending after a fall
dropped) are stored as global start indices, and gathered per batch (on the GPU for training).

Channel groups (per frame):
  P      joint_pos (J), joint_vel (J), imu_gyro (3), imu_gravity (3), cmd_vel (3)
  N      Fz per zone, L then R (2Z)
  S      Fx per zone L, Fy per zone L, Fx R, Fy R (4Z)
  N_lp   Fz per zone low-passed: causal 2nd-order Butterworth, 4 Hz (removes frame-to-frame chatter)
  Ntot   total Fz per foot (2);  Stot  total Fx, Fy per foot (4)
  U      friction utilisation u = |shear| / Fz per foot (2) and per zone (2Z); u = 0 where Fz < U_FLOOR
         (2 % of body weight, "unloaded"), clipped to [0, 2]  (exploratory input, R4)
Input sets: P, P+N, P+N+S (main), P+N_lp, P+Ntot, P+Ntot+Stot (D7 diagnostics), P+N+S+U (exploratory).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from scipy.signal import butter, lfilter

from ..data.loader import SPLITS, check_no_leakage, window_starts

INPUT_SETS_ALL = ("P", "P+N", "P+N+S", "P+N_lp", "P+Ntot", "P+Ntot+Stot")
INPUT_SETS_EXPLORATORY = ("P+N+S+U",)
CACHE_VERSION = 2  # v2 adds the U group
U_FLOOR = 0.02 * 19.666 * 9.81  # N (2 % of the K1 weight, ~3.9 N)


def cache_path(data):
    return data.rstrip("/") + f"_cache_v{CACHE_VERSION}.npz"


def utilisation(fx, fy, fz):
    u = np.where(fz >= U_FLOOR, np.hypot(fx, fy) / np.maximum(fz, U_FLOOR), 0.0)
    return np.clip(u, 0.0, 2.0)
W, STRIDE = 50, 10
LP_B, LP_A = butter(2, 4.0, fs=50.0)


def _groups(J, Z):
    g, i = {}, 0
    for name, n in (("P", 2 * J + 9), ("N", 2 * Z), ("S", 4 * Z), ("N_lp", 2 * Z), ("Ntot", 2), ("Stot", 4), ("U", 2 + 2 * Z)):
        g[name] = np.arange(i, i + n)
        i += n
    return g, i


def _band_group(tp):
    """'test_out' if any band in the terrain params (also nested strip before/after) is test_out, else the
    first band name found ('test_in', 'train_a', 'val', ...)."""
    bands = []

    def walk(d):
        for k, v in d.items():
            if k.endswith("band") and isinstance(v, str):
                bands.append(v)
            elif isinstance(v, dict):
                walk(v)
    walk(tp)
    if "test_out" in bands:
        return "test_out"
    return bands[0] if bands else ""


def _read(p):
    meta = json.load(open(p[:-4] + ".json"))
    z = np.load(p)
    Fl, Fr = z["foot_zone_L"].astype(np.float32), z["foot_zone_R"].astype(np.float32)
    for F, v in ((Fl, z["foot_valid_L"]), (Fr, z["foot_valid_R"])):
        F[~v] = 0.0
    Fl, Fr = np.nan_to_num(Fl), np.nan_to_num(Fr)
    T = len(z["t"])
    fz = np.concatenate([Fl[..., 2], Fr[..., 2]], 1)
    cols = [z["joint_pos"], z["joint_vel"], z["imu_gyro"], z["imu_gravity"], z["cmd_vel"], fz,
            Fl[..., 0], Fl[..., 1], Fr[..., 0], Fr[..., 1], lfilter(LP_B, LP_A, fz, axis=0),
            np.stack([Fl[..., 2].sum(1), Fr[..., 2].sum(1)], 1),
            np.stack([Fl[..., 0].sum(1), Fl[..., 1].sum(1), Fr[..., 0].sum(1), Fr[..., 1].sum(1)], 1),
            np.stack([utilisation(F[..., 0].sum(1), F[..., 1].sum(1), F[..., 2].sum(1)) for F in (Fl, Fr)], 1),
            utilisation(Fl[..., 0], Fl[..., 1], Fl[..., 2]), utilisation(Fr[..., 0], Fr[..., 1], Fr[..., 2])]
    X = np.concatenate([np.asarray(c, np.float32).reshape(T, -1) for c in cols], 1)
    lab = {"terrain": z["label_terrain"].astype(np.int8), "sup_L": z["label_support_L"].astype(np.int8),
           "sup_R": z["label_support_R"].astype(np.int8), "mu": z["prop_mu"].astype(np.float32),
           "logk": np.log10(z["prop_stiffness"].astype(np.float64)).astype(np.float32),
           "slope": z["prop_slope"].astype(np.float32), "t": z["t"].astype(np.float32)}
    tp = meta["terrain_params"]
    em = {"episode": str(meta["episode_id"]), "split_key": meta["split_key"], "range_split": tp.get("range_split", ""),
          "mode": meta["mode"], "class": tp.get("class", ""), "variant": tp.get("variant"),
          "source": tp.get("source", meta["motion_source"]), "band": _band_group(tp), "fall_time": meta.get("fall_time"),
          "touchdowns": meta.get("touchdowns", []), "regions": [r["label"] for r in tp.get("regions", [])],
          "lift": tp.get("lift"), "unknown_intervals": meta.get("unknown_intervals", []), "T": T,
          "J": z["joint_pos"].shape[1], "Z": Fl.shape[1],
          "bands": {k: v for k, v in _flat_items(tp) if k.endswith("band")},
          "mu_values": [r["mu"] for r in tp.get("regions", [])],
          "k_values": [r["stiffness"] for r in tp.get("regions", [])]}
    return X, lab, em


def _flat_items(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flat_items(v, prefix + k + ".")
        else:
            yield prefix + k, v


class FrameData:
    def __init__(self, X, lab, eps, starts, ep_of_win):
        self.X, self.lab, self.eps = X, lab, eps
        self.starts, self.ep_of_win = starts, ep_of_win
        J, Z = eps[0]["J"], eps[0]["Z"]
        self.groups, _ = _groups(J, Z)
        self.Z = Z
        self.last = starts + W - 1

    @classmethod
    def build(cls, root, cache=None, workers=32):
        if cache and os.path.exists(cache):
            d = np.load(cache, allow_pickle=False)
            eps = json.load(open(cache + ".json"))
            lab = {k[4:]: d[k] for k in d.files if k.startswith("lab_")}
            return cls(d["X"], lab, eps, d["starts"], d["ep_of_win"])
        paths = sorted(glob.glob(f"{root}/**/episode_*.npz", recursive=True))
        from multiprocessing import Pool
        with Pool(workers) as pool:
            res = pool.map(_read, paths, chunksize=16)
        Xs, labs, eps, starts, epw = [], {k: [] for k in res[0][1]}, [], [], []
        off = 0
        for i, (X, lab, em) in enumerate(res):
            T = len(X)
            st = window_starts(T, W, STRIDE)
            last = st + W - 1
            if em["fall_time"] is not None:
                st = st[lab["t"][last] <= em["fall_time"]]
            em["offset"] = off
            Xs.append(X)
            for k in labs:
                labs[k].append(lab[k])
            starts.append(st + off)
            epw.append(np.full(len(st), i, np.int32))
            eps.append(em)
            off += T
        X = np.concatenate(Xs)
        lab = {k: np.concatenate(v) for k, v in labs.items()}
        starts, epw = np.concatenate(starts), np.concatenate(epw)
        if cache:
            np.savez(cache, X=X, starts=starts, ep_of_win=epw, **{f"lab_{k}": v for k, v in lab.items()})
            json.dump(eps, open(cache + ".json", "w"))
        return cls(X, lab, eps, starts, epw)

    # ---- channels / labels -------------------------------------------------------------------------------
    def columns(self, input_set: str) -> np.ndarray:
        parts = input_set.split("+")
        if parts[0] != "P" or any(p not in self.groups for p in parts):
            raise ValueError(input_set)
        return np.concatenate([self.groups[p] for p in parts])

    def win_labels(self, idx=None):
        last = self.last if idx is None else self.last[idx]
        return {k: v[last] for k, v in self.lab.items()}

    # ---- splits ----------------------------------------------------------------------------------------------
    def split_of_episode(self) -> np.ndarray:
        return np.array([e["range_split"] for e in self.eps])

    def split_indices(self) -> dict:
        """{split: window indices}; split = generation-time range_split of the window's episode. Checks leakage."""
        sp = self.split_of_episode()[self.ep_of_win]
        out = {s: np.nonzero(sp == s)[0] for s in SPLITS}
        check_no_leakage({s: [self.eps[e]["split_key"] for e in np.unique(self.ep_of_win[i])] for s, i in out.items()})
        return out


def leakage_report(fd: FrameData) -> list[str]:
    """split_key in two splits; friction / stiffness values or band names shared between splits."""
    from ..sim.terrains import BANDS, RIGID_STIFFNESS, stiffness_from_tc
    lines, errs = [], []
    keys = {s: set() for s in SPLITS}
    for e in fd.eps:
        keys[e["range_split"]].add(e["split_key"])
    shared = [(a, b) for a in SPLITS for b in SPLITS if a < b and keys[a] & keys[b]]
    lines.append(f"split_keys: " + ", ".join(f"{s} {len(keys[s])}" for s in SPLITS) +
                 f"; shared between splits: {shared or 'none'}")
    errs += [f"split_key shared {p}" for p in shared]
    # band names must start with their split ('train_a' in train, 'val' in val, 'test_*' in test)
    bad_names = [e["episode"] for e in fd.eps for b in e["bands"].values() if not b.startswith(e["range_split"])]
    # every friction / stiffness value must lie only in bands of its own split
    iv = {p: [(s, lo, hi) for s, bands in BANDS[p].items() for _, lo, hi in bands] for p in BANDS}
    kiv = [(s, stiffness_from_tc(hi), stiffness_from_tc(lo)) for s, lo, hi in iv["soft_tc"]]
    bad_vals = 0
    for e in fd.eps:
        for mu in e["mu_values"]:
            own = {s for s, lo, hi in iv["mu_grip"] + iv["mu_slip"] if lo - 1e-9 <= mu <= hi + 1e-9}
            bad_vals += own != {e["range_split"]}
        for k in e["k_values"]:
            if k != RIGID_STIFFNESS:
                own = {s for s, lo, hi in kiv if lo * (1 - 1e-6) <= k <= hi * (1 + 1e-6)}
                bad_vals += own != {e["range_split"]}
    lines.append(f"band names outside their split: {len(bad_names)}; friction/stiffness values in another split's "
                 f"band (or in none): {bad_vals}")
    if bad_names or bad_vals:
        errs.append("band leakage")
    lines.append(f"LEAKAGE TEST {'PASS' if not errs else 'FAIL: ' + '; '.join(errs)}")
    return lines
