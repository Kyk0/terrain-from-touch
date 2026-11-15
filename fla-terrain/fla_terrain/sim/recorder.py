"""Episode recorder: call ``step(data, cmd_vel)`` after every ``mj_step``; ``write`` saves a schema v0.2 episode.

Frame i of the 50 Hz output covers physics time ((i)/50, (i+1)/50] after ``reset``; its
timestamp is t_i = (i+1)/50. Zone forces are the mean of the three 150 Hz sensor samples in
the frame; proprioception, IMU, labels and support fraction are sampled at the end of the
frame; ``contact_gt`` is True if the frame-mean true (noise-free) total Fz of the foot exceeds 2 N
(robust to MuJoCo's slip chatter, see NOTES_A T2; a per-step contact-count rule was not).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from ..data.writer import write_episode
from ..schema import SCHEMA_VERSION, SCHEMA_VERSION_V3
from .sensor_model import SensorModel, SensorModelConfig
from .sole import SoleSensor
from .support import SupportProbe


@dataclass
class ProprioNoise:
    q: float = 0.002      # rad
    qd: float = 0.05      # rad/s
    gyro: float = 0.01    # rad/s
    grav: float = 0.005   # unit vector components


UNKNOWN = {"terrain": -1, "mu": np.nan, "stiffness": np.nan, "slope": np.nan}


class RegionLabeler:
    """Frame labels from the terrain region(s) the feet touch (contacts in the frame's last physics step).

    One region touched by all feet in contact -> that region's label/properties. Two regions under the
    feet -> -1 and NaN properties. No foot in contact -> region of the terrain directly below the base
    (ray cast); if none is hit -> -1. Single-region terrains always give that region.
    """

    def __init__(self, model, info):
        self.m, self.info = model, info
        self.regions = info["terrain_regions"]
        self.g2r = info["terrain_geom_region"]
        self.group = np.zeros(6, np.uint8)
        from .build import TERRAIN_GROUP
        self.group[TERRAIN_GROUP] = 1
        self._gid = np.zeros(1, np.int32)

    def region_lab(self, r):
        reg = self.regions[r]
        return {"terrain": int(reg["label"]), "mu": float(reg["mu"]), "stiffness": float(reg.get("stiffness", np.nan)),
                "slope": float(reg.get("slope", 0.0))}

    def __call__(self, model, data, touched=None):
        if len(self.regions) == 1:
            return self.region_lab(0)
        rs = set()
        for s in "LR":
            rs.update(self.g2r[g] for g in (touched or {}).get(s, ()) if g in self.g2r)
        if len(rs) == 1:
            return self.region_lab(rs.pop())
        if len(rs) > 1:
            return dict(UNKNOWN)
        b = self.info["base_body"]
        mujoco.mj_ray(model, data, data.xpos[b], np.array([0.0, 0.0, -1.0]), self.group, True, -1, self._gid)
        return self.region_lab(self.g2r[int(self._gid[0])]) if int(self._gid[0]) in self.g2r else dict(UNKNOWN)


class EpisodeRecorder:
    CONTACT_N = 2.0  # contact_gt: frame-mean TRUE (noise-free) total Fz of the foot above this [N]
    def __init__(self, model, info, *, motion_source: str, mode: str, split_key: str, rng=None,
                 sensor_cfg: SensorModelConfig | None = None, proprio_noise: ProprioNoise | None = None,
                 label_fn=None, rate_hz: float = 50.0, terrain_params: dict | None = None, proprio_fn=None,
                 target_fn=None):
        """proprio_fn(data) -> (q[J], qd[J], gyro[3], grav[3]) replaces the K1 readout (foot-only rig).
        target_fn(data) -> PD position target [J] (joint_pos order) of the motion source; with it, the v0.3 optional
        arrays joint_target and joint_torque (actuator force, joint order) are recorded."""
        self.m, self.info = model, info
        self.rng = rng if rng is not None else np.random.default_rng()
        self.sole = SoleSensor(model, info["sole_cfg"])
        self.Z = self.sole.Z
        self.scfg = sensor_cfg or SensorModelConfig()
        self.sensor = SensorModel(self.scfg, 2 * self.Z, self.rng)
        # sole v2: per-point map (reconstruction from the foot wrench, or pads read directly) + point sensor model
        self.layout = info.get("point_layout")
        self.v2 = self.layout is not None
        self.keep_true = False   # set True to keep the noise-free bin means (checks)
        if self.v2:
            from .pointmap import PointMapReconstructor, PointSensorModel
            self.prm = info["sole_v2_params"]
            self.pads = info["sole_cfg"].version == "v2pads"
            self.recon = None if self.pads else PointMapReconstructor(model, info, self.layout, self.prm)
            self.psensor = PointSensorModel(self.layout, self.prm, self.rng)
            self.Zout = self.layout.P
            from .pointmap import PointLayout
            vb = self.prm.view_boundaries
            self.table_layout = self.layout if not vb else PointLayout(
                length=self.layout.length, width=self.layout.width, pitch=self.layout.pitch, boundaries=tuple(vb),
                mask=self.layout.mask)
        else:
            self.Zout = self.Z
        self.pn = proprio_noise if proprio_noise is not None else ProprioNoise()
        self.probe = SupportProbe(model, info)
        self.label_fn = label_fn or RegionLabeler(model, info)
        self.rate = rate_hz
        self.native = self.scfg.native_rate_hz
        self.sub = int(round(self.native / rate_hz))
        assert abs(self.sub * rate_hz - self.native) < 1e-9, "native rate must be an integer multiple of rate_hz"
        self.dt = model.opt.timestep
        self.meta_base = {"motion_source": motion_source, "mode": mode, "split_key": split_key,
                          "terrain_params": terrain_params if terrain_params is not None else dict(info["terrain_cfg"])}
        self.proprio_fn = proprio_fn
        self.target_fn = target_fn
        if target_fn is not None:
            aj = list(info["actuator_joints"])
            self.act_of_joint = np.array([aj.index(j) for j in info["joint_names"]])
        # bump geoms: small terrain boxes (rough bumps are 2-5 cm half size; planks, steps and slabs are larger)
        tg = info.get("terrain_geoms", [])
        self.bump_geoms = {g for g in tg if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
                           and model.geom_size[g][0] < 0.06 and model.geom_size[g][1] < 0.06}
        self.g2r = info.get("terrain_geom_region", {})
        self.fall_time = None
        self.unknown_intervals = []  # [(t_start, t_end)] episode times where label_terrain is set to -1
        self.t0 = None

    def reset(self, data):
        self.t0 = data.time
        self.bin = 0
        self.acc = np.zeros((2, self.Zout, 3)) if self.v2 else np.zeros((2 * self.Z, 3))
        self.true_samples = []
        self.nacc = 0
        self.samples = []         # 150 Hz samples within the current frame
        self.c_steps = np.zeros(2)
        self.n_steps = 0
        self.frame_regions = {"L": set(), "R": set()}
        self.frame_bump = {"L": False, "R": False}
        self.prev_contact = np.zeros(2, bool)
        self.touchdowns = []      # (t, side, [regions touched in the touchdown frame])
        self.frames = {k: [] for k in ("F", "q", "qd", "gyro", "grav", "cmd", "terrain", "mu", "stiff", "slope",
                                       "sfrac", "contact", "ground", "jt", "tau")}

    def mark_fall(self, data):
        if self.fall_time is None:
            self.fall_time = float(data.time - self.t0)

    def step(self, data, cmd_vel=(0.0, 0.0, 0.0)):
        if self.t0 is None:
            self.reset(data)
        r = self.sole.read_full(data)
        self._touched = r["geoms"]
        for sd in "LR":
            self.frame_regions[sd].update(self.g2r.get(g, -1) for g in r["geoms"][sd])
            if self.bump_geoms and not self.frame_bump[sd]:
                self.frame_bump[sd] = bool(self.bump_geoms & r["geoms"][sd])
        if self.v2:
            maps = r["F"] if self.pads else self.recon.step(data, r["wrench"])
            self.acc[0] += maps["L"]
            self.acc[1] += maps["R"]
        else:
            self.acc += np.concatenate([r["F"]["L"], r["F"]["R"]])
        self.nacc += 1
        self.c_steps += [r["F"]["L"][:, 2].sum(), r["F"]["R"][:, 2].sum()]  # true load, summed over steps
        self.n_steps += 1
        t = data.time - self.t0
        if t + self.dt / 2 >= (self.bin + 1) / self.native - 1e-12:  # next step's midpoint is in the next bin
            if self.v2:
                true = self.acc / self.nacc
                if self.keep_true:
                    self.true_samples.append((data.time - self.t0, true.copy()))
                self.samples.append(np.stack([self.psensor.apply(true[f], f) for f in (0, 1)]))
            else:
                self.samples.append(self.sensor.apply(self.acc / self.nacc))
            self.acc[:] = 0
            self.nacc = 0
            self.bin += 1
            if len(self.samples) == self.sub:
                self._frame(data, np.mean(self.samples, 0), cmd_vel)
                self.samples = []

    def _frame(self, data, F, cmd_vel):
        m, info, pn, g = self.m, self.info, self.pn, self.rng
        fr = self.frames
        fr["F"].append(F)
        if self.proprio_fn is None:
            q = data.qpos[info["qpos_adr"]]
            qd = data.qvel[info["qvel_adr"]]
            R = data.xmat[info["base_body"]].reshape(3, 3)
            gyro = data.qvel[3:6].copy()  # free joint angular velocity is in the base (trunk) frame
            grav = R.T @ np.array([0.0, 0.0, -1.0])
        else:
            q, qd, gyro, grav = self.proprio_fn(data)
        fr["q"].append(q + pn.q * g.standard_normal(q.shape))
        fr["qd"].append(qd + pn.qd * g.standard_normal(qd.shape))
        fr["gyro"].append(gyro + pn.gyro * g.standard_normal(3))
        fr["grav"].append(grav + pn.grav * g.standard_normal(3))
        fr["cmd"].append(np.asarray(cmd_vel, float))
        lab = self.label_fn(m, data, self._touched)
        fr["terrain"].append(lab["terrain"])
        fr["mu"].append(lab["mu"])
        fr["stiff"].append(lab["stiffness"])
        fr["slope"].append(lab["slope"])
        contact = self.c_steps / max(self.n_steps, 1) > self.CONTACT_N
        fr["contact"].append(contact)
        t_frame = len(fr["F"]) / self.rate
        for i, sd in enumerate("LR"):
            if contact[i] and not self.prev_contact[i]:
                self.touchdowns.append((round(t_frame, 3), sd, sorted(int(r) for r in self.frame_regions[sd])))
            self.frame_regions[sd] = set()
        self.prev_contact = contact.copy()
        sfr = [self.probe.frac(data, s) for s in "LR"]
        fr["sfrac"].append(sfr)
        gl = []
        for i, sd in enumerate("LR"):
            if contact[i]:
                gl.append(4 if self.probe.label(True, sfr[i]) == 1 else (3 if self.frame_bump[sd] else 2))
            else:
                gl.append(1 if (sfr[i] < 0.5 or self.probe.center_drop(data, sd) > 0.20) else 0)
            self.frame_bump[sd] = False
        fr["ground"].append(gl)
        if self.target_fn is not None:
            fr["jt"].append(np.asarray(self.target_fn(data), float))
            fr["tau"].append(data.actuator_force[self.act_of_joint].copy())
        self.c_steps[:] = 0
        self.n_steps = 0

    def arrays(self) -> dict:
        fr = self.frames
        T = len(fr["F"])
        F = np.array(fr["F"]).reshape(T, 2, self.Zout, 3)
        contact = np.array(fr["contact"], bool).reshape(T, 2)
        sfrac = np.array(fr["sfrac"]).reshape(T, 2)
        t = (np.arange(T) + 1) / self.rate
        terrain = np.array(fr["terrain"]).reshape(T)
        for a, b in self.unknown_intervals:
            terrain[(t > a) & (t <= b)] = -1
        if self.fall_time is not None:
            terrain[t > self.fall_time] = -1
        sup = np.array([[self.probe.label(c, f) for c, f in zip(cc, ff)] for cc, ff in zip(contact, sfrac)]).reshape(T, 2)
        out = {
            "t": t,
            ("foot_points_L" if self.v2 else "foot_zone_L"): F[:, 0], ("foot_points_R" if self.v2 else "foot_zone_R"): F[:, 1],
            "foot_valid_L": np.ones(T, bool), "foot_valid_R": np.ones(T, bool),
            "joint_pos": np.array(fr["q"]).reshape(T, -1), "joint_vel": np.array(fr["qd"]).reshape(T, -1),
            "imu_gyro": np.array(fr["gyro"]).reshape(T, 3), "imu_gravity": np.array(fr["grav"]).reshape(T, 3),
            "cmd_vel": np.array(fr["cmd"]).reshape(T, 3),
            "label_terrain": terrain, "label_support_L": sup[:, 0], "label_support_R": sup[:, 1],
            "prop_mu": np.array(fr["mu"]), "prop_stiffness": np.array(fr["stiff"]), "prop_slope": np.array(fr["slope"]),
            "prop_support_frac_L": sfrac[:, 0], "prop_support_frac_R": sfrac[:, 1],
            "contact_gt_L": contact[:, 0], "contact_gt_R": contact[:, 1],
        }
        if self.v2:
            g = np.array(fr["ground"], np.int8).reshape(T, 2)
            out["label_ground_L"], out["label_ground_R"] = g[:, 0], g[:, 1]
            if self.target_fn is not None:
                out["joint_target"] = np.array(fr["jt"]).reshape(T, -1)
                out["joint_torque"] = np.array(fr["tau"]).reshape(T, -1)
        return out

    def meta(self, notes: str = "") -> dict:
        sm = self.scfg.to_dict()
        if self.v2:
            sm = {"name": "fla_sole_v2_points", "map": "pads (each point its own rigid pad)" if self.pads else
                  "reconstruction: elastic layer from the foot wrench + per-point gaps (pointmap.py)",
                  "params": self.prm.to_dict(), "native_rate_hz": self.native,
                  "approximations": "per-point shear shared in proportion to pressure (+ torsion); blocks derived "
                                    "from the point map"}
        sm["proprio_noise"] = asdict(self.pn)
        sm["support_probe"] = self.probe.params
        sm["physics_dt"] = self.dt
        return {
            "schema_version": SCHEMA_VERSION_V3 if self.v2 else SCHEMA_VERSION, "source": "sim",
            "robot": "booster_k1_22dof_mujoco", "sensor_model": sm,
            "zone_layout": ({**self.table_layout.table(), "sole": self.info["sole_cfg"].layout()} if self.v2
                            else self.info["sole_cfg"].layout()), "rate_hz": self.rate,
            "sensor_native_rate_hz": self.native, "notes": notes, "fall_time": self.fall_time, "fell": self.fall_time is not None,
            "unknown_intervals": [list(map(float, iv)) for iv in self.unknown_intervals],
            "touchdowns": self.touchdowns,
            "joint_names": list(self.info["joint_names"]), **self.meta_base,
        }

    def write(self, out_dir, episode_id, notes: str = ""):
        return write_episode(out_dir, episode_id, self.arrays(), self.meta(notes))
