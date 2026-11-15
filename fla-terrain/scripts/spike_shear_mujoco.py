"""T2 spike: shear out of MuJoCo.

A 5 kg box on a plane is dragged in +x at constant speed by a stiff velocity
actuator on an x-slide joint (z-slide free, no rotation). Contact forces come
from mj_contactForce (contact frame), rotated to world and summed per geom.
Variant "pads": the box bottom is 2 x 4 separate pad geoms; per-pad forces must
sum to the total. Check: |F_t| / F_n within 10 % of the set friction.

Usage: python scripts/spike_shear_mujoco.py [--cone elliptic|pyramidal] [--all]
"""
from __future__ import annotations

import argparse
import sys

import mujoco
import numpy as np

XML = """
<mujoco>
  <option timestep="0.002" integrator="implicitfast" cone="{cone}" impratio="{impratio}" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" friction="{mu} 0.005 0.0001" condim="3"/>
    <body name="box" pos="0 0 {z0}">
      <joint name="x" type="slide" axis="1 0 0" damping="0"/>
      <joint name="z" type="slide" axis="0 0 1" damping="0"/>
      <inertial pos="0 0 0" mass="5" diaginertia="0.01 0.01 0.01"/>
      {geoms}
    </body>
  </worldbody>
  <actuator>
    <velocity name="drag" joint="x" kv="5000" forcerange="-1000 1000" forcelimited="true"/>
  </actuator>
</mujoco>
"""


def box_geoms(variant: str, mu: float) -> str:
    if variant == "single":
        return f'<geom name="b" type="box" size="0.1 0.05 0.02" friction="{mu} 0.005 0.0001" condim="3" mass="0"/>'
    g = []
    hx, hy = 0.1 / 4, 0.05 / 2  # 4 along x, 2 across y
    for i in range(4):
        for j in range(2):
            x = -0.1 + hx * (2 * i + 1)
            y = -0.05 + hy * (2 * j + 1)
            g.append(f'<geom name="pad{i}{j}" type="box" pos="{x} {y} 0" size="{hx*0.98} {hy*0.98} 0.02" '
                     f'friction="{mu} 0.005 0.0001" condim="3" mass="0"/>')
    return "\n      ".join(g)


def geom_forces_world(m, d):
    """{geom_id: world force on that geom (from the other geom)} summed over contacts."""
    out = {}
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        mujoco.mj_contactForce(m, d, i, f6)
        R = c.frame.reshape(3, 3)  # rows: normal, tangent1, tangent2 (world)
        fw = R.T @ f6[:3]          # force of geom1 on geom2, in world
        # contact normal points from geom1 to geom2; f6 is the force geom1 exerts on geom2
        out[c.geom2] = out.get(c.geom2, 0) + fw
        out[c.geom1] = out.get(c.geom1, 0) - fw
    return out


def run(mu: float, variant: str, cone: str, impratio: float, speed: float = 0.2, T: float = 3.0):
    xml = XML.format(cone=cone, impratio=impratio, mu=mu, z0=0.02, geoms=box_geoms(variant, mu))
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    box_geom_ids = [i for i in range(m.ngeom) if i != floor]
    F, ratios, sum_err, vx, n_act = [], [], [], [], []
    for k in range(int(T / dt)):
        d.ctrl[0] = speed
        mujoco.mj_step(m, d)
        if k * dt < 1.0:  # settle + reach steady sliding
            continue
        fg = geom_forces_world(m, d)
        per = np.array([fg.get(g, np.zeros(3)) for g in box_geom_ids])  # force of floor on each box geom (world)
        tot = per.sum(0)
        F.append(tot)
        vx.append(d.qvel[0])
        if tot[2] > 1e-6:
            ratios.append(np.hypot(tot[0], tot[1]) / tot[2])
            floor_tot = -fg.get(floor, np.zeros(3))  # independent total: reaction on the floor
            sum_err.append(np.linalg.norm(tot - floor_tot) / np.linalg.norm(floor_tot))
            n_act.append(int(np.sum(np.linalg.norm(per, axis=1) > 1e-6)))
    F = np.array(F)
    mg = float(m.body_subtreemass[1] * 9.81)
    # 150 Hz sensor view: average physics steps over 1/150 s bins
    nb = int(round(1 / 150 / dt * 3))  # 3 bins of 1/150 s = 10 physics steps at 500 Hz
    Fs = F[: len(F) // nb * nb].reshape(-1, nb, 3).mean(1)  # 50 Hz averages (10 steps), used for the dropout stat
    r_frames = float(np.mean(ratios))
    r_avg = float(np.hypot(F[:, 0].mean(), F[:, 1].mean()) / F[:, 2].mean())
    return {
        "ratio": r_frames, "ratio_avg": r_avg, "rel_err": max(abs(r_frames - mu), abs(r_avg - mu)) / mu,
        "Fn_over_mg": float(F[:, 2].mean() / mg), "Fn_cv": float(F[:, 2].std() / F[:, 2].mean()),
        "zero_frac": float(np.mean(F[:, 2] <= 1e-6)), "zero_frac_50hz": float(np.mean(Fs[:, 2] <= 1e-6)),
        "Fn_cv_50hz": float(Fs[:, 2].std() / Fs[:, 2].mean()),
        "vx": float(np.mean(vx)), "sum_err": float(np.max(sum_err)), "n_active_geoms": int(np.max(n_act)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cone", default="elliptic")
    ap.add_argument("--impratio", type=float, default=1.0)
    ap.add_argument("--speed", type=float, default=0.2, help="drag speed for the PASS/FAIL check")
    ap.add_argument("--sweep", default="0.02,0.05,0.5", help="extra speeds, reported as INFO (no PASS/FAIL)")
    ap.add_argument("--all", action="store_true", help="also run pyramidal cone and impratio 10 for comparison")
    args = ap.parse_args()
    configs = [(args.cone, args.impratio)]
    if args.all:
        configs += [(c, i) for c in ("pyramidal", "elliptic") for i in (1.0, 10.0) if (c, i) != configs[0]]
    ok_all = True
    print(f"mujoco {mujoco.__version__}; friction combine = max (both geoms = mu), condim=3, dt=0.002, implicitfast")
    for cone, imp in configs:
        for speed in [args.speed] + [float(x) for x in args.sweep.split(",") if x]:
            info = speed != args.speed
            for variant in ("single", "pads"):
                for mu in (0.3, 0.6, 0.9):
                    r = run(mu, variant, cone, imp, speed)
                    # validity: loaded by its weight on average, sliding at the commanded speed
                    valid = abs(r["Fn_over_mg"] - 1) < 0.02 and abs(r["vx"] / speed - 1) < 0.05
                    ok = valid and r["rel_err"] <= 0.10
                    if variant == "pads":
                        ok = ok and r["sum_err"] < 1e-6 and r["n_active_geoms"] == 8
                    if (cone, imp) == configs[0] and not info:
                        ok_all &= ok
                    tag = ("info-ok " if ok else "info-bad") if info else ("PASS" if ok else "FAIL")
                    print(f"{tag} {cone[:4]} imp={imp:<4g} v={speed:<4g} {variant:6s} mu={mu:.1f} "
                          f"Ft/Fn={r['ratio']:.4f} (avg {r['ratio_avg']:.4f}) err={100*r['rel_err']:.1f}% "
                          f"Fn/mg={r['Fn_over_mg']:.3f} vx={r['vx']:.3f} sumerr={r['sum_err']:.0e} pads={r['n_active_geoms']} "
                          f"| hop: Fn=0 {100*r['zero_frac']:.0f}% cv {r['Fn_cv']:.2f}; 50Hz cv {r['Fn_cv_50hz']:.2f}")
    print(f"OVERALL {'PASS' if ok_all else 'FAIL'} (cone={configs[0][0]}, impratio={configs[0][1]}, v={args.speed} m/s)")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
