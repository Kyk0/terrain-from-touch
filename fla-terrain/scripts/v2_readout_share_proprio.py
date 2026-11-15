"""Make extra sensor readouts a pure sole difference: copy the proprioception arrays (joint_pos, joint_vel, imu_gyro,
imu_gravity; the noisy streams drawn by each recorder's own RNG) from the nominal episode into the readout episode of
the same simulation. Every other array except foot_points_* must already be identical (checked).
  $PY scripts/v2_readout_share_proprio.py /path/to/fla-terrain/data/v2_sens/{nosat,shifted,g1}
"""
import glob
import os
import sys

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
import numpy as np  # noqa: E402

from fla_terrain.schema import validate_episode  # noqa: E402

KEYS = ("joint_pos", "joint_vel", "imu_gyro", "imu_gravity")


def fix(p):
    ref = os.path.join(os.path.dirname(os.path.dirname(p)), "nominal", os.path.basename(p))
    a, b = dict(np.load(p)), dict(np.load(ref))
    other = [k for k in a if k not in KEYS and not k.startswith("foot_points") and not np.array_equal(a[k], b[k], equal_nan=True)]
    if other:
        return p, f"other arrays differ: {other}"
    for k in KEYS:
        a[k] = b[k]
    tmp = p[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **a)
    os.replace(tmp, p)
    e = validate_episode(p, raise_on_error=False)
    return p, (f"invalid: {e[:1]}" if e else None)


if __name__ == "__main__":
    from multiprocessing import Pool
    for d in sys.argv[1:]:
        ps = sorted(glob.glob(f"{d}/episode_*.npz"))
        with Pool(24) as pool:
            res = pool.map(fix, ps, chunksize=8)
        bad = [r for r in res if r[1]]
        print(f"{d}: {len(ps)} episodes, proprio copied from nominal, problems {len(bad)}" + (f" e.g. {bad[0]}" if bad else ""))
