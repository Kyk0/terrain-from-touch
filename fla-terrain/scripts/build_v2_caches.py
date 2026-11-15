"""Build the v2 training caches (FrameDataV2) for the main, sensitivity and course datasets."""
import sys
import time

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from fla_terrain.models.dataset_v2 import FrameDataV2, cache_path_v2  # noqa: E402

D = "/path/to/fla-terrain/data"
for d in ("v2", "v2_sens/nominal", "v2_sens/shifted", "v2_sens/g1", "v2_sens/pads", "v2_sens/nosat", "v2_course"):
    t = time.time()
    fd = FrameDataV2.build(f"{D}/{d}", cache=cache_path_v2(f"{D}/{d}"), workers=32)
    print(f"{d}: {len(fd.eps)} episodes, {len(fd.V)} frames, {len(fd.starts)} windows, PTS {fd.PTS.nbytes / 1e9:.2f} GB, "
          f"{time.time() - t:.0f} s", flush=True)
print("CACHES_DONE", flush=True)
