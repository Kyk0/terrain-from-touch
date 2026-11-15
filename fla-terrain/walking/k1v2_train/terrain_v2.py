"""FLA-0 v2 terrain for k1_gym (the independent evaluation). Copied into adder:/path/to/fla-terrain/k1v2/k1_gym/envs/.

Generated terrain (Isaac Lab TerrainGenerator), difficulty rows x type columns, curriculum on rows:
  flat (plane), hf_rough (random uniform heightfield, 0.5-2 cm), grid (random grid blocks 0.5-3 cm),
  slope up / down (pyramid, 0 -> 10 deg with difficulty), planks (stepping stones 30 -> 12 cm wide,
  2 -> 8 cm gaps, 15 cm drop), platforms up / down (pyramid stairs, 3 -> 6 cm steps, 0.4-0.8 m treads).
Height lookup: the generator is deterministic for a fixed seed, so the same mesh is regenerated here and
ray-cast once on the GPU into a height map (4 cm cells); terrain_heights() is a nearest-cell lookup.
"""
import math

import numpy as np
import torch

import isaaclab.terrains as T
from isaaclab.terrains import TerrainGenerator, TerrainGeneratorCfg


def generator_cfg(tcfg: dict) -> TerrainGeneratorCfg:
    tan10 = math.tan(math.radians(tcfg.get("max_slope_deg", 10.0)))
    p = tcfg.get("proportions", {})
    sub = {
        "flat": T.MeshPlaneTerrainCfg(proportion=p.get("flat", 0.10)),
        "hf_rough": T.HfRandomUniformTerrainCfg(proportion=p.get("hf_rough", 0.15), noise_range=(0.0, 0.02),
                                                noise_step=0.005, downsampled_scale=0.1),
        "grid": T.MeshRandomGridTerrainCfg(proportion=p.get("grid", 0.10), grid_width=0.15,
                                           grid_height_range=(0.005, 0.03), platform_width=2.0),
        "slope_up": T.HfPyramidSlopedTerrainCfg(proportion=p.get("slope_up", 0.10), slope_range=(0.0, tan10),
                                                platform_width=2.0),
        "slope_down": T.HfInvertedPyramidSlopedTerrainCfg(proportion=p.get("slope_down", 0.10),
                                                          slope_range=(0.0, tan10), platform_width=2.0),
        "planks": T.HfSteppingStonesTerrainCfg(proportion=p.get("planks", 0.15), stone_height_max=0.0,
                                               stone_width_range=(0.30, 0.12), stone_distance_range=(0.02, 0.08),
                                               holes_depth=-0.15, platform_width=2.0),
        "platform_up": T.MeshPyramidStairsTerrainCfg(proportion=p.get("platform_up", 0.15),
                                                     step_height_range=(0.03, 0.06), step_width=0.6,
                                                     platform_width=2.0, border_width=1.0),
        "platform_down": T.MeshInvertedPyramidStairsTerrainCfg(proportion=p.get("platform_down", 0.15),
                                                               step_height_range=(0.03, 0.06), step_width=0.6,
                                                               platform_width=2.0, border_width=1.0),
    }
    if p.get("gaps", 0.0) > 0:   # v2_run5: a 3-10 cm gap around a 1.5 m platform (plank-gap crossing)
        sub["gaps"] = T.MeshGapTerrainCfg(proportion=p["gaps"], gap_width_range=(0.03, 0.10), platform_width=1.5)
    return TerrainGeneratorCfg(
        seed=int(tcfg.get("seed", 42)),
        size=(float(tcfg.get("tile_size", 8.0)), float(tcfg.get("tile_size", 8.0))),
        border_width=float(tcfg.get("border_width", 25.0)),
        num_rows=int(tcfg.get("num_rows", 10)),
        num_cols=int(tcfg.get("num_cols", 16)),
        horizontal_scale=0.05,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=True,
        sub_terrains=sub,
    )


class HeightMap:
    def __init__(self, gen_cfg: TerrainGeneratorCfg, device, res=0.04):
        from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

        mesh = TerrainGenerator(cfg=gen_cfg, device=device).terrain_mesh
        lo, hi = mesh.bounds[0], mesh.bounds[1]
        self.x0, self.y0, self.res = float(lo[0]), float(lo[1]), res
        nx, ny = int(np.ceil((hi[0] - lo[0]) / res)) + 1, int(np.ceil((hi[1] - lo[1]) / res)) + 1
        wm = convert_to_warp_mesh(mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32), device=device)
        xs = torch.arange(nx, device=device) * res + self.x0
        ys = torch.arange(ny, device=device) * res + self.y0
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        hmap = torch.empty(nx, ny, device=device)
        z_top = float(hi[2]) + 1.0
        chunk = 1 << 20
        flat_x, flat_y = gx.reshape(-1), gy.reshape(-1)
        out = hmap.view(-1)
        for i in range(0, flat_x.numel(), chunk):
            n = min(chunk, flat_x.numel() - i)
            starts = torch.stack([flat_x[i:i + n], flat_y[i:i + n], torch.full((n,), z_top, device=device)], -1)
            dirs = torch.zeros_like(starts)
            dirs[:, 2] = -1.0
            hits = raycast_mesh(starts.contiguous(), dirs.contiguous(), wm, max_dist=1e3)[0]
            z = hits[:, 2]
            out[i:i + n] = torch.where(torch.isfinite(z), z, torch.full_like(z, float(lo[2])))
        self.h = hmap
        self.nx, self.ny = nx, ny
        print(f"[terrain_v2] height map {nx} x {ny} cells @ {res} m, z {float(hmap.min()):.3f}..{float(hmap.max()):.3f}")

    def __call__(self, positions):
        ix = ((positions[..., 0] - self.x0) / self.res).round().long().clamp(0, self.nx - 1)
        iy = ((positions[..., 1] - self.y0) / self.res).round().long().clamp(0, self.ny - 1)
        return self.h[ix, iy]
