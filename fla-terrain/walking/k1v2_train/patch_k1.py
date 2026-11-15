"""Patch the k1v2 copy of k1_gym/envs/k1.py for generated terrain (run once on adder; idempotent)."""
import sys

p = sys.argv[1]
s = open(p).read()
if "terrain_v2" in s:
    print("already patched")
    sys.exit(0)


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:60], s.count(old))
    s = s.replace(old, new)


rep('''        if self.cfg["terrain"]["type"] != "plane":
            raise NotImplementedError(
                f"only the plane terrain is ported; got terrain.type={self.cfg['terrain']['type']!r}"
            )
''', '''        tcfg = self.cfg["terrain"]
        self.height_map = None
        if tcfg["type"] == "generator":  # FLA-0 v2 (terrain_v2.py)
            from .terrain_v2 import generator_cfg
            self._gen_cfg = generator_cfg(tcfg)
            scene_cfg.terrain = TerrainImporterCfg(
                prim_path="/World/ground", terrain_type="generator", terrain_generator=self._gen_cfg,
                max_init_terrain_level=int(tcfg.get("max_init_level", 3)), collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=tcfg["static_friction"], dynamic_friction=tcfg["dynamic_friction"],
                    restitution=tcfg["restitution"],
                    friction_combine_mode=tcfg.get("friction_combine_mode", "average"),
                    restitution_combine_mode="multiply"),
            )
        elif tcfg["type"] != "plane":
            raise NotImplementedError(f"terrain.type={tcfg['type']!r}")
''')
rep('''        self.env_origins = self.scene.env_origins
''', '''        self.env_origins = self.scene.env_origins
        if self.cfg["terrain"]["type"] == "generator":
            from .terrain_v2 import HeightMap
            self.height_map = HeightMap(self._gen_cfg, self.device)
            self.terrain = self.scene.terrain
            self._n_terrain_updates = 0
''')
rep('''        """Ground height under ``positions``. The plane terrain sits at z = 0."""
        return torch.zeros(positions.shape[:-1], dtype=torch.float, device=self.device)''',
    '''        """Ground height under ``positions`` (height map for generated terrain, 0 for the plane)."""
        if getattr(self, "height_map", None) is not None:
            return self.height_map(positions)
        return torch.zeros(positions.shape[:-1], dtype=torch.float, device=self.device)''')
rep('''        root_pose[:, 2] += self.terrain_heights(root_pose[:, :2])''',
    '''        # spawn height from the ground under the (randomized) xy; env origin z is the tile centre height,
        # which must not be added on top of it
        root_pose[:, 2] = self.base_init_state[2] + self.terrain_heights(root_pose[:, :2])
        if getattr(self, "height_map", None) is not None:
            if not hasattr(self, "_spawn_xy"):
                self._spawn_xy = self.env_origins[:, :2].clone()
            self._spawn_xy[env_ids] = root_pose[:, :2]''')
rep('''        self._obs_hist_reset_ids = env_ids
        self._update_curriculum(env_ids)''', '''        self._obs_hist_reset_ids = env_ids
        self._update_terrain_curriculum(env_ids)
        self._update_curriculum(env_ids)''')
rep('''    def _reset_dofs(self, env_ids):''', '''    def _update_terrain_curriculum(self, env_ids):
        """Terrain curriculum (v2_run4): path length walked vs commanded distance, both integrated every step
        (see _check_termination). Promote if the episode ran to its time limit without a fall and the robot
        walked >= 2 m and >= promote_frac (default 0.6) of the commanded distance; demote after a fall or < 30 % of it. Skipped at
        episode step 0 (first reset)."""
        if getattr(self, "height_map", None) is None or not hasattr(self, "episode_length_buf"):
            return
        if not hasattr(self, "_walked"):
            self._walked = torch.zeros(self.num_envs, device=self.device)
            self._cmd_dist = torch.zeros(self.num_envs, device=self.device)
        max_len = int(torch.ceil(torch.tensor(self.cfg["rewards"]["episode_length_s"] / self.dt)).item())
        started = self.episode_length_buf[env_ids] > 0
        fell = self.episode_length_buf[env_ids] <= max_len
        walked, asked = self._walked[env_ids], self._cmd_dist[env_ids]
        promote = float(self.cfg["terrain"].get("promote_frac", 0.6))
        move_up = started & ~fell & (walked > 2.0) & (walked > promote * asked)
        move_down = started & ~move_up & (fell | (walked < 0.3 * asked))
        self.terrain.update_env_origins(env_ids, move_up, move_down)
        self._walked[env_ids] = 0.0
        self._cmd_dist[env_ids] = 0.0
        self._n_terrain_updates += 1
        if self._n_terrain_updates % 500 == 0:
            lv = self.terrain.terrain_levels.float()
            print(f"[terrain] resets {self._n_terrain_updates}: level mean {float(lv.mean()):.2f} "
                  f"max {int(lv.max())} >=5: {float((lv >= 5).float().mean()):.2f}, up "
                  f"{float(move_up.float().mean()):.2f} down {float(move_down.float().mean()):.2f}", flush=True)

    def _reset_dofs(self, env_ids):''')
rep('''    def _check_termination(self):''', '''    def _check_termination(self):
        if getattr(self, "height_map", None) is not None:  # v2 terrain curriculum bookkeeping
            if not hasattr(self, "_walked"):
                self._walked = torch.zeros(self.num_envs, device=self.device)
                self._cmd_dist = torch.zeros(self.num_envs, device=self.device)
            self._walked += torch.norm(self.root_vel_w[:, :2], dim=1) * self.dt
            self._cmd_dist += torch.norm(self.commands[:, :2], dim=1) * self.dt''')
open(p, "w").write(s)
print("patched", p)
