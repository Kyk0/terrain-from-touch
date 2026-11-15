#!/usr/bin/env python
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Import a FLA-0 v2 k1_gym checkpoint (adder) into the MuJoCo motion source.
  python scripts/walk_k1v2_import.py --remote <adder path to model_N.pth> --name K1V2_r2_m1000
Copies the .pth to b_runs/k1v2/ckpt/, writes b_runs/k1v2/policies/<name>.npz (actor ELU MLP 61-256-128-128-16,
same layout as walking/checkpoints) and a TorchScript actor <name>.pt; checks numpy == torch on random inputs."""
import argparse, os, subprocess
import numpy as np
import torch

B = "/path/to/fla-terrain/b_runs/k1v2"
ap = argparse.ArgumentParser(); ap.add_argument("--remote", required=True); ap.add_argument("--name", required=True)
a = ap.parse_args()
os.makedirs(f"{B}/ckpt", exist_ok=True); os.makedirs(f"{B}/policies", exist_ok=True)
local = f"{B}/ckpt/{a.name}.pth"
subprocess.run(["scp", "-q", f"adder.cs.kent.edu:{a.remote}", local], check=True)
sd = torch.load(local, map_location="cpu", weights_only=True)["model"]
act = {k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")}
np.savez(f"{B}/policies/{a.name}.npz", **{k.replace(".", "_"): v.numpy() for k, v in act.items()})
net = torch.nn.Sequential(torch.nn.Linear(61, 256), torch.nn.ELU(), torch.nn.Linear(256, 128), torch.nn.ELU(),
                          torch.nn.Linear(128, 128), torch.nn.ELU(), torch.nn.Linear(128, 16))
net.load_state_dict(act)
torch.jit.script(net.eval()).save(f"{B}/policies/{a.name}.pt")
from walking.k1_velocity import MLPActor  # noqa: E402
x = np.random.default_rng(0).normal(size=(5, 61))
d = max(np.abs(MLPActor(f"{B}/policies/{a.name}.npz")(xi) - net(torch.tensor(xi, dtype=torch.float32)).detach().numpy()).max() for xi in x)
print(f"[import] {a.name}: {local} -> policies/{a.name}.npz/.pt, numpy vs torch max diff {d:.1e}")
