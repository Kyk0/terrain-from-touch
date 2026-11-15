"""Provenance of result files (integrity rule 5): the exact commit and whether the tree was clean."""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def git_state() -> dict:
    run = lambda *a: subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    # result files written by the pipeline itself (results/) do not count as a dirty tree; code/config changes do
    status = subprocess.run(["git", "-C", ROOT, "status", "--porcelain"], capture_output=True, text=True).stdout
    dirty = [ln for ln in status.splitlines() if ln.strip() and not ln[3:].startswith("results/")]
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(dirty), "dirty_files": dirty[:20]}


def require_clean(allow_dirty: bool) -> dict:
    """Return git_state(); exit if the tree is dirty and allow_dirty is False."""
    st = git_state()
    if st["dirty"] and not allow_dirty:
        sys.exit("refusing to write results from a dirty tree (commit first, or pass --allow-dirty): "
                 + ", ".join(st["dirty_files"]))
    return st
