"""The run record `run_record.json`: code commit, config, data sha256s, device, wall time, peak memory.

Only the train and val files of a tier are hashed (names starting `train` or `val`): the test split is never read.
The code commit is `CHANGEMAP_COMMIT` if set (a `git archive` export on the cluster), else this checkout's HEAD.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from . import siblings

REPO = Path(__file__).resolve().parents[1]


def _git(root, *args) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def code_commit() -> str:
    if os.environ.get("CHANGEMAP_COMMIT"):
        return os.environ["CHANGEMAP_COMMIT"]
    return _git(REPO, "rev-parse", "HEAD")


def sibling_commits() -> dict:
    out = {}
    for name, root in siblings.roots().items():
        try:
            out[name] = os.environ.get(f"{name.upper()}_COMMIT") or _git(root, "rev-parse", "HEAD")
        except (subprocess.CalledProcessError, OSError):
            out[name] = "unknown"
    return out


def data_sha256(tier) -> dict:
    out = {}
    for p in sorted(Path(tier).iterdir()):
        if p.is_file() and p.name.startswith(("train", "val")):
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            out[p.name] = h.hexdigest()
    return out


def device_name(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(device)
    return f"{device} ({platform.processor() or platform.machine()})"


def peak_memory(device: str) -> dict:
    mem = {"host_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    if device.startswith("cuda") and torch.cuda.is_available():
        mem["gpu_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 2 ** 20
    return mem


def make(tier, config: dict, status: str, step: int, wall_seconds: float, peak: dict, jobs: list) -> dict:
    return {"code_commit": code_commit(), "sibling_commits": sibling_commits(), "config": config,
            "data_sha256": data_sha256(tier), "device": device_name(config["device"]), "status": status,
            "step": step, "wall_time_seconds": wall_seconds, "peak_memory": peak, "jobs": jobs,
            "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def write(out, record: dict) -> Path:
    p = Path(out) / "run_record.json"
    p.write_text(json.dumps(record, indent=2) + "\n")
    return p
