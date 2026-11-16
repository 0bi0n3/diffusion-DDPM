"""Windows of a scaling tier: rows of `<split>_targets.jsonl` turned into Grid Map targets."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import siblings  # noqa: F401
from dit2vlm import hints  # noqa: E402

GRID = (28, 50)
REGION_CAP = 10
SPLITS = ("train", "val")
TEST = "test"


@dataclass(frozen=True)
class Window:
    id: str
    regions: tuple          # Significant Regions of the Last Step, 1920x1080 px boxes (50/20 rule, at most 10)
    grid_map: np.ndarray    # (28, 50) float32 coverage in [0, 1]; all black for a No Change Window

    @property
    def no_change(self) -> bool:
        return not self.regions


def significant_regions(row: dict) -> tuple:
    keep = [tuple(float(v) for v in b) for b, ok in zip(row["regions"], row["box_ok_50_20"]) if ok]
    return tuple(keep[:REGION_CAP])


def load_windows(tier, split: str, verdict: bool = False) -> list[Window]:
    """Windows of `split` in file order. Dropped Pairs are left out; a No Change Window (or a target row with no
    Significant Region) keeps an all-black Grid Map. The test split is refused unless `verdict` is True."""
    if split == TEST and not verdict:
        raise PermissionError("the test split is read only for a verdict: pass --verdict")
    if split not in SPLITS and split != TEST:
        raise ValueError(f"unknown split {split!r}")
    out = []
    for line in (Path(tier) / f"{split}_targets.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        status = row["target_status"]
        if status == "dropped":
            continue
        if status not in ("target", "no_change"):
            raise ValueError(f"{row['id']}: target_status {status!r} has no Grid Map")
        regions = significant_regions(row) if status == "target" else ()
        out.append(Window(row["id"], regions, hints.change_map(regions, GRID).numpy()))
    return out


def to_model_range(grid_map: np.ndarray) -> np.ndarray:
    """[0, 1] -> [-1, 1], with a trailing channel: (..., 28, 50, 1)."""
    return (np.asarray(grid_map, np.float32) * 2 - 1)[..., None]


def from_model_range(x) -> np.ndarray:
    """(..., 28, 50, 1) in about [-1, 1] -> (..., 28, 50) clipped to [0, 1]."""
    return np.clip((np.asarray(x)[..., 0] + 1) / 2, 0, 1)
