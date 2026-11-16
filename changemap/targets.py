"""Targets: what the denoiser generates for a Window, behind one small interface.

A target says how many channels each Token Grid square holds, how a Window becomes a training tensor (28, 50, C) in
[-1, 1], how a model output becomes something scorable, and which floors it is scored against. Train, sample and
score take the target by name (`--target`), record it (config.json, the samples file, report.json) and refuse a run
or samples file made for a different target.

Targets today:
  grid_map   the Grid Map (#2): 1 channel, coverage of Significant Regions per square, scorable as (28, 50) in [0, 1].
"""
from __future__ import annotations

import numpy as np

from .data import GRID, from_model_range, mean_grid_map, to_model_range

DEFAULT = "grid_map"


class GridMapTarget:
    name = "grid_map"
    channels = 1

    @property
    def shape(self) -> tuple:
        """One Window's training tensor shape: (28, 50, channels)."""
        return (*GRID, self.channels)

    def tensor(self, window) -> np.ndarray:
        """(28, 50, 1) in [-1, 1]."""
        return to_model_range(window.grid_map)

    def truth(self, window) -> np.ndarray:
        """The scorable true value: (28, 50) in [0, 1]."""
        return window.grid_map

    def decode(self, x) -> np.ndarray:
        """Model output (..., 28, 50, 1) -> scorable (..., 28, 50) in [0, 1]."""
        return from_model_range(x)

    def floors(self, tier, windows) -> dict:
        """Floor name -> (n, 1, 28, 50) scorable maps, in report order."""
        n = len(windows)
        return {
            "all_black_floor": np.zeros((n, 1, *GRID), np.float32),
            "mean_map_floor": np.broadcast_to(mean_grid_map(tier), (n, 1, *GRID)).copy(),
            "previous_target_floor": np.stack([(w.previous_map if w.has_previous else np.zeros(GRID, np.float32))[None]
                                               for w in windows]).astype(np.float32),
        }


TARGETS = {t.name: t for t in (GridMapTarget(),)}


def get(name: str) -> GridMapTarget:
    if name not in TARGETS:
        raise ValueError(f"unknown target {name!r}: choose from {tuple(TARGETS)}")
    return TARGETS[name]


def require_same_target(expected: str, found: str | None, what) -> None:
    """Refuse a run or samples file made for another target. A missing record means a run made before targets
    existed, which can only be a Grid Map run."""
    found = found or DEFAULT
    if found != expected:
        raise ValueError(f"{what} was made for target {found!r}, not {expected!r} (pass --target {found})")
