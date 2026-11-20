"""Windows of a scaling tier: rows of `<split>_targets.jsonl` turned into Grid Map targets.

Per-Window context (Conditioning Arms other than Unconditional) lives beside the targets, one file per split:
`<tier>/<split>_context.npz` with arrays aligned on the first axis (ticket #3 builds the real cache in this format)
  ids      (n,)             Window id strings; may include Dropped Pairs and extra Windows (looked up by id)
  f8       (n, 28, 50, 768) Previous Frame DINOv3 tokens, already normalised (float16 or float32)
  caption  (n, 77, 768)     Caption CLIP ViT-L/14 text tokens (float16 or float32)
The Unconditional arm never opens these files.
"""
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


def window_key(id_: str, source_window_id: str | None = None) -> str:
    """Unique row key: `<id>@<source_window_id>` for two-frame Step Pairs, else `<id>`."""
    return f"{id_}@{source_window_id}" if source_window_id else id_


@dataclass(frozen=True)
class Window:
    id: str
    regions: tuple          # Significant Regions of the Last Step, 1920x1080 px boxes (50/20 rule, at most 10)
    grid_map: np.ndarray    # (28, 50) float32 coverage in [0, 1]; all black for a No Change Window
    previous_map: np.ndarray | None = None   # Grid Map of the F7 -> F8 Step Pair; None when there is no previous pair
    previous_frame: str | None = None        # optional image path of the Previous Frame (F8), for panels only
    previous_regions: tuple | None = None    # Significant Regions of the F7 -> F8 Step Pair; None when there is no previous pair
    source_window_id: str | None = None      # parent Window id for two-frame Step Pairs (None for Last-Step tiers)

    @property
    def key(self) -> str:
        return window_key(self.id, self.source_window_id)

    @property
    def no_change(self) -> bool:
        return not self.regions

    @property
    def has_previous(self) -> bool:
        return self.previous_map is not None


def significant_regions(row: dict) -> tuple:
    keep = [tuple(float(v) for v in b) for b, ok in zip(row["regions"], row["box_ok_50_20"]) if ok]
    return tuple(keep[:REGION_CAP])


def previous_regions(prev) -> tuple | None:
    """Significant Regions of a previous-pair record; None when the record is absent or its pair was dropped."""
    if not prev or prev.get("dropped"):
        return None
    return significant_regions(prev)


def previous_grid_map(prev) -> np.ndarray | None:
    """Grid Map of a previous-pair record `{"regions", "box_ok_50_20", "dropped"?}`; None when the record is absent
    or its pair was dropped (the Previous-Target Floor then has no map: scoring treats it as all black)."""
    if not prev or prev.get("dropped"):
        return None
    return hints.change_map(significant_regions(prev), GRID).numpy()


def read_previous(tier, split: str) -> dict:
    """Previous-pair records by Window id, from the explicit per-split file `<split>_previous.jsonl`
    (rows `{"id", "previous": {...} | null}`). `{}` when the file is absent (inline row fields are used instead)."""
    path = Path(tier) / f"{split}_previous.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["id"]: r.get("previous") for r in rows}


def load_windows(tier, split: str, verdict: bool = False) -> list[Window]:
    """Windows of `split` in file order. Dropped Pairs are left out; a No Change Window (or a target row with no
    Significant Region) keeps an all-black Grid Map. The test split is refused unless `verdict` is True."""
    if split == TEST and not verdict:
        raise PermissionError("the test split is read only for a verdict: pass --verdict")
    if split not in SPLITS and split != TEST:
        raise ValueError(f"unknown split {split!r}")
    out = []
    external = read_previous(tier, split)
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
        key = window_key(row["id"], row.get("source_window_id"))
        prev = external[key] if key in external else (external[row["id"]] if row["id"] in external else row.get("previous"))
        out.append(Window(row["id"], regions, hints.change_map(regions, GRID).numpy(), previous_grid_map(prev),
                          row.get("previous_frame"), previous_regions(prev), row.get("source_window_id")))
    return out


def load_index(tier, split: str, windows: list[Window], verdict: bool = False) -> list[tuple]:
    """Join `windows` to `<tier>/<split>_index.jsonl` by Step Pair id and source Window (`<id>@<source_window_id>`).
    Refuses missing files, missing rows, or regions that differ from the tier row's Significant Regions. Returns one
    tuple of Significant Regions per Window."""
    if split == TEST and not verdict:
        raise PermissionError("the test split is read only for a verdict: pass --verdict")
    path = Path(tier) / f"{split}_index.jsonl"
    if not path.exists():
        raise ValueError(f"{path} not found: change_map_pixels needs the Change Map image index")
    by_win = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        by_win[r["window"]] = tuple(tuple(float(v) for v in b) for b in r["regions_px_1920x1080"])
    out = []
    for w in windows:
        key = w.key
        if key not in by_win:
            raise ValueError(f"{path}: no index row for {key!r}")
        regs = by_win[key]
        if regs != w.regions:
            raise ValueError(f"{path}: index regions for {key!r} differ from the tier row's Significant Regions")
        out.append(regs)
    return out


def to_model_range(grid_map: np.ndarray) -> np.ndarray:
    """[0, 1] -> [-1, 1], with a trailing channel: (..., 28, 50, 1)."""
    return (np.asarray(grid_map, np.float32) * 2 - 1)[..., None]


def from_model_range(x) -> np.ndarray:
    """(..., 28, 50, 1) in about [-1, 1] -> (..., 28, 50) clipped to [0, 1]."""
    return np.clip((np.asarray(x)[..., 0] + 1) / 2, 0, 1)


def load_context(tier, split: str, ids, verdict: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(f8 (n, 28, 50, 768), caption (n, 77, 768)) float32, rows in the order of `ids`."""
    if split == TEST and not verdict:
        raise PermissionError("the test split is read only for a verdict: pass --verdict")
    path = Path(tier) / f"{split}_context.npz"
    if not path.exists():
        raise ValueError(f"{path} not found: Conditioning Arms that read context need it (format: see changemap.data)")
    z = np.load(path)
    row = {str(i): j for j, i in enumerate(z["ids"])}
    missing = [i for i in ids if i not in row]
    if missing:
        raise ValueError(f"{path} has no context for Windows {missing[:3]}")
    idx = [row[i] for i in ids]
    return z["f8"][idx].astype(np.float32), z["caption"][idx].astype(np.float32)

def mean_grid_map(tier) -> np.ndarray:
    """Mean-Map Floor: the average training Grid Map, (28, 50)."""
    return np.mean([w.grid_map for w in load_windows(tier, "train")], axis=0).astype(np.float32)
