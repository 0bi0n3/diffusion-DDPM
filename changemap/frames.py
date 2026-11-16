"""Frames of the Masked Frame target: the frame cache, exact-pixel Change Masks, and 8x8 space-to-depth folding.

Frame cache, one file per split beside the targets: `<tier>/<split>_frames.npz`
  ids  (n,)               Window id strings; may include Dropped Pairs and extra Windows (looked up by id)
  f8   (n, 224, 400, 3)   Previous Frame (F8) RGB, uint8
  f9   (n, 224, 400, 3)   F9 RGB, uint8
The cache holds unmasked frames; the Change Mask is applied at load time from the Window's Significant Regions.
Its sha256 enters the run record with the other train/val files of the tier (`runrecord.data_sha256`).

Change Mask: a 1920x1080 box scales to x * 400/1920, y * 224/1080 with edges rounded to the nearest pixel (halves up);
a region narrower than one pixel keeps one pixel row/column. RGB outside the mask is the fill value 0.

Folding: (224, 400, 4) RGB+mask <-> (28, 50, 256). Square (r, c) holds pixel rows 8r..8r+7 and cols 8c..8c+7.
Channel order inside a square is pixel-major, then RGBM: index = (dy * 8 + dx) * 4 + ch, ch 0..3 = R, G, B, mask.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import GRID, TEST

H, W = 224, 400
SRC_W, SRC_H = 1920, 1080
BLOCK = 8
FILL = 0
CHANNELS = 4 * BLOCK * BLOCK          # 256
ON = 0.5                              # a generated mask is on where it exceeds this
MIN_CROP = 32


def pixel_box(box) -> tuple:
    """(x0, y0, x1, y1) of a 1920x1080 box on the 400x224 frame, at least one pixel wide and high, inside the frame."""
    def edge(v, scale, hi):
        return int(min(max(np.floor(float(v) * scale + 0.5), 0), hi))      # halves round up
    x0, y0 = edge(box[0], W / SRC_W, W), edge(box[1], H / SRC_H, H)
    x1, y1 = edge(box[2], W / SRC_W, W), edge(box[3], H / SRC_H, H)
    x0, y0 = min(x0, W - 1), min(y0, H - 1)
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def change_mask(regions) -> np.ndarray:
    """(224, 400) float32: 1 inside any Significant Region, 0 elsewhere (all zero for a No Change Window)."""
    m = np.zeros((H, W), np.float32)
    for b in regions:
        x0, y0, x1, y1 = pixel_box(b)
        m[y0:y1, x0:x1] = 1
    return m


def crop_box(regions) -> tuple:
    """(y0, x0, y1, x1): bounding box of the union of the regions' pixels, padded to at least 32x32 about its centre
    and kept inside the frame."""
    boxes = np.array([pixel_box(b) for b in regions])

    def span(a, b, size):
        if b - a < MIN_CROP:
            a = (a + b) // 2 - MIN_CROP // 2
            b = a + MIN_CROP
        a = min(max(a, 0), size - (b - a))
        return int(a), int(a + (b - a))
    ya, yb = span(int(boxes[:, 1].min()), int(boxes[:, 3].max()), H)
    xa, xb = span(int(boxes[:, 0].min()), int(boxes[:, 2].max()), W)
    return ya, xa, yb, xb


def masked_rgb(f9: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """F9 (224, 400, 3) uint8 where the mask is on, the fill value elsewhere."""
    return composite(f9, mask)


def composite(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """RGB where the mask is on (> 0.5), the fill value elsewhere. Works on any leading axes."""
    return np.where(np.asarray(mask)[..., None] > ON, rgb, np.uint8(FILL)).astype(np.uint8)


def fold(img: np.ndarray) -> np.ndarray:
    """(..., 224, 400, 4) -> (..., 28, 50, 256), exactly invertible by `unfold`."""
    img = np.asarray(img)
    lead = img.shape[:-3]
    x = img.reshape(*lead, GRID[0], BLOCK, GRID[1], BLOCK, 4)
    x = np.moveaxis(x, -4, -3)                         # (..., r, c, dy, dx, ch)
    return x.reshape(*lead, *GRID, CHANNELS)


def unfold(sq: np.ndarray) -> np.ndarray:
    """(..., 28, 50, 256) -> (..., 224, 400, 4)."""
    sq = np.asarray(sq)
    lead = sq.shape[:-3]
    x = sq.reshape(*lead, *GRID, BLOCK, BLOCK, 4)
    x = np.moveaxis(x, -3, -4)                         # (..., r, dy, c, dx, ch)
    return x.reshape(*lead, H, W, 4)


def to_model_range(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """uint8 RGB (224, 400, 3) and mask (224, 400) in [0, 1] -> folded (28, 50, 256) float32 in [-1, 1]."""
    x = np.concatenate([np.asarray(rgb, np.float32) / 255, np.asarray(mask, np.float32)[..., None]], -1)
    return fold(x * 2 - 1).astype(np.float32)


def from_model_range(x) -> tuple:
    """Model output (..., 28, 50, 256) -> (uint8 RGB (..., 224, 400, 3), mask (..., 224, 400) float32 in [0, 1])."""
    img = np.clip((unfold(np.asarray(x, np.float32)) + 1) / 2, 0, 1)
    return np.rint(img[..., :3] * 255).astype(np.uint8), img[..., 3].astype(np.float32)


def frames_path(tier, split: str) -> Path:
    return Path(tier) / f"{split}_frames.npz"


def write_cache(tier, split: str, ids, f8, f9) -> Path:
    path = frames_path(tier, split)
    np.savez(path, ids=np.array(list(ids)), f8=np.asarray(f8, np.uint8), f9=np.asarray(f9, np.uint8))
    return path


def load_frames(tier, split: str, ids, verdict: bool = False) -> tuple:
    """(f8, f9) uint8 (n, 224, 400, 3), rows in the order of `ids`."""
    if split == TEST and not verdict:
        raise PermissionError("the test split is read only for a verdict: pass --verdict")
    path = frames_path(tier, split)
    if not path.exists():
        raise ValueError(f"{path} not found: the masked_frame target needs the frame cache (format: see changemap.frames)")
    z = np.load(path)
    row = {str(i): j for j, i in enumerate(z["ids"])}
    missing = [i for i in ids if i not in row]
    if missing:
        raise ValueError(f"{path} has no frames for Windows {missing[:3]}")
    idx = [row[i] for i in ids]
    f8, f9 = z["f8"][idx], z["f9"][idx]
    if f9.shape[1:] != (H, W, 3) or f8.shape[1:] != (H, W, 3):
        raise ValueError(f"{path}: frames must be uint8 ({H}, {W}, 3)")
    return f8, f9
