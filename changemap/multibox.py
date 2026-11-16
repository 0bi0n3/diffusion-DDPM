"""Matched mean IoU terms, copied from DiT2VLM (copied code, not imported).

  Source repo: /mnt/hitchcock/scratch/oberon/DiT2VLM
  Branch:      frame10-a2-map-sig
  Commit:      bd6f1a15be01c6fd390fda682861f57194a17b01
  File:        dit2vlm/multibox_score.py   (`iou_matrix`, `window_terms`, `pooled`; bodies unchanged)

Per Window: a one-to-one Hungarian matching of boxes to targets that maximises total IoU (no IoU cutoff); the terms are
(sum of matched IoU, boxes + targets - matched pairs). Pooled over Windows: sum of numerators / sum of denominators.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    p, t = np.asarray(p, float).reshape(-1, 4), np.asarray(t, float).reshape(-1, 4)
    iw = np.minimum(p[:, None, 2], t[None, :, 2]) - np.maximum(p[:, None, 0], t[None, :, 0])
    ih = np.minimum(p[:, None, 3], t[None, :, 3]) - np.maximum(p[:, None, 1], t[None, :, 1])
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    ap, at = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1]), (t[:, 2] - t[:, 0]) * (t[:, 3] - t[:, 1])
    union = ap[:, None] + at[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def window_terms(p: np.ndarray, t: np.ndarray) -> tuple[float, int]:
    """(sum of matched IoU, boxes + targets - matched pairs) of one Window."""
    p, t = np.asarray(p, float).reshape(-1, 4), np.asarray(t, float).reshape(-1, 4)
    if not len(p) or not len(t):
        return 0.0, len(p) + len(t)
    iou = iou_matrix(p, t)
    r, c = linear_sum_assignment(-iou)
    return float(iou[r, c].sum()), len(p) + len(t) - len(r)


def pooled(terms) -> float | None:
    num, den = sum(n for n, _ in terms), sum(d for _, d in terms)
    return num / den if den else None
