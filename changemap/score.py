"""Matched mean IoU of boxes read from Grid Maps, against Significant Regions (DiT2VLM's definition, `multibox.py`).

Boxes are read from a map by `ceiling.map_boxes(map, "half")`. Per Window: one-to-one Hungarian matching of boxes to
Significant Regions maximising total IoU (no cutoff); terms = (sum of matched IoU, boxes + regions - matched pairs).
Pooled over Windows: sum of numerators / sum of denominators. A No Change Window with no box adds nothing; a box on a
No Change Window adds to the denominator. With k samples, each sample index j is pooled over Windows on its own and the
row reports the mean over j (per-sample values and per-Window terms are kept for a later bootstrap).
No Change accuracy is separate: the share of No Change Windows whose prediction has no box
(`change_regions.scoring.no_change_correct`: an empty answer is correct).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import siblings  # noqa: F401
from dit2vlm import ceiling  # noqa: E402

from .data import load_windows
from .multibox import pooled, window_terms

THRESHOLD = "half"
RULE = ("matched mean IoU (DiT2VLM multibox_score): boxes = map_boxes(map, 'half'); per Window a one-to-one "
        "Hungarian matching maximising IoU, terms = (sum matched IoU, boxes + regions - matched); pooled over Windows "
        "= sum numerators / sum denominators; with k samples, pooled per sample index then averaged over the k. "
        "A No Change Window with no box adds nothing; a box on it adds to the denominator. "
        "No Change accuracy is reported apart: share of No Change Windows whose prediction has no box.")


def window_boxes(pred_map) -> list:
    return ceiling.map_boxes(pred_map, THRESHOLD)


def _row(maps_per_window, windows) -> dict:
    """maps_per_window[i]: (k, 28, 50) maps of Window i."""
    k = len(maps_per_window[0])
    boxes = [[window_boxes(m) for m in maps] for maps in maps_per_window]          # [window][sample]
    terms = [[window_terms(boxes[i][j], w.regions) for j in range(k)] for i, w in enumerate(windows)]
    per_sample = [pooled([terms[i][j] for i in range(len(windows))]) for j in range(k)]
    nc = [i for i, w in enumerate(windows) if w.no_change]
    nc_acc = [float(np.mean([not boxes[i][j] for i in nc])) for j in range(k)] if nc else None
    return {"k": k, "matched_miou": float(np.mean(per_sample)), "per_sample_miou": per_sample,
            "no_change_accuracy": float(np.mean(nc_acc)) if nc else None,
            "boxes_per_window": float(np.mean([len(b) for bw in boxes for b in bw])),
            "terms": {w.id: [list(t) for t in terms[i]] for i, w in enumerate(windows)}}


def score(samples, tier, split: str, out, verdict: bool = False) -> dict:
    windows = load_windows(tier, split, verdict)
    ids = [w.id for w in windows]
    rows = {}
    for path in samples:
        z = np.load(path)
        if list(z["ids"]) != ids:
            raise ValueError(f"{path}: Window ids differ from the {split} split")
        rows[str(z["arm"])] = _row(z["maps"], windows)
    rows["all_black_floor"] = _row([np.zeros((1, 28, 50))] * len(windows), windows)
    rows["oracle"] = _row([w.grid_map[None] for w in windows], windows)
    report = {"split": split, "threshold": THRESHOLD, "scoring_rule": RULE, "n_windows": len(ids),
              "n_no_change": sum(w.no_change for w in windows), "window_ids": ids, "rows": rows}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    return report
