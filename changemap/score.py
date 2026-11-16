"""Matched mIoU of boxes read from Grid Maps, against Significant Regions.

Rule (one Window): boxes are read from a map by `ceiling.map_boxes(map, "half")`. Predicted and true boxes are paired
one to one by the assignment that maximises total IoU (`scoring.iou` of `change_regions`); the Window's score is
the sum of the paired IoUs divided by max(#predicted, #true), so a missed region and a spurious box both cost.
A No Change Window (no Significant Region) scores 1.0 if the prediction has no box, else 0.0, as
`change_regions.scoring.no_change_correct` does. Per arm: mean over the k samples of a Window, then mean over Windows.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import siblings  # noqa: F401
from dit2vlm import ceiling  # noqa: E402
from change_regions import scoring  # noqa: E402

from .data import load_windows

THRESHOLD = "half"
RULE = ("matched mIoU: boxes = map_boxes(map, 'half'); one-to-one assignment maximising total IoU; score = sum of "
        "matched IoU / max(n_pred, n_true). No Change Window: 1.0 if the prediction has no box, else 0.0.")


def matched_miou(pred, true) -> float:
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    iou = np.array([[scoring.iou(tuple(p), tuple(t)) for t in true] for p in pred])
    r, c = linear_sum_assignment(-iou)
    return float(iou[r, c].sum() / max(len(pred), len(true)))


def window_score(pred_map, regions) -> float:
    return matched_miou(ceiling.map_boxes(pred_map, THRESHOLD), list(regions))


def score(samples, tier, split: str, out, verdict: bool = False) -> dict:
    windows = load_windows(tier, split, verdict)
    ids = [w.id for w in windows]
    rows = {}
    for path in samples:
        z = np.load(path)
        if list(z["ids"]) != ids:
            raise ValueError(f"{path}: Window ids differ from the {split} split")
        rows[str(z["arm"])] = {"k": int(z["maps"].shape[1]), "per_window": {
            w.id: float(np.mean([window_score(m, w.regions) for m in z["maps"][i]])) for i, w in enumerate(windows)}}
    rows["all_black_floor"] = {"per_window": {w.id: window_score(np.zeros((28, 50)), w.regions) for w in windows}}
    rows["oracle"] = {"per_window": {w.id: window_score(w.grid_map, w.regions) for w in windows}}
    for r in rows.values():
        r["matched_miou"] = float(np.mean(list(r["per_window"].values())))
    report = {"split": split, "threshold": THRESHOLD, "scoring_rule": RULE, "n_windows": len(ids),
              "n_no_change": sum(w.no_change for w in windows), "window_ids": ids, "rows": rows}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    return report
