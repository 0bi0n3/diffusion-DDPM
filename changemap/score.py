"""Score sampled Grid Maps (arms) and floors against the true Grid Maps: LPIPS, pixel MSE, PSNR per row, paired bootstrap.

Rows: each arm from its samples file, plus three floors scored identically (k = 1): All-Black, Mean-Map (average
training Grid Map) and Previous-Target (the Grid Map of the previous F7 -> F8 Step Pair, `data.load_windows`; a Window
with no previous pair gets an all black map and is counted in `n_no_previous`).
Per Window: single = mean over the k samples, best_of_k = minimum over them (an oracle selection).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import metrics, panels, report, targets
from .data import GRID, load_windows

NO_PREVIOUS_RULE = "A Window with no previous Step Pair (absent or dropped) is scored as all black."
ARM_ORDER = ("unconditional", "caption_only", "frame_only", "frame_caption")
FLOOR_NAMES = ("all_black_floor", "mean_map_floor", "previous_target_floor")
PANEL_WINDOWS = 4


def _row(kind, maps, truth, windows, model, batch, device) -> dict:
    """maps: (n, k, 28, 50); truth: (n, 28, 50)."""
    n, k = maps.shape[:2]
    flat_t = np.repeat(truth, k, axis=0)
    flat_p = maps.reshape(n * k, *GRID)
    pw = {"lpips": metrics.lpips_per_sample(model, flat_p, flat_t, batch, device).reshape(n, k),
          "mse": metrics.mse_per_sample(flat_p, flat_t).reshape(n, k)}
    nc = np.array([w.no_change for w in windows])
    summary = {"all": metrics.summarise(pw, np.ones(n, bool)), "change": metrics.summarise(pw, ~nc),
               "no_change": metrics.summarise(pw, nc)}
    row = {"kind": kind, "k": k, "summary": summary,
           "per_window": {w.id: {"lpips": pw["lpips"][i].tolist(), "mse": pw["mse"][i].tolist()}
                          for i, w in enumerate(windows)}}
    return row, pw


def _comparisons(values: dict, arms: list) -> list:
    """values[name] = {"single": {metric: (n,)}, "best_of_k": {...}}."""
    pairs = []
    for a in arms:
        pairs += [(f"{a}:{'single'}", f"{f}:single") for f in FLOOR_NAMES]
    for a, b in (("frame_caption", "caption_only"), ("caption_only", "unconditional")):
        if a in arms and b in arms:
            pairs.append((f"{a}:single", f"{b}:single"))
    pairs += [(f"{a}:best_of_k", f"{a}:single") for a in arms]
    out = []
    for a, b in pairs:
        for metric in ("lpips", "mse"):
            va, vb = (values[x.split(":")[0]][x.split(":")[1]][metric] for x in (a, b))
            out.append({"a": a, "b": b, "metric": metric, **metrics.paired_bootstrap(va - vb)})
    return out


def score(samples, tier, split: str, out, verdict: bool = False, lpips_dir=None, device: str = "cpu",
          batch: int = 4, panel_windows: int = PANEL_WINDOWS, target: str = targets.DEFAULT) -> dict:
    tg = targets.get(target)
    windows = load_windows(tier, split, verdict)
    ids = [w.id for w in windows]
    truth = np.stack([tg.truth(w) for w in windows])
    model = metrics.load_lpips(lpips_dir, device)
    arm_maps = {}
    for path in samples:
        z = np.load(path)
        targets.check(tg.name, str(z["target"]) if "target" in z.files else None, f"samples {path}")
        if list(z["ids"]) != ids:
            raise ValueError(f"{path}: Window ids differ from the {split} split")
        arm_maps[str(z["arm"])] = z["maps"]
    arm_maps = dict(sorted(arm_maps.items(), key=lambda kv: (ARM_ORDER.index(kv[0]) if kv[0] in ARM_ORDER else 99, kv[0])))
    n = len(windows)
    floor_maps = tg.floors(tier, windows)
    n_no_previous = sum(not w.has_previous for w in windows)
    rows, values, per_arm = {}, {}, {}
    for kind, group in (("arm", arm_maps), ("floor", floor_maps)):
        for name, maps in group.items():
            rows[name], pw = _row(kind, maps, truth, windows, model, batch, device)
            values[name] = {"single": {m: pw[m].mean(1) for m in pw}, "best_of_k": {m: pw[m].min(1) for m in pw}}
            if kind == "arm":
                per_arm[name] = {"maps": maps, "lpips": pw["lpips"]}
    rows["previous_target_floor"]["n_no_previous"] = n_no_previous
    panel_ids = ids[:panel_windows]
    out = Path(out)
    rep = {"target": tg.name, "split": split, "n_windows": n, "n_no_change": sum(w.no_change for w in windows), "window_ids": ids,
           "k": max((m.shape[1] for m in arm_maps.values()), default=1), "n_no_previous": n_no_previous,
           "no_previous_rule": NO_PREVIOUS_RULE, "bootstrap": {"resamples": metrics.RESAMPLES, "seed": metrics.BOOTSTRAP_SEED},
           "lpips": {"module": getattr(model, "__module__", type(model).__module__), "upscale": metrics.UPSCALE,
                     "weights_dir": str(lpips_dir) if lpips_dir else None},
           "caveats": [report.CAPTION_LEAK, report.LPIPS_CAVEAT], "panel_windows": panel_ids,
           "rows": rows, "comparisons": _comparisons(values, list(arm_maps))}
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")
    (out / "report.md").write_text(report.render(rep))
    panels.write_panels(out / "panels", per_arm, windows, truth, panel_ids)
    return rep
