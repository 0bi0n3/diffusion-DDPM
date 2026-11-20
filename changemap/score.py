"""Score sampled targets (arms) and floors against the truth, with the target's metric set (`changemap.targets`);
Grid Map: LPIPS, pixel MSE; Masked Frame: full-frame and Region LPIPS, MSE, Mask MSE, Spill. PSNR per row, paired bootstrap.

Rows: each arm from its samples file, plus three floors scored identically (k = 1): All-Black, Mean-Map (average
training Grid Map) and Previous-Target (the Grid Map of the previous F7 -> F8 Step Pair, `data.load_windows`; a Window
with no previous pair gets an all black map and is counted in `n_no_previous`).
Per Window: single = mean over the k samples, best_of_k = minimum over them (an oracle selection).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import metrics, panels, report, targets, tokens
from .data import load_windows

NO_PREVIOUS_RULE = "A Window with no previous Step Pair (absent or dropped) is scored as all black."
ARM_ORDER = ("unconditional", "caption_only", "frame_only", "frame_caption")
PANEL_WINDOWS = 4


def _row(kind, pw, windows, higher_better) -> dict:
    """pw: {metric: (n, k)} per sample, NaN where undefined (None in the JSON)."""
    n, k = next(iter(pw.values())).shape
    nc = np.array([w.no_change for w in windows])
    hb = higher_better
    summary = {"all": metrics.summarise(pw, np.ones(n, bool), hb), "change": metrics.summarise(pw, ~nc, hb),
               "no_change": metrics.summarise(pw, nc, hb)}
    return {"kind": kind, "k": k, "summary": summary,
            "per_window": {w.id: {m: [None if np.isnan(v) else float(v) for v in pw[m][i]] for m in pw}
                           for i, w in enumerate(windows)}}


def _comparisons(values: dict, arms: list, tg, compare_metrics=None, groups=None) -> list:
    """values[name] = {"single": {metric: (n,)}, "best_of_k": {...}}."""
    pairs = []
    for a in arms:
        pairs += [(f"{a}:{'single'}", f"{f}:single") for f in tg.floor_names]
    for a, b in (("frame_caption", "caption_only"), ("caption_only", "unconditional")):
        if a in arms and b in arms:
            pairs.append((f"{a}:single", f"{b}:single"))
    pairs += [(f"{a}:best_of_k", f"{a}:single") for a in arms]
    out = []
    for a, b in pairs:
        for metric in compare_metrics or tg.compare_metrics:
            va, vb = (values[x.split(":")[0]][x.split(":")[1]][metric] for x in (a, b))
            boot = metrics.paired_bootstrap(va - vb, groups=groups)
            if boot is not None:
                h = getattr(tg, "hypotheses", {}).get((a, b))
                out.append({"a": a, "b": b, "metric": metric, **({"hypothesis": h} if h else {}), **boot})
    return out


def hypothesis_readings(comparisons: list) -> list:
    """Per hypothesis (H1-H5) that has both: the pixel reading (Region LPIPS, supported when the upper bound is < 0) and the
    token reading (Region Token Cosine, higher is better: supported when the lower bound is > 0), and whether they
    disagree. Flagged, never resolved."""
    by = {}
    for c in comparisons:
        if "hypothesis" in c and c["metric"] in ("region_lpips", "region_token_cosine"):
            by.setdefault(c["hypothesis"], {})[c["metric"]] = c
    out = []
    for h in sorted(by):
        pix, tok = by[h].get("region_lpips"), by[h].get("region_token_cosine")
        if pix is None or tok is None:
            continue
        ps, ts = pix["ci_hi"] < 0, tok["ci_lo"] > 0
        out.append({"hypothesis": h, "a": pix["a"], "b": pix["b"], "pixel_ci_hi": pix["ci_hi"], "pixel_supported": bool(ps),
                    "token_ci_lo": tok["ci_lo"], "token_supported": bool(ts), "disagree": bool(ps != ts)})
    return out


def score(samples, tier, split: str, out, verdict: bool = False, lpips_dir=None, device: str = "cpu",
          batch: int = 4, panel_windows: int = PANEL_WINDOWS, target: str = targets.DEFAULT,
          token_metrics: bool | None = None, encoder: str = "dinov3", encoder_device: str = "cpu",
          latent_stats=None) -> dict:
    tg = targets.get(target)
    is_latent = tg.name == "map_latent"
    supports_tokens = tg.name in ("masked_frame", "change_map_pixels", "map_latent")
    if token_metrics is None:
        token_metrics = supports_tokens
    if token_metrics and not supports_tokens:
        raise ValueError(f"--token-metrics on is not available for target {tg.name!r}")
    if is_latent and not token_metrics:
        raise ValueError("--token-metrics off is not available for target 'map_latent'")
    enc = tokens.load_encoder(encoder, encoder_device) if token_metrics else None
    stats = tokens.load_stats(latent_stats) if token_metrics else None
    stats_src = str(latent_stats) if latent_stats else None
    higher_better = tg.higher_better | (tokens.HIGHER_BETTER if token_metrics else frozenset())
    extra_cmp = ("region_token_cosine",) if (token_metrics and "region_token_cosine" not in tg.compare_metrics) else ()
    compare_metrics = tg.compare_metrics + extra_cmp
    windows = load_windows(tier, split, verdict)
    ids = [w.id for w in windows]
    if is_latent and stats is None:
        stats, sp = tokens.train_map_latent_stats(tier, enc)
        stats_src = str(sp)
    truth = tg.truths(tier, split, windows, verdict, enc=enc, stats=stats) if is_latent else tg.truths(tier, split, windows, verdict)
    model = None if is_latent else metrics.load_lpips(lpips_dir, device)
    arm_maps = {}
    for path in samples:
        z = np.load(path)
        targets.require_same_target(tg.name, str(z["target"]) if "target" in z.files else None, f"samples {path}")
        if list(z["ids"]) != ids:
            raise ValueError(f"{path}: Window ids differ from the {split} split")
        missing = [key for key in tg.sample_keys if key not in z.files]
        if missing:
            raise ValueError(f"{path}: no {missing} arrays (a {tg.name} samples file holds {tg.sample_keys})")
        arm_maps[str(z["arm"])] = {key: z[key].astype(np.float32) if key == "mask" else z[key] for key in tg.sample_keys}
    arm_maps = dict(sorted(arm_maps.items(), key=lambda kv: (ARM_ORDER.index(kv[0]) if kv[0] in ARM_ORDER else 99, kv[0])))
    n = len(windows)
    if token_metrics and not is_latent:
        if tg.name == "masked_frame":
            true_tok = tokens.true_tokens(tier, split, enc, ids, truth["rgb"])
            region = truth["grid"] > 0
        else:
            true_tok = tokens.true_tokens(tier, split, enc, ids, tokens.map_rgb(truth["maps"]))
            region = np.stack([metrics.pool_mask(m) > 0 for m in truth["maps"]])
            if stats is None:
                stats, sp = tokens.train_map_latent_stats(tier, enc)
                stats_src = str(sp)
    floor_maps = tg.floors(tier, split, windows, truth, enc=enc, stats=stats) if is_latent else tg.floors(tier, split, windows, truth)
    n_no_previous = sum(not w.has_previous for w in windows)
    rows, values, per_arm = {}, {}, {}
    for kind, group in (("arm", arm_maps), ("floor", floor_maps)):
        for name, smp in group.items():
            pw = (tg.score_samples(smp, truth, windows, model, batch, device, stats=stats) if is_latent
                  else tg.score_samples(smp, truth, windows, model, batch, device))
            if token_metrics and not is_latent:
                if tg.name == "masked_frame":
                    pw.update(tokens.score_samples(enc, true_tok, region, smp["rgb"], smp["mask"], stats))
                else:
                    pw.update(tokens.score_maps(enc, true_tok, region, smp["maps"], stats))
            rows[name] = _row(kind, pw, windows, higher_better)
            values[name] = {"single": {m: metrics.mean_ignoring_nan(pw[m], axis=1) for m in pw},
                            "best_of_k": {m: metrics.best_of_k(pw[m], m in higher_better) for m in pw}}
            if kind == "arm":
                per_arm[name] = {"samples": smp, "pw": pw}
    if "previous_target_floor" in rows:
        rows["previous_target_floor"]["n_no_previous"] = n_no_previous
    panel_ids = ids[:panel_windows] if tg.panel_rows > 0 else []
    out = Path(out)
    rep = {"target": tg.name, "split": split, "n_windows": n, "n_no_change": sum(w.no_change for w in windows), "window_ids": ids,
           "k": max((m[tg.sample_keys[0]].shape[1] for m in arm_maps.values()), default=1),
           **({"n_no_previous": n_no_previous, "no_previous_rule": getattr(tg, "no_previous_rule", NO_PREVIOUS_RULE)}
              if "previous_target_floor" in tg.floor_names else {}),
           **({"tokens": {"encoder": {"kind": encoder, "id": enc.id, "revision": enc.revision, "device": encoder_device},
                          "latent_stats": stats_src, "normalised": stats is not None,
                          "cache": str(tokens.cache_path(tier, split, enc))}} if token_metrics else {}),
           **({"metric_groups": tg.groups} if hasattr(tg, "groups") else {}),
           "bootstrap": {"resamples": metrics.RESAMPLES, "seed": metrics.BOOTSTRAP_SEED},
           **({"lpips": {"module": getattr(model, "__module__", type(model).__module__), "upscale": tg.upscale,
                         "weights_dir": str(lpips_dir) if lpips_dir else None}} if model is not None else {}),
           "caveats": [report.CAPTION_LEAK] + ([] if is_latent else [report.LPIPS_CAVEAT]), "panel_windows": panel_ids,
           "rows": rows, "comparisons": _comparisons(values, list(arm_maps), tg, compare_metrics,
                                                     groups=[w.source_window_id or w.id for w in windows])}
    if token_metrics:
        rep["hypothesis_readings"] = hypothesis_readings(rep["comparisons"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")
    (out / "report.md").write_text(report.render(rep))
    if tg.panel_rows > 0:
        try:
            panels.write_panels(out / "panels", per_arm, windows, truth, panel_ids, tg)
        except ModuleNotFoundError as e:                 # the AISurrey container has no matplotlib: the report is complete without
            print(f"changemap score: panels skipped ({e}); report.json and report.md are written", file=sys.stderr)
    return rep
