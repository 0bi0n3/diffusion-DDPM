"""Image-similarity metrics of Grid Maps and Masked Frames: LPIPS (VGG16, `latentdit.lpips`), pixel MSE, PSNR, paired bootstrap.

LPIPS module injection: `load_lpips` imports the module named by `CHANGEMAP_LPIPS_MODULE` (default
`latentdit.lpips`) and calls its `load(dir)`. Any module with that contract works, which is how tests use a tiny fake.
The weights folder is `--lpips-dir`, else `CHANGEMAP_LPIPS_DIR`, else the real module's own `LPIPS_DIR`. Nothing is
downloaded.
"""
from __future__ import annotations

import importlib
import os
import warnings

import numpy as np
import torch

from . import siblings  # noqa: F401

UPSCALE = 32                 # 28x50 grid -> 896x1600 pixels
RESAMPLES = 1000
BOOTSTRAP_SEED = 35


def load_lpips(directory=None, device: str = "cpu"):
    mod = importlib.import_module(os.environ.get("CHANGEMAP_LPIPS_MODULE", "latentdit.lpips"))
    directory = directory or os.environ.get("CHANGEMAP_LPIPS_DIR") or None
    return mod.load(directory).to(device).eval()


def to_lpips_input(maps) -> torch.Tensor:
    """(B, 28, 50) maps in [0, 1] -> (B, 3, 896, 1600) in [-1, 1]: nearest-neighbour x32, repeated to 3 channels."""
    x = torch.as_tensor(np.asarray(maps, np.float32))
    x = x.repeat_interleave(UPSCALE, 1).repeat_interleave(UPSCALE, 2)[:, None].expand(-1, 3, -1, -1)
    return x * 2 - 1


def _distance(model, x, y) -> torch.Tensor:
    """Per-sample (B,) LPIPS. The batched path uses the VGG LPIPS structure (`_features`, `lins`, `shift`, `scale`)
    but keeps the batch axis the module's `forward` averages away; any other module is called once per sample."""
    if not all(hasattr(model, a) for a in ("_features", "lins", "shift", "scale")):
        return torch.stack([model(x[i:i + 1], y[i:i + 1]).reshape(()) for i in range(len(x))])
    fx, fy = model._features((x - model.shift) / model.scale), model._features((y - model.shift) / model.scale)
    total = 0
    for a, b, lin in zip(fx, fy, model.lins):
        a = a / (a.pow(2).sum(1, keepdim=True).sqrt() + 1e-10)
        b = b / (b.pow(2).sum(1, keepdim=True).sqrt() + 1e-10)
        total = total + lin((a - b) ** 2).mean([2, 3])
    return total.reshape(len(x))


@torch.no_grad()
def lpips_per_sample(model, preds, targets, batch: int = 4, device: str = "cpu") -> np.ndarray:
    """LPIPS of each (pred, target) pair of (N, 28, 50) maps, in batches of `batch`; lower is more similar."""
    out = []
    for i in range(0, len(preds), batch):
        x = to_lpips_input(preds[i:i + batch]).to(device)
        y = to_lpips_input(targets[i:i + batch]).to(device)
        out.append(_distance(model, x, y).cpu().numpy())
    return np.concatenate(out).astype(np.float64) if out else np.zeros(0)


def mse_per_sample(preds, targets) -> np.ndarray:
    d = np.asarray(preds, np.float64) - np.asarray(targets, np.float64)
    return (d ** 2).mean(axis=(-2, -1))


def psnr(mean_mse):
    """10 log10(1 / MSE) of a row's mean MSE; None when there is no Window or the match is exact (infinite)."""
    return None if mean_mse is None or mean_mse <= 0 else float(10 * np.log10(1 / mean_mse))


def mean_ignoring_nan(a, axis=None):
    """Mean ignoring NaN (an undefined metric); plain `np.mean` when nothing is undefined."""
    if not np.isnan(a).any():
        return np.mean(a, axis=axis)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(a, axis=axis)


def best_of_k(a, higher: bool):
    """Best sample of each Window over axis 1 (min, or max for higher-is-better); NaN when all are undefined."""
    if not np.isnan(a).any():
        return a.max(axis=1) if higher else a.min(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmax(a, axis=1) if higher else np.nanmin(a, axis=1)


def summarise(per_window: dict, mask, higher_better=()) -> dict:
    """per_window: {metric: (n, k)}, NaN where a metric is undefined for a Window (region metrics of a No Change
    Window, Spill when nothing is on); mask: (n,) bool of the Windows in this part. Each metric is averaged over the
    Windows where it is defined (None when there is none). PSNR comes from the row's mean `mse`."""
    mask = np.asarray(mask, bool)
    names = list(per_window)
    empty = {**{m: None for m in names}, "psnr": None}
    if not mask.any():
        return {"n": 0, "single": dict(empty), "best_of_k": dict(empty)}
    out = {"n": int(mask.sum())}
    for variant in ("single", "best_of_k"):
        vals = {}
        for m in names:
            a = per_window[m][mask]
            per = mean_ignoring_nan(a, axis=1) if variant == "single" else best_of_k(a, m in higher_better)
            vals[m] = None if np.isnan(per).all() else float(mean_ignoring_nan(per))
        vals["psnr"] = psnr(vals["mse"]) if "mse" in vals else None
        out[variant] = vals
    return out


def paired_bootstrap(diffs, resamples: int = RESAMPLES, seed: int = BOOTSTRAP_SEED, groups=None) -> dict | None:
    """Mean of per-entry differences and a 95% percentile interval over Window resamples. Entries where the difference
    is undefined (NaN) are left out; None when none is left. When `groups` has repeated Window ids (two-frame Step
    Pairs), whole Windows are resampled so every Step Pair of a Window moves together."""
    d = np.asarray(diffs, np.float64)
    keep = ~np.isnan(d)
    d = d[keep]
    if not len(d):
        return None
    if groups is None:
        idx = np.random.default_rng(seed).integers(0, len(d), (resamples, len(d)))
        means = d[idx].mean(axis=1)
        extra = {}
    else:
        g_keep = [g for g, ok in zip(groups, keep) if ok]
        uniq = list(dict.fromkeys(g_keep))
        if len(uniq) == len(d):
            idx = np.random.default_rng(seed).integers(0, len(d), (resamples, len(d)))
            means = d[idx].mean(axis=1)
            extra = {}
        else:
            pos = {u: i for i, u in enumerate(uniq)}
            sums = np.zeros(len(uniq), np.float64)
            counts = np.zeros(len(uniq), np.float64)
            for v, g in zip(d, g_keep):
                i = pos[g]
                sums[i] += v
                counts[i] += 1.0
            idx = np.random.default_rng(seed).integers(0, len(uniq), (resamples, len(uniq)))
            means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
            extra = {"n_windows": len(uniq)}
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "n": int(len(d)),
            **extra, "resamples": resamples, "seed": seed}


# ---- Masked Frame (224 x 400 RGB + Change Mask, no upscaling) --------------------------------------------------

def _to_input(img) -> torch.Tensor:
    """(B, H, W, 3) uint8 -> (B, 3, H, W) float in [-1, 1]."""
    return torch.as_tensor(np.ascontiguousarray(img)).permute(0, 3, 1, 2).float() / 127.5 - 1


@torch.no_grad()
def lpips_images(model, preds, truths, batch: int = 4, device: str = "cpu") -> np.ndarray:
    """LPIPS of each (pred, truth) pair of (N, H, W, 3) uint8 images, at their own size (no upscaling)."""
    out = []
    for i in range(0, len(preds), batch):
        x, y = _to_input(preds[i:i + batch]).to(device), _to_input(truths[i:i + batch]).to(device)
        out.append(_distance(model, x, y).cpu().numpy())
    return np.concatenate(out).astype(np.float64) if out else np.zeros(0)


def frame_mse(pred, truth) -> float:
    """MSE of two uint8 RGB images on the [0, 1] scale."""
    d = (np.asarray(pred, np.float32) - np.asarray(truth, np.float32)) / 255
    return float((d.astype(np.float64) ** 2).mean())


def mask_metrics(pred_mask, true_mask) -> tuple:
    """(Mask MSE, Spill) of a generated Change Mask against the true one (values in [0, 1], on where > 0.5).
    Spill is the share of mask-on pixels outside the true regions; NaN when nothing is on."""
    p, t = np.asarray(pred_mask, np.float64), np.asarray(true_mask, np.float64)
    on = p > 0.5
    n_on = int(on.sum())
    return float(((p - t) ** 2).mean()), (float((on & (t <= 0.5)).sum() / n_on) if n_on else float("nan"))


def pool_mask(mask, block: int = 8) -> np.ndarray:
    """(H, W) Change Mask -> (H/8, W/8) Grid Map: the share of each square that is on (> 0.5)."""
    on = (np.asarray(mask) > 0.5).astype(np.float32)
    h, w = on.shape
    return on.reshape(h // block, block, w // block, block).mean((1, 3))
