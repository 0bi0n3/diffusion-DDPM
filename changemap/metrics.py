"""Image-similarity metrics of Grid Maps: LPIPS (VGG16, `latentdit.lpips`), pixel MSE, PSNR, paired bootstrap.

LPIPS module injection: `load_lpips` imports the module named by `CHANGEMAP_LPIPS_MODULE` (default
`latentdit.lpips`) and calls its `load(dir)`. Any module with that contract works, which is how tests use a tiny fake.
The weights folder is `--lpips-dir`, else `CHANGEMAP_LPIPS_DIR`, else the real module's own `LPIPS_DIR`. Nothing is
downloaded.
"""
from __future__ import annotations

import importlib
import os

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


def summarise(per_window: dict, mask) -> dict:
    """per_window: {"lpips": (n, k), "mse": (n, k)}; mask: (n,) bool of the Windows in this part."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        empty = {"lpips": None, "mse": None, "psnr": None}
        return {"n": 0, "single": dict(empty), "best_of_k": dict(empty)}
    out = {"n": int(mask.sum())}
    for name, reduce in (("single", np.mean), ("best_of_k", np.min)):
        l = float(reduce(per_window["lpips"][mask], axis=1).mean())
        m = float(reduce(per_window["mse"][mask], axis=1).mean())
        out[name] = {"lpips": l, "mse": m, "psnr": psnr(m)}
    return out


def paired_bootstrap(diffs, resamples: int = RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict:
    """Mean of per-Window differences and a 95% percentile interval over Window resamples."""
    d = np.asarray(diffs, np.float64)
    idx = np.random.default_rng(seed).integers(0, len(d), (resamples, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "n": int(len(d)),
            "resamples": resamples, "seed": seed}
