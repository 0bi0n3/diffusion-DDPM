"""Token-space scoring: generated and true Masked Frames through the DiT study's frozen encoder path.

Both sides take the same steps: composite (RGB where the mask is on, fill 0 elsewhere), bicubic x4 from 224x400 to
896x1600, then `encoder.encode(frames) -> (n, 28, 50, C)` (real path: resize 1600x896, ImageNet normalisation, DINOv3
ViT-B/16, 2x2 pool; see `latentdit.encode`). Per sample (all NaN where undefined):
  token_cosine         mean over the 1,400 squares of the cosine between generated and true tokens (two zero tokens: 1)
  region_token_cosine  the same mean over the squares that overlap a true region (true pooled mask > 0); NaN for a
                       No Change Window
  token_mse            mean squared token difference, in the DiT study's normalised latent space ((x - mean) / std from
                       its stats.npz) when stats are given, else on raw tokens (the report says which)
Higher is better for the cosines, lower for MSE.

Encoder injection (like `metrics.load_lpips`): `load_encoder(kind, device)` imports the module named by
`CHANGEMAP_ENCODER_MODULE` (default: this module) and calls its `load(kind, device)`. `kind` is `dinov3` (the real
encoder) or `standin` (the DiT study's StandInEncoder on the 28x50 grid, 8 channels). Tests inject `tests/fake_encoder.py`.

True-frame cache: `<tier>/token_cache/<split>__<encoder id>__<revision>.npz` (ids, input_sha256 per Window of the
composited true frame, tokens float16 (n, 28, 50, C)) and a `.json` sidecar with the encoder id and revision and the
sha256 of the .npz. A Window whose id or input hash is missing is encoded and the cache rewritten; a failed checksum is refused.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
from pathlib import Path

import numpy as np

from . import frames, siblings  # noqa: F401

KINDS = ("dinov3", "standin")
HIGHER_BETTER = frozenset({"token_cosine", "region_token_cosine"})
METRICS = ("token_cosine", "region_token_cosine", "token_mse")
SCALE = 4                       # 224x400 -> 896x1600
CHUNK = 16                      # frames upscaled and encoded at once
EPS = 1e-8


def load_encoder(kind: str = "dinov3", device: str = "cpu"):
    mod = importlib.import_module(os.environ.get("CHANGEMAP_ENCODER_MODULE", __name__))
    return mod.load(kind, device)


def load(kind, device="cpu"):
    """The default encoder module's loader: the DiT study's encoders."""
    from latentdit import encode
    if kind == "dinov3":
        return encode.DinoV3Encoder(device)
    if kind == "standin":
        return encode.StandInEncoder(grid=(28, 50), dim=8)
    raise ValueError(f"unknown encoder {kind!r}: choose from {KINDS}")


def load_stats(path):
    """(mean (C,), std (C,)) float32 of the DiT study's latent stats.npz, or None."""
    if path is None:
        return None
    z = np.load(path)
    return z["mean"].astype(np.float32), z["std"].astype(np.float32)


def upscale(frame_rgb) -> np.ndarray:
    import cv2
    return cv2.resize(np.ascontiguousarray(frame_rgb), (frames.W * SCALE, frames.H * SCALE), interpolation=cv2.INTER_CUBIC)


def encode_frames(encoder, rgb) -> np.ndarray:
    """Composited uint8 frames (n, 224, 400, 3) -> tokens float16 (n, 28, 50, C), `CHUNK` at a time."""
    out = [np.asarray(encoder.encode([upscale(f) for f in rgb[i:i + CHUNK]])) for i in range(0, len(rgb), CHUNK)]
    return np.concatenate(out) if out else np.zeros((0, 28, 50, 0), np.float16)


def cosine_map(a, b) -> np.ndarray:
    """Per-square cosine of token grids (..., 28, 50, C); two (near) zero tokens count as identical (1)."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    cos = (a * b).sum(-1) / np.maximum(na * nb, EPS)
    return np.where((na < EPS) & (nb < EPS), 1.0, cos)


def token_scores(pred_tokens, true_tokens, region, stats=None) -> dict:
    """One sample: tokens (28, 50, C) each, `region` (28, 50) bool = squares overlapping a true region."""
    cos = cosine_map(pred_tokens, true_tokens)
    a, b = np.asarray(pred_tokens, np.float32), np.asarray(true_tokens, np.float32)
    if stats is not None:
        if stats[0].shape != (a.shape[-1],):
            raise ValueError(f"latent stats have {stats[0].shape[0]} channels, the encoder {a.shape[-1]}")
        a, b = (a - stats[0]) / stats[1], (b - stats[0]) / stats[1]
    return {"token_cosine": float(cos.mean()),
            "region_token_cosine": float(cos[region].mean()) if region.any() else float("nan"),
            "token_mse": float(((a - b) ** 2).mean())}


# ---- true-frame cache -------------------------------------------------------------------------------------------

def _file_sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_path(tier, split, encoder) -> Path:
    tag = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{split}__{encoder.id}__{encoder.revision}")
    return Path(tier) / "token_cache" / f"{tag}.npz"


def true_tokens(tier, split, encoder, ids, true_rgb) -> np.ndarray:
    """Tokens (n, 28, 50, C) of the composited true frames `true_rgb` (n, 224, 400, 3), rows by `ids`. Cached per
    tier, split, encoder id and revision, keyed by Window id and the sha256 of the encoder input."""
    path = cache_path(tier, split, encoder)
    side = path.with_suffix(".json")
    shas = [hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest() for f in true_rgb]
    have = {}
    if path.exists():
        if not side.exists() or json.loads(side.read_text())["sha256"] != _file_sha(path):
            raise ValueError(f"{path}: token cache checksum mismatch (delete it to re-encode)")
        z = np.load(path)
        have = {(str(i), str(s)): t for i, s, t in zip(z["ids"], z["input_sha256"], z["tokens"])}
    todo, seen = [], set()
    for i, (w, s) in enumerate(zip(ids, shas)):
        if (w, s) not in have and (w, s) not in seen:
            todo.append(i)
            seen.add((w, s))
    if todo:
        for i, t in zip(todo, encode_frames(encoder, true_rgb[todo])):
            have[(ids[i], shas[i])] = t
        keys = sorted(have)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, ids=np.array([k[0] for k in keys]), input_sha256=np.array([k[1] for k in keys]),
                 tokens=np.stack([have[k] for k in keys]))
        side.write_text(json.dumps({"encoder": encoder.id, "revision": encoder.revision, "split": split,
                                    "sha256": _file_sha(path)}, indent=1) + "\n")
    return np.stack([have[(w, s)] for w, s in zip(ids, shas)])


def map_rgb(maps) -> np.ndarray:
    """(..., H, W) Change Map in [0, 1] -> (..., H, W, 3) uint8 white-on-black RGB."""
    u = np.rint(np.clip(np.asarray(maps, np.float32), 0, 1) * 255).astype(np.uint8)
    return np.repeat(u[..., None], 3, axis=-1)


def stats_path(tier, encoder) -> Path:
    tag = re.sub(r"[^A-Za-z0-9._-]+", "-", f"train_stats__{encoder.id}__{encoder.revision}")
    return Path(tier) / "token_cache" / f"{tag}.npz"


def map_latent_stats(tier, encoder, train_ids, train_maps) -> tuple[tuple[np.ndarray, np.ndarray], Path]:
    """Per-channel (mean (C,), std (C,)) float32 of the training Map Latents, cached with sha256 beside the token cache.
    Recomputed if the training token cache changes; near-zero std channels are clamped to 1.0."""
    toks = true_tokens(tier, "train", encoder, train_ids, map_rgb(train_maps))
    train_sha = _file_sha(cache_path(tier, "train", encoder))
    path = stats_path(tier, encoder)
    side = path.with_suffix(".json")
    if path.exists():
        meta = json.loads(side.read_text()) if side.exists() else {}
        if not side.exists() or meta.get("sha256") != _file_sha(path):
            raise ValueError(f"{path}: Map Latent stats checksum mismatch (delete it to recompute)")
        if meta.get("train_cache_sha256") == train_sha:
            z = np.load(path)
            return (z["mean"].astype(np.float32), z["std"].astype(np.float32)), path
    x = toks.astype(np.float64)
    mean = x.mean(axis=(0, 1, 2)).astype(np.float32)
    std = x.std(axis=(0, 1, 2))
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=mean, std=std)
    side.write_text(json.dumps({"encoder": encoder.id, "revision": encoder.revision,
                                "train_cache_sha256": train_sha, "sha256": _file_sha(path)}, indent=1) + "\n")
    return (mean, std), path


def train_map_latent_stats(tier, encoder) -> tuple[tuple[np.ndarray, np.ndarray], Path]:
    """Load/compute per-channel training Map Latent stats for `tier` from its verified train index."""
    from .data import load_index, load_windows
    train_w = load_windows(tier, "train")
    train_maps = np.stack([frames.change_mask(r) for r in load_index(tier, "train", train_w)])
    return map_latent_stats(tier, encoder, [w.id for w in train_w], train_maps)


def build_map_latents(tier, splits=("train", "val"), encoder_kind: str = "dinov3", device: str = "cpu",
                      verdict: bool = False) -> dict:
    """Pre-build per-split Map Latent caches and training Map Latent stats under `<tier>/token_cache/`."""
    from .data import TEST, load_index, load_windows
    for s in splits:
        if s == TEST and not verdict:
            raise PermissionError("the test split is read only for a verdict: pass --verdict")
    enc = load_encoder(encoder_kind, device)
    out = {"encoder": {"kind": encoder_kind, "id": enc.id, "revision": enc.revision}, "splits": {}}
    train_w = train_maps = None
    for s in splits:
        wins = load_windows(tier, s, verdict)
        regs = load_index(tier, s, wins, verdict)
        maps = np.stack([frames.change_mask(r) for r in regs])
        true_tokens(tier, s, enc, [w.id for w in wins], map_rgb(maps))
        cp = cache_path(tier, s, enc)
        out["splits"][s] = {"windows": len(wins), "cache": str(cp), "sha256": _file_sha(cp)}
        if s == "train":
            train_w, train_maps = wins, maps
    if train_w is not None:
        _, sp = map_latent_stats(tier, enc, [w.id for w in train_w], train_maps)
        out["train_stats"] = {"path": str(sp), "sha256": _file_sha(sp)}
    return out


def score_samples(encoder, tokens_true, region, rgb, mask, stats=None) -> dict:
    """Token metrics {name: (n, k)} of generated samples rgb (n, k, 224, 400, 3), mask (n, k, 224, 400)."""
    n, k = rgb.shape[:2]
    comp = frames.composite(rgb, mask).reshape(n * k, frames.H, frames.W, 3)
    return _score_rgb(encoder, tokens_true, region, comp, n, k, stats)


def score_maps(encoder, tokens_true, region, maps, stats=None) -> dict:
    """Token metrics {name: (n, k)} of generated Change Maps (n, k, 224, 400) in [0, 1]."""
    n, k = maps.shape[:2]
    comp = map_rgb(maps.reshape(n * k, frames.H, frames.W))
    return _score_rgb(encoder, tokens_true, region, comp, n, k, stats)


def _score_rgb(encoder, tokens_true, region, comp, n: int, k: int, stats=None) -> dict:
    out = {m: np.full((n, k), np.nan) for m in METRICS}
    for s in range(0, n * k, CHUNK):
        toks = encode_frames(encoder, comp[s:s + CHUNK])
        for off, t in enumerate(toks):
            i, j = divmod(s + off, k)
            for m, v in token_scores(t, tokens_true[i], region[i], stats).items():
                out[m][i, j] = v
    return out
