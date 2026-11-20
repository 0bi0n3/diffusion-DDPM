"""Context cache of a tier split: `changemap context` and `changemap previous` (formats: see `changemap.data`).

`<out>/<split>_context.npz`: ids, f8 (n, 28, 50, C) float16, caption (n, 77, C) float16; one row per non-dropped Window
of `<tier>/<split>_targets.jsonl`, in file order (the same Windows as the frame cache).
  f8       F8 = row["frames"][8] (seconds). Read from the DiT study's Frame Latent store (`latentdit.encode.read_shard`,
           shards/t<T>/seconds.npy + latents.npy under each `--store`). The store holds RAW encoder tokens (the study's
           mean/std live beside it in stats.npz and are applied by its trainer), so every row is normalised exactly once
           here, in float32, (x - mean) / std with the DiT study's stats.npz, then stored as float16. A Window whose
           second is not in the store is encoded fresh with the same encoder (`tokens.load_encoder`, CPU by default;
           at most `--max-fresh`, else the command stops with the count) and normalised the same way. The encoder id of
           a shard must equal the fresh encoder's id.
  caption  frozen CLIP ViT-L/14 text (HF openai/clip-vit-large-patch14) `last_hidden_state`, 77 tokens, of
           "goal. subgoal. action" (each field stripped of trailing full stops, joined with ". "); never `evidence`.
           Padding is the tokenizer's, no attention mask (as in Stable Diffusion conditioning).
`<out>/<split>_previous.jsonl`: rows `{"id", "previous": {"regions", "box_ok_50_20"} | {"dropped": true} | null}`, the
F7 -> F8 Step Pair: `dit2vlm.tiers.previous_pair_id` looks the pair up in the row's `target_source` records; its first
REGION_CAP regions and their 50/20 flags (`dit2vlm.tiers.size_ok`, as `tiers.floor_answer`). null: no such pair.
`<out>/context_manifest.json`: per split and per command, counts, sha256 and bytes per file, CLIP and encoder ids.

CLIP injection (like `tokens.load_encoder`): `load_clip(device)` imports the module named by `CHANGEMAP_CLIP_MODULE`
(default: this module) and calls its `load(device)`; the encoder has `id`, `revision`, `encode(texts) -> (n, 77, C)`.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path

import numpy as np

from . import siblings, tokens  # noqa: F401
from .data import TEST, window_key

CLIP_ID = "openai/clip-vit-large-patch14"
MAX_FRESH = 50
CAPTION_FIELDS = ("goal", "subgoal", "action")


def load_clip(device: str = "cpu"):
    return importlib.import_module(os.environ.get("CHANGEMAP_CLIP_MODULE", __name__)).load(device)


def load(device: str = "cpu"):
    """The real CLIP ViT-L/14 text encoder from the HF cache."""
    import torch
    from transformers import CLIPTextModel, CLIPTokenizer
    from huggingface_hub import snapshot_download
    revision = Path(snapshot_download(CLIP_ID, allow_patterns=["config.json"])).name      # the snapshot's commit hash
    tok = CLIPTokenizer.from_pretrained(CLIP_ID, revision=revision)
    model = CLIPTextModel.from_pretrained(CLIP_ID, revision=revision).to(device).eval()

    class ClipText:
        id = CLIP_ID

        def encode(self, texts):
            out = []
            with torch.no_grad():
                for i in range(0, len(texts), 32):
                    ids = tok(list(texts[i:i + 32]), padding="max_length", max_length=77, truncation=True,
                              return_tensors="pt").input_ids.to(device)
                    out.append(model(input_ids=ids).last_hidden_state.float().cpu().numpy())
            return np.concatenate(out) if out else np.zeros((0, 77, 768), np.float32)
    ClipText.revision = revision
    return ClipText()


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def windows(tier, split: str, verdict: bool) -> list[dict]:
    """Non-dropped rows of the split's targets file, in file order."""
    if split == TEST and not verdict:
        raise PermissionError("the test split is read only for a verdict: pass --verdict")
    path = Path(tier) / f"{split}_targets.jsonl"
    if split not in ("train", "val", TEST) or not path.exists():
        raise ValueError(f"unknown split {split!r} or {path} not found")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if r["target_status"] != "dropped"]


def caption_text(row: dict) -> str:
    return ". ".join(str(row[k]).strip().rstrip(".").strip() for k in CAPTION_FIELDS)


def _file_info(path) -> dict:
    return {"sha256": sha256(path), "bytes": Path(path).stat().st_size}


def _update_manifest(out, split: str, kind: str, entry: dict) -> None:
    path = Path(out) / "context_manifest.json"
    m = json.loads(path.read_text()) if path.exists() else {}
    m.setdefault(split, {})[kind] = entry
    path.write_text(json.dumps(m, indent=1) + "\n")


def _store_index(stores, tutorials) -> dict:
    """{tutorial: (store dir, {second: row})} for the first store that holds each tutorial's finished shard."""
    from latentdit.encode import shard_dir
    idx = {}
    for t in sorted(tutorials):
        for s in stores:
            d = shard_dir(s, t)
            if (d / "shard.json").exists():
                info = json.loads((d / "shard.json").read_text())
                idx[t] = (s, {int(k): i for i, k in enumerate(np.load(d / "seconds.npy"))}, info.get("encoder"),
                          info.get("revision"))
                break
    return idx


def input_frame_second(row: dict) -> int:
    """Input Frame Ft of a Step Pair: frames[-2] is F8 on 10-frame Last-Step rows and F0 on 2-frame rows."""
    return int(row["frames"][-2])


def row_key(row: dict) -> str:
    return window_key(row["id"], row.get("source_window_id"))


def build_context(tier, split: str, out, stores, stats_path, verdict: bool = False, encoder_kind: str = "dinov3",
                  encoder_device: str = "cpu", clip_device: str = "cpu", max_fresh: int = MAX_FRESH,
                  verify_store: bool = False) -> dict:
    from latentdit.encode import decode_chunks, read_shard
    rows = windows(tier, split, verdict)
    mean, std = tokens.load_stats(stats_path)
    idx = _store_index([Path(s) for s in stores], {r["tutorial"] for r in rows})
    encoders = {(h[2], h[3]) for h in idx.values()}
    where = []                                   # per row: (store, row index) or None
    for r in rows:
        sec = input_frame_second(r)
        hit = idx.get(r["tutorial"])
        where.append((hit[0], hit[1][sec]) if hit and sec in hit[1] else None)
    fresh = [i for i, w in enumerate(where) if w is None]
    if len(fresh) > max_fresh:
        raise ValueError(f"{len(fresh)} Windows of {split} are not in the Frame Latent store (limit {max_fresh}): "
                         f"encode them in an AISurrey job, or raise --max-fresh")
    f8 = [None] * len(rows)
    shard_cache = {}
    for i, w in enumerate(where):
        if w is None:
            continue
        t = rows[i]["tutorial"]
        if (w[0], t) not in shard_cache:
            _, lat, info = read_shard(w[0], t, verify=verify_store)
            shard_cache[(w[0], t)] = (lat, info)
        f8[i] = np.asarray(shard_cache[(w[0], t)][0][w[1]], np.float32)
    enc_info = None
    if fresh:
        enc = tokens.load_encoder(encoder_kind, encoder_device)
        bad = {k for k in encoders if k[0] != enc.id}
        if bad:
            raise ValueError(f"the store was made by encoder {sorted(bad)}, the fresh encoder is {enc.id!r}")
        enc_info = {"id": enc.id, "revision": enc.revision}
        for i in fresh:
            r = rows[i]
            sec = input_frame_second(r)
            _, frames = next(decode_chunks(r["video"], [sec]), (None, None))
            if not frames:
                raise ValueError(f"{row_key(r)}: second {sec} cannot be decoded from {r['video']}")
            f8[i] = np.asarray(enc.encode([frames[0]])[0], np.float32)
    clip = load_clip(clip_device)
    caption = clip.encode([caption_text(r) for r in rows])
    f8 = np.stack(f8) if f8 else np.zeros((0, 28, 50, len(mean)), np.float32)
    if f8.shape[-1] != mean.shape[0]:
        raise ValueError(f"latent stats have {mean.shape[0]} channels, the tokens {f8.shape[-1]}")
    f8 = ((f8 - mean) / std).astype(np.float16)                       # normalised exactly once, here
    Path(out).mkdir(parents=True, exist_ok=True)
    path = Path(out) / f"{split}_context.npz"
    keys = [row_key(r) for r in rows]
    np.savez(path, ids=np.array(keys), f8=f8, caption=np.asarray(caption).astype(np.float16))
    entry = {"windows": len(rows), "from_store": len(rows) - len(fresh), "encoded_fresh": len(fresh),
             "fresh_ids": [keys[i] for i in fresh], "fresh_encoder": enc_info,
             "store_encoders": sorted(f"{a}@{b}" for a, b in encoders), "stores": [str(s) for s in stores],
             "stats_sha256": sha256(stats_path), "normalisation": "(x - mean) / std, once, float32 then float16",
             "clip": {"id": clip.id, "revision": clip.revision}, "caption": "goal. subgoal. action",
             "files": {path.name: _file_info(path)}}
    _update_manifest(out, split, "context", entry)
    return entry


def previous_record(row: dict):
    """The F7 -> F8 Step Pair record of a tier row (see the module docstring)."""
    from dit2vlm import tiers
    rec = tiers._pairs(row["target_source"]).get(
        tiers.previous_pair_id({"target_pair": row["target_key"], "tutorial": row["tutorial"]}))
    if rec is None:
        return None
    if rec.get("dropped"):
        return {"dropped": True}
    boxes = [list(b) for b in rec["regions"][:tiers.REGION_CAP]]
    return {"regions": boxes, "box_ok_50_20": [tiers.size_ok(b, "50_20") for b in boxes]}


def build_previous(tier, split: str, out, verdict: bool = False) -> dict:
    rows = windows(tier, split, verdict)
    recs = [previous_record(r) for r in rows]
    Path(out).mkdir(parents=True, exist_ok=True)
    path = Path(out) / f"{split}_previous.jsonl"
    path.write_text("".join(json.dumps({"id": row_key(r), "previous": p}) + "\n" for r, p in zip(rows, recs)))
    entry = {"windows": len(rows), "with_previous": sum(bool(p) and not p.get("dropped") for p in recs),
             "dropped": sum(bool(p) and bool(p.get("dropped")) for p in recs), "none": sum(p is None for p in recs),
             "no_significant_region": sum(bool(p) and not p.get("dropped") and not any(p["box_ok_50_20"]) for p in recs),
             "files": {path.name: _file_info(path)}}
    _update_manifest(out, split, "previous", entry)
    return entry
