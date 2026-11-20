from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import siblings  # noqa: F401
from latentdit import transport  # noqa: E402

from . import targets
from .data import GRID, load_context, load_windows
from .model import build_model, uses_context


def sample(run, tier, split: str, k: int, steps: int, seed: int, out, verdict: bool = False,
           device: str = "cpu", batch: int = 32, target: str = targets.DEFAULT) -> Path:
    """k samples per Window, to `out/<arm>.npz`: ids, the target's arrays (Grid Map: maps (n, k, 28, 50) in [0, 1];
    Masked Frame: rgb (n, k, 224, 400, 3) uint8 and mask (n, k, 224, 400) in [0, 1]), arm, seed, steps, split, target.
    Sample j of a Window starts from `window_noise` seeded seed + j: per Window and reproducible."""
    run = Path(run)
    cfg = json.loads((run / "config.json").read_text())
    targets.require_same_target(target, cfg.get("target"), f"run {run}")
    tg = targets.get(target)
    windows = load_windows(tier, split, verdict)
    model = build_model(cfg["arm"], cfg["hidden"], cfg["depth"], cfg["heads"], tg.channels,
                        qk_norm=cfg.get("qk_norm", False))
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    model.to(device).eval()
    ids = [w.id for w in windows]
    if not ids or k < 1:
        raise ValueError(f"nothing to sample: {len(ids)} Windows, k = {k}")
    ctx = [torch.from_numpy(a) for a in load_context(tier, split, ids, verdict)] if uses_context(cfg["arm"]) else None
    out_arrays = None
    for j in range(k):
        for i in range(0, len(ids), batch):
            noise = transport.window_noise(ids[i:i + batch], tg.shape, seed + j).to(device)
            x = transport.sample_euler(model, noise, (ctx[0][i:i + batch].to(device), ctx[1][i:i + batch].to(device)) if ctx else None, steps, cfg["shift"])
            y = tg.decode(x.cpu().numpy())
            if "mask" in y:
                y["mask"] = y["mask"].astype(np.float16)       # half the size on disk (0.5 is exact); score casts back
            if out_arrays is None:
                out_arrays = {key: np.zeros((len(ids), k, *v.shape[1:]), v.dtype) for key, v in y.items()}
            for key, v in y.items():
                out_arrays[key][i:i + batch, j] = v
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{cfg['arm']}.npz", ids=np.array(ids), **out_arrays, arm=cfg["arm"], seed=seed, steps=steps,
             split=split, target=tg.name)
    return out / f"{cfg['arm']}.npz"
