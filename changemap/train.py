from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import siblings  # noqa: F401
from latentdit import transport  # noqa: E402

from .data import GRID, load_windows, to_model_range
from .model import build_model, empty_context


def train(tier, arm: str, steps: int, out, batch: int = 64, lr: float = 1e-4, hidden: int = 384, depth: int = 12,
          heads: int = 6, seed: int = transport.SEED, device: str = "cpu") -> dict:
    model = build_model(arm, hidden, depth, heads).to(device)
    data = torch.from_numpy(np.stack([to_model_range(w.grid_map) for w in load_windows(tier, "train")])).to(device)
    shift = transport.time_shift((*GRID, 1))
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    log = []
    model.train()
    for step in range(1, steps + 1):
        x0 = data[torch.randint(len(data), (batch,), generator=g).to(device)]
        t, xt, ut = transport.training_batch(x0, shift, g)
        loss = transport.velocity_loss(model(xt, t, empty_context(len(x0), device)), ut)
        opt.zero_grad()
        loss.backward()
        opt.step()
        log.append({"step": step, "loss": float(loss)})
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config = {"arm": arm, "steps": steps, "batch": batch, "lr": lr, "hidden": hidden, "depth": depth, "heads": heads,
              "seed": seed, "shift": shift, "tier": str(tier), "n_train_windows": len(data), "device": device}
    (out / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    (out / "log.json").write_text(json.dumps(log) + "\n")
    torch.save(model.state_dict(), out / "model.pt")
    return config
