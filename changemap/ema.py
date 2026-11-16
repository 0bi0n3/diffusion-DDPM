"""Exponential moving average of the weights, used for sampling and validation.

Same scheme as the DiT latent study's trainer (latentdit/trainer.py `_update_ema`): the decay is warmed up as
min(decay, (1 + n) / (10 + n)) so the early EMA weights are usable.
"""
from __future__ import annotations

from copy import deepcopy

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.module = deepcopy(model).requires_grad_(False)
        self.decay, self.updates = decay, 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for e, p in zip(self.module.parameters(), model.parameters()):
            e.mul_(d).add_(p.detach(), alpha=1 - d)

    def state_dict(self) -> dict:
        return {"module": self.module.state_dict(), "updates": self.updates}

    def load_state_dict(self, state: dict) -> None:
        self.module.load_state_dict(state["module"])
        self.updates = state["updates"]
