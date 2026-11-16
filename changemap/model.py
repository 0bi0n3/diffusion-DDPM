"""The Change Map denoiser: the DiT study's `latentdit.dit.DiT` over the 28x50 grid, one channel, no context yet."""
from __future__ import annotations

import torch

from . import siblings  # noqa: F401
from latentdit.dit import DiT  # noqa: E402

from .data import GRID

ARMS = ("unconditional",)


def build_model(arm: str, hidden: int = 384, depth: int = 12, heads: int = 6) -> DiT:
    if arm not in ARMS:
        raise ValueError(f"arm {arm!r} is not available yet: only {ARMS} are accepted; the other "
                         "Conditioning Arms come in #4")
    return DiT(grid=GRID, channels=1, context_channels=0, hidden_size=hidden, depth=depth, num_heads=heads)


def empty_context(n: int, device="cpu") -> torch.Tensor:
    """The Unconditional arm's context: zero channels per square."""
    return torch.zeros(n, *GRID, 0, device=device)
