"""The Change Map denoiser for all four Conditioning Arms: one class, so one architecture and one parameter count.

Built on the DiT study's `latentdit.dit.DiT` (28x50 grid, one channel). `latentdit` has no cross-attention, so
`ChangeMapDiT` subclasses it here and adds the Caption path; `latentdit` is not edited.

Context is a pair `(f8, caption)` passed through the transport as the opaque `context`:
  f8       (B, 28, 50, 768) Previous Frame DINOv3 tokens, or None
  caption  (B, 77, 768)     Caption CLIP text tokens, or None
Previous Frame: DiT's per-square input layer takes [noisy map || f8] (context_channels = 768); a linear layer over a
concatenation equals a 768 -> hidden projection of f8 added per token, which is what the spec asks for.
Caption: a cross-attention layer (queries from the squares, keys/values from the 77 Caption tokens, each projected
768 -> hidden) after every block.
Absent context: each model owns a learned null tensor of the same shape (`null_frame` (28, 50, 768) and
`null_caption` (77, 768)), always allocated, so the arms differ only in which context they read. An arm that does not
use a context ignores whatever is passed for it (and None is fine), so swapping that context cannot change its output.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import siblings  # noqa: F401
from latentdit.dit import DiT, modulate  # noqa: E402,F401

from .data import GRID

FRAME_CHANNELS = 768
CAPTION_TOKENS = 77
CAPTION_CHANNELS = 768

# arm -> (reads Previous Frame, reads Caption)
ARMS = {"unconditional": (False, False), "caption_only": (False, True),
        "frame_only": (True, False), "frame_caption": (True, True)}


def uses_context(arm: str) -> bool:
    return any(ARMS[arm])


class CrossAttention(nn.Module):
    """Multi-head cross-attention: squares attend to Caption tokens."""

    def __init__(self, dim, num_heads, ctx_dim=CAPTION_CHANNELS):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(ctx_dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, ctx):
        b, n, c = x.shape
        h = self.num_heads
        q = self.q(self.norm(x)).reshape(b, n, h, c // h).transpose(1, 2)
        k, v = self.kv(ctx).reshape(b, ctx.shape[1], 2, h, c // h).permute(2, 0, 3, 1, 4)
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c))


class ChangeMapDiT(DiT):
    def __init__(self, arm: str, hidden_size=384, depth=12, num_heads=6, channels: int = 1):
        super().__init__(grid=GRID, channels=channels, context_channels=FRAME_CHANNELS, hidden_size=hidden_size,
                         depth=depth, num_heads=num_heads)
        self.arm = arm
        self.use_frame, self.use_caption = ARMS[arm]
        self.null_frame = nn.Parameter(torch.zeros(*GRID, FRAME_CHANNELS))
        self.null_caption = nn.Parameter(torch.zeros(CAPTION_TOKENS, CAPTION_CHANNELS))
        nn.init.normal_(self.null_frame, std=0.02)
        nn.init.normal_(self.null_caption, std=0.02)
        self.cross = nn.ModuleList([CrossAttention(hidden_size, num_heads) for _ in range(depth)])
        for m in self.cross:
            nn.init.xavier_uniform_(m.q.weight), nn.init.xavier_uniform_(m.kv.weight)
            nn.init.xavier_uniform_(m.proj.weight)
            nn.init.zeros_(m.q.bias), nn.init.zeros_(m.kv.bias), nn.init.zeros_(m.proj.bias)

    def resolve_context(self, b, context, device, dtype):
        f8, cap = context if context is not None else (None, None)
        if self.use_frame:
            if f8 is None:
                raise ValueError(f"arm {self.arm!r} needs Previous Frame tokens")
            f8 = f8.to(device=device, dtype=dtype)
        else:
            f8 = self.null_frame.to(dtype).expand(b, *GRID, FRAME_CHANNELS)
        if self.use_caption:
            if cap is None:
                raise ValueError(f"arm {self.arm!r} needs Caption tokens")
            cap = cap.to(device=device, dtype=dtype)
        else:
            cap = self.null_caption.to(dtype).expand(b, CAPTION_TOKENS, CAPTION_CHANNELS)
        return f8, cap

    def forward(self, x, t, context):
        b, h, w, _ = x.shape
        f8, cap = self.resolve_context(b, context, x.device, x.dtype)
        x = self.x_embedder(torch.cat([x, f8], dim=-1).reshape(b, h * w, -1)) + self.pos_embed
        c = self.t_embedder(t * self.t_scale)
        for block, cross in zip(self.blocks, self.cross):
            x = block(x, c)
            x = x + cross(x, cap)
        return self.final_layer(x, c).reshape(b, h, w, self.channels)


def build_model(arm: str, hidden: int = 384, depth: int = 12, heads: int = 6, channels: int = 1) -> ChangeMapDiT:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}: choose from {tuple(ARMS)}")
    return ChangeMapDiT(arm, hidden, depth, heads, channels)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
