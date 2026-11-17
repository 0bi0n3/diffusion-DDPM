"""Tiny stand-in for the CLIP ViT-L/14 text encoder, same contract: `load(device)` returns an encoder with `id`,
`revision` and `encode(texts) -> (n, 77, C)`. Deterministic per text; records every text it is asked to encode.
Selected in tests with CHANGEMAP_CLIP_MODULE=fake_clip; the real weights are never loaded."""
import zlib

import numpy as np

DIM = 8
TEXTS = []


class FakeClip:
    id, revision = "fake-clip", "rev1"

    def encode(self, texts):
        TEXTS.extend(texts)
        out = []
        for t in texts:
            g = np.random.default_rng(zlib.crc32(t.encode()))
            out.append(g.normal(size=(77, DIM)).astype(np.float32))
        return np.stack(out) if out else np.zeros((0, 77, DIM), np.float32)


def load(device="cpu"):
    return FakeClip()


def reset():
    TEXTS.clear()
