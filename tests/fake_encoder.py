"""Tiny stand-in for the DINOv3 encoder path, same contract: `load(kind, device)` returns an encoder with `id`,
`revision` and `encode(frames) -> (n, 28, 50, C)`. Wraps the DiT study's `StandInEncoder` (grid 28x50, dim DIM) and
counts the frames it is asked to encode, so tests can check that true-frame encodings are cached.
Selected in tests with CHANGEMAP_ENCODER_MODULE=fake_encoder; the real weights are never loaded."""
from latentdit.encode import StandInEncoder

DIM = 8
LOADED = []        # (kind, device) of every load
ENCODED = []       # shape of the frames of every encode call


class CountingEncoder(StandInEncoder):
    def encode(self, frames):
        ENCODED.append((len(frames), tuple(frames[0].shape) if len(frames) else None))
        return super().encode(frames)

    @property
    def n_frames(self):
        return sum(n for n, _ in ENCODED)


def load(kind, device="cpu"):
    LOADED.append((kind, device))
    return CountingEncoder(grid=(28, 50), dim=DIM)


def reset():
    LOADED.clear()
    ENCODED.clear()


def frames_encoded():
    return sum(n for n, _ in ENCODED)
