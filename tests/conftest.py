import torch

import changemap.siblings  # noqa: F401  (puts latentdit, dit2vlm and change_regions on the path)

torch.set_num_threads(1)
import pytest


@pytest.fixture(autouse=True)
def fake_lpips(monkeypatch):
    """Every test scores with the tiny fake LPIPS in tests/fake_lpips.py; nothing is downloaded."""
    monkeypatch.setenv("CHANGEMAP_LPIPS_MODULE", "fake_lpips")
    monkeypatch.delenv("CHANGEMAP_LPIPS_DIR", raising=False)


@pytest.fixture(autouse=True)
def fake_encoder(monkeypatch):
    """Token-space scoring uses the stand-in encoder in tests/fake_encoder.py (28x50 grid); the DINOv3 weights are never loaded."""
    import fake_encoder as fe
    fe.reset()
    monkeypatch.setenv("CHANGEMAP_ENCODER_MODULE", "fake_encoder")
    return fe


@pytest.fixture(autouse=True)
def fake_clip(monkeypatch):
    """Captions go through the tiny stand-in in tests/fake_clip.py; the CLIP weights are never loaded."""
    import fake_clip as fc
    fc.reset()
    monkeypatch.setenv("CHANGEMAP_CLIP_MODULE", "fake_clip")
    return fc
