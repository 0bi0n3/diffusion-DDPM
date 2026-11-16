import torch

import changemap.siblings  # noqa: F401  (puts latentdit, dit2vlm and change_regions on the path)

torch.set_num_threads(1)
import pytest


@pytest.fixture(autouse=True)
def fake_lpips(monkeypatch):
    """Every test scores with the tiny fake LPIPS in tests/fake_lpips.py; nothing is downloaded."""
    monkeypatch.setenv("CHANGEMAP_LPIPS_MODULE", "fake_lpips")
    monkeypatch.delenv("CHANGEMAP_LPIPS_DIR", raising=False)
