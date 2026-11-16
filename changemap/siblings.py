"""Sibling projects, imported rather than copied.

`latentdit` (DiT latent study: flow-matching transport, DiT network) and `dit2vlm` (grid `change_map`, ceiling
`map_boxes`; importing it also puts `change_regions` on the path). Roots come from `CHANGEMAP_LATENTDIT_ROOT` and
`CHANGEMAP_DIT2VLM_ROOT`. Importing this module is what makes `latentdit` and `dit2vlm` importable.
"""
from __future__ import annotations

import os
import sys

DEFAULT_LATENTDIT_ROOT = "/mnt/hitchcock/scratch/oberon/DiT-experiments"
DEFAULT_DIT2VLM_ROOT = "/mnt/hitchcock/scratch/oberon/DiT2VLM"


def roots() -> dict[str, str]:
    return {"latentdit": os.environ.get("CHANGEMAP_LATENTDIT_ROOT", DEFAULT_LATENTDIT_ROOT),
            "dit2vlm": os.environ.get("CHANGEMAP_DIT2VLM_ROOT", DEFAULT_DIT2VLM_ROOT)}


for _root in roots().values():
    if _root not in sys.path:
        sys.path.insert(0, _root)

import dit2vlm.cr  # noqa: E402,F401  (puts change_regions on the path)
