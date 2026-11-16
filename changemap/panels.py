"""Example panels for a fixed set of Windows: Previous Frame, true map, k sampled maps with per-sample LPIPS."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def write_panels(out, arms: dict, windows, true_maps, panel_ids, max_samples: int = 8) -> list[Path]:
    """arms: {name: {"maps": (n, k, 28, 50), "lpips": (n, k)}}, rows aligned with `windows`. One PNG per arm and Window."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    index = {w.id: i for i, w in enumerate(windows)}
    paths = []
    for name, a in arms.items():
        for wid in panel_ids:
            i = index[wid]
            k = min(a["maps"].shape[1], max_samples)
            fig, axes = plt.subplots(1, 2 + k, figsize=(2.2 * (2 + k), 2.4), squeeze=False)
            ax = axes[0, 0]
            frame = windows[i].previous_frame
            if frame and Path(frame).is_file():
                ax.imshow(plt.imread(frame))
            else:
                ax.text(0.5, 0.5, "Previous Frame\nnot available", ha="center", va="center", fontsize=7)
            ax.set_title("Previous Frame", fontsize=8)
            axes[0, 1].imshow(true_maps[i], cmap="gray", vmin=0, vmax=1)
            axes[0, 1].set_title("true map", fontsize=8)
            for j in range(k):
                axes[0, 2 + j].imshow(a["maps"][i, j], cmap="gray", vmin=0, vmax=1)
                axes[0, 2 + j].set_title(f"sample {j + 1}\nLPIPS {a['lpips'][i, j]:.3f}", fontsize=8)
            for ax in axes.ravel():
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle(f"{name}: Window {wid}", fontsize=9)
            fig.tight_layout()
            path = out / f"{name}_{wid}.png"
            fig.savefig(path, dpi=80)
            plt.close(fig)
            paths.append(path)
    return paths
