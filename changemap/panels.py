"""Example panels for a fixed set of Windows: Previous Frame, the truth, k samples with per-sample scores.

What is drawn comes from the target (`panel_previous`, `panel_truth`, `panel_sample`, `panel_title`): one panel row for
a Grid Map; two for a Masked Frame (composited RGB above, Change Mask below)."""
from __future__ import annotations

from pathlib import Path


def write_panels(out, arms: dict, windows, truth, panel_ids, target, max_samples: int = 8) -> list[Path]:
    """arms: {name: {"samples": {key: (n, k, ...)}, "pw": {metric: (n, k)}}}, rows aligned with `windows`.
    One PNG per arm and Window."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    index = {w.id: i for i, w in enumerate(windows)}
    rows = target.panel_rows
    paths = []
    for name, a in arms.items():
        for wid in panel_ids:
            i = index[wid]
            k = min(a["pw"][target.metrics[0]].shape[1], max_samples)
            fig, axes = plt.subplots(rows, 2 + k, figsize=(2.2 * (2 + k), 2.4 * rows), squeeze=False)
            ax = axes[0, 0]
            frame = target.panel_previous(truth, i, windows[i])
            if frame is not None:
                ax.imshow(frame)
            else:
                ax.text(0.5, 0.5, "Previous Frame\nnot available", ha="center", va="center", fontsize=7)
            ax.set_title("Previous Frame", fontsize=8)
            for r, img in enumerate(target.panel_truth(truth, i)):
                axes[r, 1].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[0, 1].set_title("true map" if rows == 1 else "true Masked Frame", fontsize=8)
            for j in range(k):
                for r, img in enumerate(target.panel_sample(a["samples"], i, j)):
                    axes[r, 2 + j].imshow(img, cmap="gray", vmin=0, vmax=1)
                axes[0, 2 + j].set_title(target.panel_title(a["pw"], i, j), fontsize=8)
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
