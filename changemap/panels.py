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

    from .data import window_ids
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    index = {wid: i for i, wid in enumerate(window_ids(windows))}
    rows = target.panel_rows
    paths = []
    for name, a in arms.items():
        for wid in panel_ids:
            i = index[wid]
            k = min(a["pw"][target.metrics[0]].shape[1], max_samples)
            fig, axes = plt.subplots(rows, 2 + k, figsize=(2.2 * (2 + k), 2.4 * rows), squeeze=False)
            ax = axes[0, 0]
            input_frames = truth.get("input_frames") if isinstance(truth, dict) else None
            frame = input_frames.get(wid) if input_frames else None
            if frame is None:
                frame = target.panel_previous(truth, i, windows[i])
            if frame is not None:
                ax.imshow(frame)
            else:
                ax.text(0.5, 0.5, "Input Frame\nnot available" if target.name == "change_map_pixels" else "Previous Frame\nnot available",
                        ha="center", va="center", fontsize=7)
            ax.set_title("Input Frame" if target.name == "change_map_pixels" else "Previous Frame", fontsize=8)
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


def _decode_input_frames(tier, split: str, panel_ids: list[str]) -> dict:
    """Best-effort decode of Input Frame Ft (frames[-2]) at 224x400 RGB for just the selected panel Step Pairs."""
    import json
    from . import frames as frames_mod
    from .data import load_windows, window_ids

    path = Path(tier) / f"{split}_targets.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("target_status") != "dropped"]
    windows = load_windows(tier, split)
    ids = window_ids(windows)
    wanted_set = set(panel_ids)
    by_video: dict[str, set[int]] = {}
    row_lookup: dict[str, tuple[str, int]] = {}
    for wid, r in zip(ids, rows):
        if wid not in wanted_set:
            continue
        vid = r.get("video")
        fr = r.get("frames")
        if vid and fr and len(fr) >= 2 and Path(vid).is_file():
            sec = int(fr[-2])
            by_video.setdefault(vid, set()).add(sec)
            row_lookup[wid] = (vid, sec)
    if not by_video:
        return {}
    decoded = {}
    for vid, secs in by_video.items():
        _, got, _ = frames_mod._decode_video((vid, sorted(secs)))
        decoded[vid] = got
    out = {}
    for wid, (vid, sec) in row_lookup.items():
        if sec in decoded.get(vid, {}):
            out[wid] = decoded[vid][sec]
    return out


def render_from_report(tier, split: str, target_name: str, samples_paths, report_path, out,
                       n: int = 8, window_ids_override=None, verdict: bool = False) -> list[Path]:
    """Render CPU panels from already-sampled `.npz` files and an existing `report.json` (no LPIPS/GPU needed)."""
    import json
    import numpy as np
    from . import targets
    from .data import load_windows, window_ids

    tg = targets.get(target_name)
    if tg.panel_rows <= 0:
        raise ValueError(f"target {tg.name!r} has no image panels (not decoded to pixels)")
    windows = load_windows(tier, split, verdict)
    ids = window_ids(windows)
    truth = tg.truths(tier, split, windows, verdict)
    rep = json.loads(Path(report_path).read_text())

    per_arm = {}
    for sp in samples_paths:
        z = np.load(sp)
        targets.require_same_target(tg.name, str(z["target"]) if "target" in z.files else None, f"samples {sp}")
        if list(z["ids"]) != ids:
            raise ValueError(f"{sp}: Window ids differ from the {split} split")
        arm = str(z["arm"])
        if arm not in rep["rows"]:
            raise ValueError(f"arm {arm!r} not found in {report_path}")
        pw_json = rep["rows"][arm]["per_window"]
        pw = {
            m: np.array([[np.nan if v is None else float(v) for v in pw_json[wid][m]] for wid in ids], dtype=np.float64)
            for m in tg.metrics
        }
        smp = {key: z[key].astype(np.float32) if key == "mask" else z[key] for key in tg.sample_keys}
        per_arm[arm] = {"samples": smp, "pw": pw}

    if window_ids_override:
        panel_ids = list(window_ids_override)
    else:
        change_ids = [wid for wid, w in zip(ids, windows) if not w.no_change]
        other_ids = [wid for wid, w in zip(ids, windows) if w.no_change]
        panel_ids = (change_ids + other_ids)[:n]

    if tg.name == "change_map_pixels":
        truth = dict(truth)
        truth["input_frames"] = _decode_input_frames(tier, split, panel_ids)

    return write_panels(out, per_arm, windows, truth, panel_ids, tg)
