"""Targets: what the denoiser generates for a Window, behind one small interface.

A target says how many channels each Token Grid square holds, how Windows become training tensors (n, 28, 50, C) in
[-1, 1], how a model output becomes something scorable, which floors it is scored against, which metrics it reports
(and which of them are compared by paired bootstrap), how a batch of samples is scored per sample, and how a panel is
drawn. Train, sample and score take the target by name (`--target`), record it (config.json, the samples file,
report.json) and refuse a run or samples file made for a different target.

The interface (every target):
  name, channels, shape, upscale           identity, channels per square, one tensor's shape, LPIPS upscaling
  sample_keys                              arrays a samples file holds (leading axes n, k), e.g. ("maps",)
  metrics, compare_metrics, higher_better  reported metrics; those compared by bootstrap; those where higher is better
  floor_names                              floor names in report order
  tensors(tier, split, windows)            (n, 28, 50, C) float32 in [-1, 1]
  truths(tier, split, windows)             dict of arrays with leading axis n: the scorable truth
  decode(x)                                model output (n, 28, 50, C) -> dict of arrays (the sample keys)
  floors(tier, split, windows, truth)      floor name -> dict of arrays with leading axes (n, 1), in floor_names order
  score_samples(pred, truth, windows, lpips_model, batch, device)   {metric: (n, k)}, NaN where undefined
  validation_mse(decoded, truth)           the target-neutral validation number logged as `val_mse`
  panel_*                                  see changemap.panels

Targets today:
  grid_map      the Grid Map (#2): 1 channel, coverage of Significant Regions per square, scorable as (28, 50) in [0, 1].
  masked_frame  the Masked Frame (#10): F9 at 224x400 RGB + Change Mask folded 8x8 into 256 channels per square.
"""
from __future__ import annotations

import numpy as np

from . import frames, metrics
from .data import GRID, from_model_range, load_index, load_windows, mean_grid_map, to_model_range

DEFAULT = "masked_frame"


class GridMapTarget:
    name = "grid_map"
    channels = 1
    upscale = metrics.UPSCALE
    sample_keys = ("maps",)
    metrics = ("lpips", "mse")
    compare_metrics = ("lpips", "mse")
    higher_better = frozenset()
    floor_names = ("all_black_floor", "mean_map_floor", "previous_target_floor")
    panel_rows = 1

    @property
    def shape(self) -> tuple:
        """One Window's training tensor shape: (28, 50, channels)."""
        return (*GRID, self.channels)

    def tensors(self, tier, split, windows) -> np.ndarray:
        """(n, 28, 50, 1) in [-1, 1]."""
        return np.stack([to_model_range(w.grid_map) for w in windows])

    def truths(self, tier, split, windows, verdict: bool = False) -> dict:
        """The scorable true values: maps (n, 28, 50) in [0, 1]."""
        return {"maps": np.stack([w.grid_map for w in windows])}

    def decode(self, x) -> dict:
        """Model output (..., 28, 50, 1) -> maps (..., 28, 50) in [0, 1]."""
        return {"maps": from_model_range(x)}

    def floors(self, tier, split, windows, truth=None) -> dict:
        """Floor name -> {"maps": (n, 1, 28, 50)}, in report order."""
        n = len(windows)
        maps = {
            "all_black_floor": np.zeros((n, 1, *GRID), np.float32),
            "mean_map_floor": np.broadcast_to(mean_grid_map(tier), (n, 1, *GRID)).copy(),
            "previous_target_floor": np.stack([(w.previous_map if w.has_previous else np.zeros(GRID, np.float32))[None]
                                               for w in windows]).astype(np.float32),
        }
        return {k: {"maps": v} for k, v in maps.items()}

    def score_samples(self, pred, truth, windows, model, batch, device) -> dict:
        maps = pred["maps"]
        n, k = maps.shape[:2]
        flat_t = np.repeat(truth["maps"], k, axis=0)
        flat_p = maps.reshape(n * k, *GRID)
        return {"lpips": metrics.lpips_per_sample(model, flat_p, flat_t, batch, device).reshape(n, k),
                "mse": metrics.mse_per_sample(flat_p, flat_t).reshape(n, k)}

    def validation_mse(self, decoded, truth) -> float:
        return float(np.mean((decoded["maps"] - truth["maps"]) ** 2))

    # panels: images in [0, 1], one list entry per panel row
    def panel_previous(self, truth, i, window):
        frame = window.previous_frame
        if frame:
            from pathlib import Path
            if Path(frame).is_file():
                import matplotlib.pyplot as plt
                return plt.imread(frame)
        return None

    def panel_truth(self, truth, i) -> list:
        return [truth["maps"][i]]

    def panel_sample(self, samples, i, j) -> list:
        return [samples["maps"][i, j]]

    def panel_title(self, pw, i, j) -> str:
        return f"sample {j + 1}\nLPIPS {pw['lpips'][i, j]:.3f}"


class MaskedFrameTarget:
    """F9 at 224x400 as RGB plus Change Mask, folded 8x8 into (28, 50, 256) (see changemap.frames for the channel
    order). Scoring separates where (Mask MSE, Spill, the mask pooled to a Grid Map) from what (Region LPIPS and
    in-region RGB MSE on the uncomposited RGB), plus full-frame LPIPS and MSE on composited frames."""
    name = "masked_frame"
    channels = frames.CHANNELS
    upscale = 1
    sample_keys = ("rgb", "mask")
    metrics = ("lpips", "region_lpips", "mse", "region_mse", "mask_mse", "spill", "grid_lpips", "grid_mse")
    compare_metrics = ("lpips", "region_lpips", "mse")
    higher_better = frozenset()
    floor_names = ("all_black_floor", "mean_map_floor", "previous_target_floor", "copy_f8_floor")
    no_previous_rule = ("A Window with no previous Step Pair (absent or dropped) gets an all-zero mask for the "
                        "Previous-Target Floor.")
    panel_rows = 2
    # hypotheses of #8 (PREREG), each an (a, b) comparison read on Region LPIPS: negative upper bound supports it
    hypotheses = {("frame_caption:single", "previous_target_floor:single"): "H1",
                  ("frame_caption:single", "caption_only:single"): "H2",
                  ("caption_only:single", "unconditional:single"): "H3",
                  ("frame_caption:best_of_k", "frame_caption:single"): "H4",
                  ("frame_caption:single", "copy_f8_floor:single"): "H5"}
    groups = {"image": ["lpips", "mse"], "where": ["mask_mse", "spill", "grid_lpips", "grid_mse"],
              "what": ["region_lpips", "region_mse"]}

    @property
    def shape(self) -> tuple:
        return (*GRID, self.channels)

    def _frames(self, tier, split, windows, verdict=False):
        return frames.load_frames(tier, split, [w.id for w in windows], verdict)

    def tensors(self, tier, split, windows) -> np.ndarray:
        """(n, 28, 50, 256) in [-1, 1]: F9 where the Change Mask is on, fill 0 elsewhere, plus the mask."""
        _, f9 = self._frames(tier, split, windows)
        out = []
        for w, a in zip(windows, f9):
            m = frames.change_mask(w.regions)
            out.append(frames.to_model_range(frames.masked_rgb(a, m), m))
        return np.stack(out)

    def truths(self, tier, split, windows, verdict: bool = False) -> dict:
        """rgb (n, 224, 400, 3) uint8 masked F9, mask (n, 224, 400) in {0, 1}, f8 (n, 224, 400, 3) uint8 (Previous
        Frame, unmasked), crop (n, 4) the Region LPIPS crop (y0, x0, y1, x1; zeros for a No Change Window), grid
        (n, 28, 50) the true Change Mask pooled per square (pooled like the generated one, so a perfect mask scores 0)."""
        f8, f9 = self._frames(tier, split, windows, verdict)
        mask = np.stack([frames.change_mask(w.regions) for w in windows])
        rgb = np.stack([frames.masked_rgb(a, m) for a, m in zip(f9, mask)])
        crop = np.array([frames.crop_box(w.regions) if w.regions else (0, 0, 0, 0) for w in windows], np.int64)
        return {"rgb": rgb, "mask": mask, "f8": f8, "crop": crop, "grid": np.stack([metrics.pool_mask(m) for m in mask])}

    def decode(self, x) -> dict:
        rgb, mask = frames.from_model_range(x)
        return {"rgb": rgb, "mask": mask}

    def _mean_training(self, tier) -> tuple:
        """Mean-Map Floor: the mean training Change Mask (224, 400) float32 and the mean training RGB, the mean of the
        training Masked Frames' RGB (fill 0 outside their masks), rounded to uint8 (224, 400, 3)."""
        train = load_windows(tier, "train")
        _, f9 = self._frames(tier, "train", train)
        masks = np.stack([frames.change_mask(w.regions) for w in train])
        rgb = np.stack([frames.masked_rgb(a, m) for a, m in zip(f9, masks)])
        return masks.mean(0).astype(np.float32), np.rint(rgb.mean(0)).astype(np.uint8)

    def floors(self, tier, split, windows, truth=None) -> dict:
        n = len(windows)
        mean_mask, mean_rgb = self._mean_training(tier)
        prev_mask = np.stack([frames.change_mask(w.previous_regions or ()) for w in windows])
        return {"all_black_floor": {"rgb": np.zeros((n, 1, frames.H, frames.W, 3), np.uint8),
                                    "mask": np.zeros((n, 1, frames.H, frames.W), np.float32)},
                # oracle location: the true mask, with F8's pixels inside it (fill elsewhere, like a Masked Frame)
                "mean_map_floor": {"rgb": np.broadcast_to(mean_rgb, (n, 1, *mean_rgb.shape)).copy(),
                                   "mask": np.broadcast_to(mean_mask, (n, 1, *mean_mask.shape)).copy()},
                "previous_target_floor": {"rgb": frames.composite(truth["f8"], prev_mask)[:, None],
                                          "mask": prev_mask[:, None]},
                "copy_f8_floor": {"rgb": frames.composite(truth["f8"], truth["mask"])[:, None],
                                  "mask": truth["mask"][:, None].copy()}}

    def score_samples(self, pred, truth, windows, model, batch, device) -> dict:
        rgb, mask = pred["rgb"], pred["mask"]
        n, k = rgb.shape[:2]
        comp = np.stack([[frames.composite(rgb[i, j], mask[i, j]) for j in range(k)] for i in range(n)])
        flat_c = comp.reshape(n * k, *comp.shape[2:])
        flat_t = np.repeat(truth["rgb"], k, axis=0)             # truth RGB is already composited (masked F9)
        out = {m: np.full((n, k), np.nan) for m in self.metrics}
        out["lpips"] = metrics.lpips_images(model, flat_c, flat_t, batch, device).reshape(n, k)
        pooled = np.zeros((n * k, *GRID), np.float32)
        for i in range(n):
            tm = truth["mask"][i] > 0.5
            y0, x0, y1, x1 = truth["crop"][i]
            for j in range(k):
                out["mse"][i, j] = metrics.frame_mse(comp[i, j], truth["rgb"][i])
                out["mask_mse"][i, j], out["spill"][i, j] = metrics.mask_metrics(mask[i, j], truth["mask"][i])
                pooled[i * k + j] = metrics.pool_mask(mask[i, j])
                if windows[i].regions:
                    out["region_mse"][i, j] = metrics.frame_mse(rgb[i, j][tm], truth["rgb"][i][tm])
                    out["region_lpips"][i, j] = metrics.lpips_images(
                        model, rgb[i, j][None, y0:y1, x0:x1], truth["rgb"][i][None, y0:y1, x0:x1], 1, device)[0]
        flat_g = np.repeat(truth["grid"], k, axis=0)
        out["grid_lpips"] = metrics.lpips_per_sample(model, pooled, flat_g, batch, device).reshape(n, k)
        out["grid_mse"] = metrics.mse_per_sample(pooled, flat_g).reshape(n, k)
        return out

    def validation_mse(self, decoded, truth) -> float:
        """Full-frame MSE of the composited samples against the masked truth, on the [0, 1] scale."""
        comp = frames.composite(decoded["rgb"], decoded["mask"])
        return float(np.mean([metrics.frame_mse(c, t) for c, t in zip(comp, truth["rgb"])]))

    def panel_previous(self, truth, i, window):
        return truth["f8"][i] / 255

    def panel_truth(self, truth, i) -> list:
        return [truth["rgb"][i] / 255, truth["mask"][i]]

    def panel_sample(self, samples, i, j) -> list:
        return [frames.composite(samples["rgb"][i, j], samples["mask"][i, j]) / 255, samples["mask"][i, j]]

    def panel_title(self, pw, i, j) -> str:
        rl = pw["region_lpips"][i, j]
        rl = "n/a" if np.isnan(rl) else f"{rl:.3f}"
        return f"sample {j + 1}\nRegion LPIPS {rl}\nMask MSE {pw['mask_mse'][i, j]:.4f}"


def _map_rgb(m: np.ndarray) -> np.ndarray:
    """(..., H, W) float in [0, 1] -> (..., H, W, 3) uint8 grayscale for LPIPS."""
    u = np.rint(np.clip(np.asarray(m, np.float32), 0, 1) * 255).astype(np.uint8)
    return np.repeat(u[..., None], 3, axis=-1)


class ChangeMapPixelsTarget:
    """Change Map (pixels) at 224x400, 1 channel drawn from the verified Change Map image index and folded 8x8 into
    (28, 50, 64). Scored with full-frame LPIPS at 224x400, Region LPIPS on the padded crop, MSE and Spill."""
    name = "change_map_pixels"
    channels = frames.BLOCK * frames.BLOCK
    upscale = 1
    sample_keys = ("maps",)
    metrics = ("lpips", "region_lpips", "mse", "spill")
    compare_metrics = ("lpips", "region_lpips", "mse")
    higher_better = frozenset()
    floor_names = ("all_black_floor",)
    panel_rows = 1

    @property
    def shape(self) -> tuple:
        return (*GRID, self.channels)

    def tensors(self, tier, split, windows) -> np.ndarray:
        """(n, 28, 50, 64) float32 in [-1, 1]."""
        regs = load_index(tier, split, windows)
        return np.stack([frames.map_to_model_range(frames.change_mask(r)) for r in regs])

    def truths(self, tier, split, windows, verdict: bool = False) -> dict:
        """maps (n, 224, 400) float32 in {0, 1}, crop (n, 4) int64 Region LPIPS box (zeros for No Change)."""
        regs = load_index(tier, split, windows, verdict)
        maps = np.stack([frames.change_mask(r) for r in regs])
        crop = np.array([frames.crop_box(r) if r else (0, 0, 0, 0) for r in regs], np.int64)
        return {"maps": maps, "crop": crop}

    def decode(self, x) -> dict:
        return {"maps": frames.map_from_model_range(x)}

    def floors(self, tier, split, windows, truth=None) -> dict:
        return {"all_black_floor": {"maps": np.zeros((len(windows), 1, frames.H, frames.W), np.float32)}}

    def score_samples(self, pred, truth, windows, model, batch, device) -> dict:
        maps = pred["maps"]
        n, k = maps.shape[:2]
        flat_p = _map_rgb(maps.reshape(n * k, frames.H, frames.W))
        flat_t = _map_rgb(np.repeat(truth["maps"], k, axis=0))
        out = {m: np.full((n, k), np.nan) for m in self.metrics}
        out["lpips"] = metrics.lpips_images(model, flat_p, flat_t, batch, device).reshape(n, k)
        for i in range(n):
            y0, x0, y1, x1 = truth["crop"][i]
            t_rgb = _map_rgb(truth["maps"][i])
            for j in range(k):
                out["mse"][i, j], out["spill"][i, j] = metrics.mask_metrics(maps[i, j], truth["maps"][i])
                if windows[i].regions:
                    p_rgb = _map_rgb(maps[i, j])
                    out["region_lpips"][i, j] = metrics.lpips_images(
                        model, p_rgb[None, y0:y1, x0:x1], t_rgb[None, y0:y1, x0:x1], 1, device)[0]
        return out

    def validation_mse(self, decoded, truth) -> float:
        return float(np.mean((decoded["maps"] - truth["maps"]) ** 2))

    def panel_previous(self, truth, i, window):
        frame = window.previous_frame
        if frame:
            from pathlib import Path
            if Path(frame).is_file():
                import matplotlib.pyplot as plt
                return plt.imread(frame)
        return None

    def panel_truth(self, truth, i) -> list:
        return [truth["maps"][i]]

    def panel_sample(self, samples, i, j) -> list:
        return [samples["maps"][i, j]]

    def panel_title(self, pw, i, j) -> str:
        rl = pw["region_lpips"][i, j]
        rl = "n/a" if np.isnan(rl) else f"{rl:.3f}"
        return f"sample {j + 1}\nLPIPS {pw['lpips'][i, j]:.3f}\nRegion {rl}"


TARGETS = {t.name: t for t in (GridMapTarget(), MaskedFrameTarget(), ChangeMapPixelsTarget())}


def get(name: str):
    if name not in TARGETS:
        raise ValueError(f"unknown target {name!r}: choose from {tuple(TARGETS)}")
    return TARGETS[name]


def require_same_target(expected: str, found: str | None, what) -> None:
    """Refuse a run or samples file made for another target. A missing record means a run made before targets
    existed, which can only be a Grid Map run."""
    found = found or "grid_map"
    if found != expected:
        raise ValueError(f"{what} was made for target {found!r}, not {expected!r} (pass --target {found})")
