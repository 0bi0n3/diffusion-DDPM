"""Training that is safe for real runs: EMA, AdamW with warm-up + cosine LR, bf16 autocast + TF32 (on a GPU),
checkpoint and resume, validation every N steps, a JSON log (plus W&B), and a run record.

Files in `out`:
  last.pt        the full training state (model, ema, optimiser, noise generator, log, counters); written every
                 `ckpt_every` steps and whenever the run stops. `train` resumes from it when it exists.
  model.pt       the EMA weights alone (a plain state dict): what `sample` loads. Sampling therefore uses the EMA.
  config.json    the config; `log.json`  {"train": [{step, loss, lr}], "validation": [{step, val_loss, val_grid_mse}]};
  run_record.json  see runrecord.py.
Resume is bit-exact on CPU: a run stopped and resumed ends with the same weights as an uninterrupted run.
"""
from __future__ import annotations

import json
import math
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import siblings  # noqa: F401
from latentdit import transport  # noqa: E402

from . import runrecord
from .data import GRID, from_model_range, load_context, load_windows, to_model_range
from .ema import EMA
from .model import build_model, n_params, uses_context
from .wandb_log import PROJECT, RunLog

MIN_LR_RATIO = 0.01
# config keys that must match for a resume (the device and the tier's path may change between jobs)
RESUME_IGNORES = ("device", "tier")


@dataclass
class TrainOptions:
    ema_decay: float = 0.9999
    warmup: int = 500
    amp: str = "auto"                 # auto: bf16 on a CUDA device that has it; bf16; off
    val_every: int = 1000
    val_n: int = 100                  # fixed seeded val subset
    val_sample_steps: int = 10        # Euler steps of the few EMA samples scored by Grid Map pixel MSE
    val_samples: int = 4
    ckpt_every: int = 1000
    log_every: int = 10               # W&B only; log.json has every step
    resume: str = "auto"              # auto | never | require
    stop_after: int | None = None     # stop cleanly after this many steps in this job
    time_limit: float | None = None   # stop cleanly after this many seconds in this job
    wandb_mode: str = "off"           # off | online | offline
    wandb_project: str = PROJECT
    run_name: str | None = None


def add_arguments(t) -> None:
    d = TrainOptions()
    t.add_argument("--ema-decay", type=float, default=d.ema_decay)
    t.add_argument("--warmup", type=int, default=d.warmup, help="LR warm-up steps")
    t.add_argument("--amp", choices=("auto", "bf16", "off"), default=d.amp)
    t.add_argument("--val-every", type=int, default=d.val_every)
    t.add_argument("--val-n", type=int, default=d.val_n)
    t.add_argument("--val-sample-steps", type=int, default=d.val_sample_steps)
    t.add_argument("--ckpt-every", type=int, default=d.ckpt_every)
    t.add_argument("--log-every", type=int, default=d.log_every)
    t.add_argument("--resume", choices=("auto", "never", "require"), default=d.resume)
    t.add_argument("--stop-after", type=int, default=None, help="stop cleanly after N steps of this job")
    t.add_argument("--time-limit", type=float, default=None, help="stop cleanly after this many seconds of this job")
    t.add_argument("--wandb-mode", choices=("off", "online", "offline"), default=d.wandb_mode)
    t.add_argument("--wandb-project", default=d.wandb_project)
    t.add_argument("--run-name", default=None)


def options_from(a) -> TrainOptions:
    return TrainOptions(a.ema_decay, a.warmup, a.amp, a.val_every, a.val_n, a.val_sample_steps, ckpt_every=a.ckpt_every,
                        log_every=a.log_every, resume=a.resume, stop_after=a.stop_after, time_limit=a.time_limit,
                        wandb_mode=a.wandb_mode, wandb_project=a.wandb_project, run_name=a.run_name)


def lr_at(step: int, lr: float, warmup: int, steps: int) -> float:
    """LR of the 1-based `step`: linear warm-up to `lr`, then cosine decay to MIN_LR_RATIO * lr."""
    if warmup > 0 and step <= warmup:
        return lr * step / warmup
    span = max(steps - warmup, 1)
    progress = min((step - warmup) / span, 1.0)
    return lr * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * progress)))


def _save(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _ctx(ctx, idx):
    """The (f8, caption) context of the rows `idx`, or None for the Unconditional arm."""
    return (ctx[0][idx], ctx[1][idx]) if ctx else None


@torch.no_grad()
def validate(ema_model, val_x, val_ctx, val_ids, val_maps, shift, seed, opts, device, autocast_cfg) -> dict:
    """Flow-matching loss of the EMA on the fixed val subset (fixed noise and times), and the Grid Map pixel MSE of a
    few EMA samples against their true Grid Maps. No boxes and no IoU."""
    ema_model.eval()
    g = torch.Generator().manual_seed(seed + 1)
    total = 0.0
    for i in range(0, len(val_x), 32):
        x0 = val_x[i:i + 32]
        t, xt, ut = transport.training_batch(x0, shift, g)
        with torch.autocast(**autocast_cfg):
            pred = ema_model(xt, t, _ctx(val_ctx, slice(i, i + 32)))
        total += float(transport.velocity_loss(pred.float(), ut)) * len(x0)
    n = min(opts.val_samples, len(val_ids))
    noise = transport.window_noise(val_ids[:n], (*GRID, 1), seed).to(device)
    x = transport.sample_euler(ema_model, noise, _ctx(val_ctx, slice(0, n)), opts.val_sample_steps, shift)
    mse = float(np.mean((from_model_range(x.cpu().numpy()) - val_maps[:n]) ** 2))
    return {"val_loss": total / len(val_x), "val_grid_mse": mse}


def train(tier, arm: str, steps: int, out, batch: int = 64, lr: float = 1e-4, hidden: int = 384, depth: int = 12,
          heads: int = 6, seed: int = transport.SEED, device: str = "cpu", options: TrainOptions | None = None) -> dict:
    opts = options or TrainOptions()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    windows = load_windows(tier, "train")
    config = {"arm": arm, "steps": steps, "batch": batch, "lr": lr, "hidden": hidden, "depth": depth, "heads": heads,
              "seed": seed, "shift": transport.time_shift((*GRID, 1)), "tier": str(tier),
              "n_train_windows": len(windows), "device": device, "ema_decay": opts.ema_decay, "warmup": opts.warmup,
              "amp": opts.amp, "val_every": opts.val_every, "val_n": opts.val_n,
              "val_sample_steps": opts.val_sample_steps}
    shift = config["shift"]
    last = out / "last.pt"
    if last.exists() and opts.resume == "never":
        raise ValueError(f"{last} exists and --resume never was given")
    if not last.exists() and opts.resume == "require":
        raise ValueError(f"--resume require, but {last} does not exist")

    torch.manual_seed(seed)
    model = build_model(arm, hidden, depth, heads).to(device)
    config["n_params"] = n_params(model)
    ema = EMA(model, opts.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    step, log, wall_before, peak_before, jobs = 0, {"train": [], "validation": []}, 0.0, {}, []
    if last.exists():
        st = torch.load(last, map_location="cpu")
        old = {k: v for k, v in st["config"].items() if k not in RESUME_IGNORES}
        new = {k: v for k, v in config.items() if k not in RESUME_IGNORES}
        if old != new:
            diff = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
            raise ValueError(f"cannot resume {last}: config differs in {diff}")
        model.load_state_dict(st["model"])
        ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"])
        g.set_state(st["gen"])
        step, log, wall_before, peak_before, jobs = st["step"], st["log"], st["wall_seconds"], st["peak"], st["jobs"]

    use_cuda = device.startswith("cuda")
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    bf16 = opts.amp == "bf16" or (opts.amp == "auto" and use_cuda and torch.cuda.is_bf16_supported())
    ac = dict(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=bf16)

    data = torch.from_numpy(np.stack([to_model_range(w.grid_map) for w in windows])).to(device)
    ctx = [torch.from_numpy(a).to(device) for a in load_context(tier, "train", [w.id for w in windows])] \
        if uses_context(arm) else None
    val = load_windows(tier, "val")
    keep = sorted(np.random.default_rng(seed).permutation(len(val))[:opts.val_n])
    val = [val[i] for i in keep]
    val_x = torch.from_numpy(np.stack([to_model_range(w.grid_map) for w in val])).to(device)
    val_ids, val_maps = [w.id for w in val], np.stack([w.grid_map for w in val])
    val_ctx = [torch.from_numpy(a).to(device) for a in load_context(tier, "val", val_ids)] if uses_context(arm) else None

    wb = RunLog.open(opts.wandb_mode, opts.wandb_project, opts.run_name or out.name, config, out)
    stop = {"flag": False}
    try:
        old_handler = signal.signal(signal.SIGUSR1, lambda *_: stop.update(flag=True))
    except ValueError:                         # not the main thread
        old_handler = None
    t0, start_step = time.monotonic(), step

    def wall():
        return wall_before + time.monotonic() - t0

    def checkpoint(status: str) -> None:
        peak = {k: max(v, peak_before.get(k, 0)) for k, v in runrecord.peak_memory(device).items()}
        jobs_now = jobs + [{"from_step": start_step, "to_step": step, "wall_seconds": time.monotonic() - t0,
                            "status": status}]
        _save(last, {"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                     "gen": g.get_state(), "step": step, "log": log, "config": config, "wall_seconds": wall(),
                     "peak": peak, "jobs": jobs_now})
        torch.save(ema.module.state_dict(), out / "model.pt")
        (out / "config.json").write_text(json.dumps(config, indent=1) + "\n")
        (out / "log.json").write_text(json.dumps(log) + "\n")
        runrecord.write(out, runrecord.make(tier, config, status, step, wall(), peak, jobs_now))

    try:
        model.train()
        while step < steps:
            if stop["flag"] or (opts.stop_after is not None and step - start_step >= opts.stop_after) or \
                    (opts.time_limit is not None and time.monotonic() - t0 >= opts.time_limit):
                break
            step_lr = lr_at(step + 1, lr, opts.warmup, steps)
            for grp in opt.param_groups:
                grp["lr"] = step_lr
            idx = torch.randint(len(data), (batch,), generator=g).to(device)
            x0 = data[idx]
            t, xt, ut = transport.training_batch(x0, shift, g)
            with torch.autocast(**ac):
                pred = model(xt, t, _ctx(ctx, idx))
            loss = transport.velocity_loss(pred.float(), ut)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ema.update(model)
            step += 1
            log["train"].append({"step": step, "loss": float(loss), "lr": step_lr})
            if step % opts.log_every == 0 or step == steps:
                wb.train(step, float(loss), step_lr)
            if (opts.val_every and step % opts.val_every == 0) or step == steps:
                v = validate(ema.module, val_x, val_ctx, val_ids, val_maps, shift, seed, opts, device, ac)
                log["validation"].append({"step": step, **v})
                wb.validation(step, v["val_loss"], v["val_grid_mse"])
            if opts.ckpt_every and step % opts.ckpt_every == 0 and step < steps:
                checkpoint("running")
        status = "finished" if step >= steps else "interrupted"
        checkpoint(status)
    finally:
        wb.close()
        if old_handler is not None:
            signal.signal(signal.SIGUSR1, old_handler)
    return {**config, "status": status}
