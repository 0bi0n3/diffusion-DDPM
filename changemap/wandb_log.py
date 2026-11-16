"""Weights & Biases logging for `train`. The only module that imports `wandb`, and only inside functions.

Adapted from the DiT latent study's latentdit/wandb_log.py (/mnt/hitchcock/scratch/oberon/DiT-experiments, commit
217a3630556fc0ca638298e6553c3898f84412a4): every call is guarded, so a missing package or a network error prints one
line and training continues. The x-axis is the metric `step` (never `wandb.log(step=...)`). `<run>/wandb_run.json`
keeps the run id and how far each stream was logged, so a resumed job continues the same W&B run and never logs a
step twice.
"""
from __future__ import annotations

import json
import math
import secrets
import string
from pathlib import Path

PROJECT = "change-map"
META = "wandb_run.json"


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


class RunLog:
    def __init__(self, run, path: Path, meta: dict):
        self.run, self.path, self.meta = run, path, meta

    @staticmethod
    def _guarded(what, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 (a missing package or a network error must not stop training)
            print(f"[wandb] {what} failed, training continues: {e!r}", flush=True)

    @classmethod
    def open(cls, mode: str, project: str, name: str, config: dict, run_dir) -> "RunLog":
        """`mode` off returns a log that does nothing; online/offline go to wandb.init(mode=...)."""
        path = Path(run_dir) / META
        meta = {"train": 0, "val": 0}
        if mode == "off":
            return cls(None, path, meta)

        def start():
            import wandb
            nonlocal meta
            if path.exists():
                meta.update(json.loads(path.read_text()))
            meta["id"] = meta.get("id") or "".join(secrets.choice(string.digits + string.ascii_lowercase)
                                                   for _ in range(8))
            run = wandb.init(project=project, name=name, config=config, id=meta["id"], resume="allow",
                             dir=str(run_dir), mode=mode)
            run.define_metric("step")
            run.define_metric("*", step_metric="step")
            return run

        run = cls._guarded("init", start)
        log = cls(run, path, meta)
        if run is not None:
            log._guarded("save run id", log._save)
        return log

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.meta))
        tmp.replace(self.path)

    def _log(self, stream: str, step: int, data: dict) -> None:
        if self.run is None or step <= self.meta[stream]:
            return

        def go():
            self.run.log({"step": step, **{k: v for k, v in data.items() if _ok(v)}})
            self.meta[stream] = step
            self._save()
        self._guarded(f"{stream} step {step}", go)

    def train(self, step: int, loss: float, lr: float) -> None:
        self._log("train", step, {"train/loss": loss, "train/lr": lr})

    def validation(self, step: int, val_loss: float, grid_mse: float) -> None:
        self._log("val", step, {"val/loss": val_loss, "val/grid_mse": grid_mse})

    def close(self) -> None:
        if self.run is not None:
            self._guarded("finish", self.run.finish)
