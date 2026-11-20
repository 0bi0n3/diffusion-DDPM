"""Stability gate: read a run's training log and pass or fail it on collapse signatures.

Collapse signatures:
  - training loss > 10x its running minimum (or non-finite)
  - pre-clip gradient norm > 100 (or non-finite)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

MAX_LOSS_RATIO = 10.0
MAX_GRAD_NORM = 100.0


@dataclass(frozen=True)
class GateResult:
    ok: bool
    steps: int
    first_bad_step: int | None
    reason: str


def _log_path(run: Path) -> Path:
    p = Path(run)
    return p if p.is_file() else p / "log.json"


def check(run: Path, max_loss_ratio: float = MAX_LOSS_RATIO, max_grad_norm: float = MAX_GRAD_NORM) -> GateResult:
    path = _log_path(run)
    if not path.exists():
        raise FileNotFoundError(f"training log not found: {path}")
    entries = json.loads(path.read_text()).get("train", [])
    if not entries:
        raise ValueError(f"empty training log: {path}")

    min_loss: float | None = None
    max_gn: float | None = None
    for entry in entries:
        step = entry["step"]
        loss = float(entry["loss"])
        if not math.isfinite(loss):
            return GateResult(False, len(entries), step, f"step {step}: non-finite loss ({loss})")
        if min_loss is not None and loss > max_loss_ratio * min_loss:
            return GateResult(
                False, len(entries), step,
                f"step {step}: loss {loss:.4g} > {max_loss_ratio:g}x running minimum {min_loss:.4g}",
            )
        min_loss = loss if min_loss is None else min(min_loss, loss)
        if "grad_norm" in entry and entry["grad_norm"] is not None:
            gn = float(entry["grad_norm"])
            if not math.isfinite(gn):
                return GateResult(False, len(entries), step, f"step {step}: non-finite grad_norm ({gn})")
            if gn > max_grad_norm:
                return GateResult(
                    False, len(entries), step,
                    f"step {step}: grad_norm {gn:.4g} > {max_grad_norm:g}",
                )
            max_gn = gn if max_gn is None else max(max_gn, gn)

    gn_note = f", max grad_norm {max_gn:.4g}" if max_gn is not None else ""
    return GateResult(True, len(entries), None, f"{len(entries)} steps clean (min loss {min_loss:.4g}{gn_note})")
