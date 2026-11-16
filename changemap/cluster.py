"""AISurrey job script for one Conditioning Arm: render the sbatch file, build the sbatch command, keep the ledger.

    python -m changemap.cluster render --arm A --sha SHA --tier DIR --out DIR --steps N [--time HH:MM:SS] [--mem 16G]
                                       [--script PATH]        write the sbatch file (runs nothing)
    python -m changemap.cluster submit ... [--dependency afterany:JOBID] [--dry-run]
                                                              sbatch it and add a ledger row; --dry-run only prints

Policy (from the DiT latent study's `latentdit.cluster`, which supplies the time parsing, the 24 h limit and the ledger
helpers): one GPU per arm, `--time` at most 24 h, code named by commit sha, memory right-sized. The job runs
`changemap train ... --resume auto`, which continues from `<out>/last.pt`; it stops itself (`--time-limit`) shortly before
Slurm's limit and on USR1 (sent 10 min before), writing `last.pt`. Chain jobs with `--dependency afterany:<jobid>` and the
same arguments until `<out>/run_record.json` says "finished". The user's rule: experiments run only on AISurrey.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from . import siblings  # noqa: F401
from latentdit import cluster as lc  # noqa: E402

HOME = os.environ.get("CHANGEMAP_HOME", "/mnt/fast/nobackup/users/od00383/change_map")
SELF_STOP_MARGIN = 900          # seconds before the Slurm limit at which the program stops itself
USR1_WARNING = 600
SIF = "/mnt/fast/nobackup/users/od00383/dit2vlm/containers/dit2vlm_v2.sif"        # DiT2VLM's image, read only, shared
SIF_SHA256 = "cd1de889e69ba46416554a772046322b50c2e90be71592ecd7a7faa3edff75f1"
WANDB_DEPS = "/mnt/fast/nobackup/users/od00383/dit2vlm/pydeps/wandb-0.31.0"
PARTITION = "a100"


def render_sbatch(arm: str, sha: str, tier: str, out: str, steps: int, time: str = "24:00:00", mem: str = "16G",
                  cpus: int = 3, home: str = HOME, train_args: str = "") -> str:
    lc.submit_command("train", sha, {}, time)                  # reuse its sha and 24 h checks (raises ValueError)
    limit = lc.seconds(time) - SELF_STOP_MARGIN
    if limit <= 0:
        raise ValueError(f"--time {time} leaves no room for the {SELF_STOP_MARGIN} s stop margin")
    q = shlex.quote
    return f"""#!/bin/bash
# Change Map Diffusion: one Conditioning Arm ({arm}) on one {PARTITION} GPU. Rendered by `python -m changemap.cluster render`.
# Resumes from {out}/last.pt, so resubmitting (or chaining with --dependency afterany:<jobid>) continues the run until
# {out}/run_record.json says "finished". Stops itself {SELF_STOP_MARGIN} s before the time limit; USR1 comes {USR1_WARNING} s before.
# --mem is right-sized: tier_1k's Grid Maps are tiny; raise it when the Previous Frame tokens (#4) are loaded.
#SBATCH --job-name=changemap-{arm}
#SBATCH --partition={PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={home}/logs/%x.%j.out
#SBATCH --error={home}/logs/%x.%j.err
#SBATCH --signal=B:USR1@{USR1_WARNING}

set -uo pipefail
P={q(home)}
SHA={sha}
CODE=$P/code/$SHA
TIER={q(tier)}
OUT={q(out)}
SIF={SIF}
SIF_SHA256={SIF_SHA256}

mkdir -p "$P/logs" "$OUT"
[ -e "$CODE/changemap/cli.py" ] || {{ echo "code not staged: $CODE"; exit 2; }}
[ -e "$TIER/train_targets.jsonl" ] || {{ echo "tier not staged: $TIER"; exit 2; }}
[ -e "$SIF" ] || {{ echo "missing image $SIF"; exit 2; }}
[ "$(sha256sum "$SIF" | cut -d' ' -f1)" = "$SIF_SHA256" ] || {{ echo "image checksum differs from the pinned one"; exit 2; }}

# Host conda/python variables must not leak into the container.
unset PYTHONHOME PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE
export PYTHONNOUSERSITE=1
export HF_HOME=$P/hf HF_HUB_OFFLINE=1 TORCH_COMPILE_DISABLE=1 USE_TF=0
export CHANGEMAP_LATENTDIT_ROOT=${{CHANGEMAP_LATENTDIT_ROOT:-$P/siblings/latentdit}}
export CHANGEMAP_DIT2VLM_ROOT=${{CHANGEMAP_DIT2VLM_ROOT:-$P/siblings/dit2vlm}}
export PYTHONPATH=$CODE:{WANDB_DEPS}
export CHANGEMAP_COMMIT=$SHA
export WANDB_DIR=$OUT

echo "job $SLURM_JOB_ID sha $SHA node $(hostname) arm {arm} tier $TIER out $OUT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

cd "$OUT"
# shellcheck disable=SC2086
apptainer exec --nv "$SIF" python -m changemap.cli train --tier "$TIER" --arm {arm} --steps {steps} --out "$OUT" \\
    --device cuda --resume auto --time-limit {limit} --wandb-mode online --run-name {arm} {train_args} &
PID=$!
trap 'kill -USR1 $PID' USR1 TERM       # time-limit warning: the run writes last.pt and returns; a re-run resumes
RC=0
while kill -0 $PID 2>/dev/null; do     # a trapped signal interrupts wait; loop until it really exits
    wait $PID; RC=$?
done
cp "$P/logs/${{SLURM_JOB_NAME}}.${{SLURM_JOB_ID}}".* "$OUT/" 2>/dev/null
exit $RC
"""


def _add_job_args(s) -> None:
    s.add_argument("--arm", required=True)
    s.add_argument("--sha", required=True)
    s.add_argument("--tier", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--steps", type=int, required=True)
    s.add_argument("--time", default="24:00:00")
    s.add_argument("--mem", default="16G")
    s.add_argument("--cpus", type=int, default=3)
    s.add_argument("--train-args", default="", help="extra `changemap train` flags, e.g. '--batch 64'")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="changemap.cluster")
    p.add_argument("--home", default=HOME)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    _add_job_args(r)
    r.add_argument("--script", type=Path, help="where to write the sbatch file (default stdout)")
    s = sub.add_parser("submit")
    _add_job_args(s)
    s.add_argument("--dependency", metavar="afterany:JOBID")
    s.add_argument("--note", default="")
    s.add_argument("--dry-run", action="store_true", help="print the sbatch command; render and submit nothing")
    a = p.parse_args(argv)
    try:
        text = render_sbatch(a.arm, a.sha, a.tier, a.out, a.steps, a.time, a.mem, a.cpus, a.home, a.train_args)
        script = Path(a.home) / "scripts" / f"train_{a.arm}.sbatch" if a.cmd == "submit" else a.script
        if a.cmd == "submit":
            cmd = lc.submit_command("train", a.sha, {}, a.time, a.home, a.mem, a.dependency)[:-1]
            cmd = [c for c in cmd if not c.startswith("--job-name=")] + [f"--job-name=changemap-{a.arm}", str(script)]
    except ValueError as e:
        p.exit(2, f"changemap.cluster: error: {e}\n")
    if a.cmd == "render":
        if script:
            Path(script).write_text(text)
        else:
            print(text, end="")
        return 0
    print(shlex.join(cmd))
    if a.dry_run:
        return 0
    Path(script).parent.mkdir(parents=True, exist_ok=True)
    Path(script).write_text(text)
    job_id = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip().split(";")[0]
    lc.append(Path(a.home) / "ledger.tsv",
              lc.ledger_row(job_id, f"changemap-{a.arm}", PARTITION, 1, a.sha, lc.IMAGE_TAG, lc._now(), a.note))
    print(job_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
