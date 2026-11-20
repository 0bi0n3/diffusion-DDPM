"""AISurrey job script for one Conditioning Arm: render the sbatch file, build the sbatch command, keep the ledger.

    python -m changemap.cluster render --arm A --sha SHA --tier DIR --out DIR --steps N [--time HH:MM:SS] [--mem 16G]
                                       [--script PATH]        write the sbatch file (runs nothing)
    python -m changemap.cluster submit ... [--dependency afterany:JOBID] [--dry-run]
                                                              sbatch it and add a ledger row; --dry-run only prints
    python -m changemap.cluster render-score | submit-score --sha SHA --tier DIR --runs DIR --arms A B ... --samples DIR
                                       --out DIR --lpips-dir DIR --latent-stats NPZ [--dependency afterok:ID1:ID2...]
                                                              the sample + score job: after all train jobs finish, `changemap
                                                              sample` per arm (k = 8, 50 Euler steps), then one `changemap score`
                                                              with the real DINOv3 encoder on the GPU

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
                  cpus: int = 3, home: str = HOME, train_args: str = "", job_name: str | None = None) -> str:
    lc.submit_command("train", sha, {}, time)                  # reuse its sha and 24 h checks (raises ValueError)
    limit = lc.seconds(time) - SELF_STOP_MARGIN
    if limit <= 0:
        raise ValueError(f"--time {time} leaves no room for the {SELF_STOP_MARGIN} s stop margin")
    jname = job_name or f"changemap-{arm}"
    rname = job_name or arm
    q = shlex.quote
    return f"""#!/bin/bash
# Change Map Diffusion: one Conditioning Arm ({arm}) on one {PARTITION} GPU. Rendered by `python -m changemap.cluster render`.
# Resumes from {out}/last.pt, so resubmitting (or chaining with --dependency afterany:<jobid>) continues the run until
# {out}/run_record.json says "finished". Stops itself {SELF_STOP_MARGIN} s before the time limit; USR1 comes {USR1_WARNING} s before.
# --mem is right-sized: tier_1k's Grid Maps are tiny; raise it when the Previous Frame tokens (#4) are loaded.
#SBATCH --job-name={jname}
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
export CHANGE_REGIONS_REPO=${{CHANGE_REGIONS_REPO:-$P/siblings/change_regions}}
export PYTHONPATH=$CODE:{WANDB_DEPS}
export CHANGEMAP_COMMIT=$SHA
export WANDB_DIR=$OUT

echo "job $SLURM_JOB_ID sha $SHA node $(hostname) arm {arm} tier $TIER out $OUT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

cd "$OUT"
# shellcheck disable=SC2086
apptainer exec --nv "$SIF" python -m changemap.cli train --tier "$TIER" --arm {arm} --steps {steps} --out "$OUT" \\
    --device cuda --resume auto --time-limit {limit} --wandb-mode online --run-name {rname} {train_args} &
PID=$!
trap 'kill -USR1 $PID' USR1 TERM       # time-limit warning: the run writes last.pt and returns; a re-run resumes
RC=0
while kill -0 $PID 2>/dev/null; do     # a trapped signal interrupts wait; loop until it really exits
    wait $PID; RC=$?
done
cp "$P/logs/${{SLURM_JOB_NAME}}.${{SLURM_JOB_ID}}".* "$OUT/" 2>/dev/null
exit $RC
"""


def render_score_sbatch(sha: str, tier: str, runs: str, arms: list, samples: str, out: str, lpips_dir: str,
                        latent_stats: str, time: str = "06:00:00", mem: str = "24G", cpus: int = 4, home: str = HOME,
                        k: int = 8, sample_steps: int = 50, seed: int = 35, split: str = "val") -> str:
    """One GPU job: wait for every arm's run to be finished, sample k per Window per arm, score all arms plus the floors
    with the DINOv3 token metrics. Validation split only (`--split val`; never the test split)."""
    lc.submit_command("train", sha, {}, time)
    if split != "val":
        raise ValueError("the experiment scores the validation split only")
    q = shlex.quote
    arm_list = " ".join(q(a) for a in arms)
    return f"""#!/bin/bash
# Change Map Diffusion: sample + score for {len(arms)} Conditioning Arms on one {PARTITION} GPU. Rendered by
# `python -m changemap.cluster render-score`. Submit it with --dependency afterok:<every train job id> (the last job of
# each resume chain). Re-running skips arms whose samples file exists, so it can be resubmitted after a failure.
#SBATCH --job-name=changemap-score
#SBATCH --partition={PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={home}/logs/%x.%j.out
#SBATCH --error={home}/logs/%x.%j.err

set -uo pipefail
P={q(home)}
SHA={sha}
CODE=$P/code/$SHA
TIER={q(tier)}
RUNS={q(runs)}
SAMPLES={q(samples)}
OUT={q(out)}
LPIPS={q(lpips_dir)}
STATS={q(latent_stats)}
SIF={SIF}
SIF_SHA256={SIF_SHA256}

mkdir -p "$P/logs" "$SAMPLES" "$OUT"
[ -e "$CODE/changemap/cli.py" ] || {{ echo "code not staged: $CODE"; exit 2; }}
[ -e "$TIER/val_targets.jsonl" ] || {{ echo "tier not staged: $TIER"; exit 2; }}
[ -e "$LPIPS/vgg.pth" ] && [ -e "$STATS" ] || {{ echo "LPIPS weights or latent stats not staged"; exit 2; }}
ls "$P"/hf/hub/models--facebook--dinov3-vitb16-pretrain-lvd1689m/snapshots/*/model.safetensors >/dev/null 2>&1 \
    || {{ echo "DINOv3 weights not staged under $P/hf/hub"; exit 2; }}
[ -e "$SIF" ] || {{ echo "missing image $SIF"; exit 2; }}
[ "$(sha256sum "$SIF" | cut -d' ' -f1)" = "$SIF_SHA256" ] || {{ echo "image checksum differs from the pinned one"; exit 2; }}

# Host conda/python variables must not leak into the container.
unset PYTHONHOME PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE
export PYTHONNOUSERSITE=1
export HF_HOME=$P/hf HF_HUB_OFFLINE=1 TORCH_COMPILE_DISABLE=1 USE_TF=0
export CHANGEMAP_LATENTDIT_ROOT=${{CHANGEMAP_LATENTDIT_ROOT:-$P/siblings/latentdit}}
export CHANGEMAP_DIT2VLM_ROOT=${{CHANGEMAP_DIT2VLM_ROOT:-$P/siblings/dit2vlm}}
export CHANGE_REGIONS_REPO=${{CHANGE_REGIONS_REPO:-$P/siblings/change_regions}}
export PYTHONPATH=$CODE
export CHANGEMAP_COMMIT=$SHA

echo "job $SLURM_JOB_ID sha $SHA node $(hostname) arms {' '.join(arms)} tier $TIER out $OUT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

for ARM in {arm_list}; do
    apptainer exec "$SIF" python -c "import json,sys; sys.exit(json.load(open(sys.argv[1]))['status'] != 'finished')" \
        "$RUNS/$ARM/run_record.json" || {{ echo "run $RUNS/$ARM is not finished"; exit 3; }}
done

cd "$OUT"
for ARM in {arm_list}; do
    [ -e "$SAMPLES/$ARM.npz" ] && continue
    apptainer exec --nv "$SIF" python -m changemap.cli sample --run "$RUNS/$ARM" --tier "$TIER" --split {split} \
        --k {k} --steps {sample_steps} --seed {seed} --out "$SAMPLES" --device cuda || exit $?
done
apptainer exec --nv "$SIF" python -m changemap.cli score --tier "$TIER" --split {split} --samples "$SAMPLES" \
    --out "$OUT" --lpips-dir "$LPIPS" --device cuda --encoder dinov3 --encoder-device cuda --latent-stats "$STATS"
RC=$?
cp "$P/logs/${{SLURM_JOB_NAME}}.${{SLURM_JOB_ID}}".* "$OUT/" 2>/dev/null
exit $RC
"""


def render_map_latents_sbatch(sha: str, tier: str, splits=("train", "val"), time: str = "02:00:00", mem: str = "64G",
                              cpus: int = 4, home: str = HOME, encoder: str = "dinov3") -> str:
    """One GPU job: pre-build per-split Map Latent caches and training stats under `<tier>/token_cache/`."""
    lc.submit_command("train", sha, {}, time)
    if "test" in splits:
        raise ValueError("map-latents job encodes train/val only; never the test split")
    q = shlex.quote
    split_args = " ".join(q(s) for s in splits)
    return f"""#!/bin/bash
# Change Map Diffusion: pre-build Map Latent token caches and train stats on one {PARTITION} GPU.
# Rendered by `python -m changemap.cluster render-map-latents`.
#SBATCH --job-name=changemap-map-latents
#SBATCH --partition={PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={home}/logs/%x.%j.out
#SBATCH --error={home}/logs/%x.%j.err

set -uo pipefail
P={q(home)}
SHA={sha}
CODE=$P/code/$SHA
TIER={q(tier)}
SIF={SIF}
SIF_SHA256={SIF_SHA256}

mkdir -p "$P/logs"
[ -e "$CODE/changemap/cli.py" ] || {{ echo "code not staged: $CODE"; exit 2; }}
[ -e "$TIER/train_targets.jsonl" ] && [ -e "$TIER/train_index.jsonl" ] || {{ echo "tier not staged: $TIER"; exit 2; }}
ls "$P"/hf/hub/models--facebook--dinov3-vitb16-pretrain-lvd1689m/snapshots/*/model.safetensors >/dev/null 2>&1 \\
    || {{ echo "DINOv3 weights not staged under $P/hf/hub"; exit 2; }}
[ -e "$SIF" ] || {{ echo "missing image $SIF"; exit 2; }}
[ "$(sha256sum "$SIF" | cut -d' ' -f1)" = "$SIF_SHA256" ] || {{ echo "image checksum differs from the pinned one"; exit 2; }}

unset PYTHONHOME PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE
export PYTHONNOUSERSITE=1
export HF_HOME=$P/hf HF_HUB_OFFLINE=1 TORCH_COMPILE_DISABLE=1 USE_TF=0
export CHANGEMAP_LATENTDIT_ROOT=${{CHANGEMAP_LATENTDIT_ROOT:-$P/siblings/latentdit}}
export CHANGEMAP_DIT2VLM_ROOT=${{CHANGEMAP_DIT2VLM_ROOT:-$P/siblings/dit2vlm}}
export CHANGE_REGIONS_REPO=${{CHANGE_REGIONS_REPO:-$P/siblings/change_regions}}
export PYTHONPATH=$CODE
export CHANGEMAP_COMMIT=$SHA

echo "job $SLURM_JOB_ID sha $SHA node $(hostname) tier $TIER splits {' '.join(splits)}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

apptainer exec --nv "$SIF" python -m changemap.cli map-latents --tier "$TIER" --splits {split_args} \\
    --encoder {q(encoder)} --device cuda
"""


def render_two_frame_score_sbatch(sha: str, tier: str, pixel_run: str, latent_run: str, samples: str, out: str,
                                  lpips_dir: str, arm: str = "frame_caption", time: str = "06:00:00",
                                  mem: str = "64G", cpus: int = 4, home: str = HOME, k: int = 8,
                                  sample_steps: int = 50, seed: int = 35, split: str = "val") -> str:
    """One GPU job: verify both two-frame runs finished, sample change_map_pixels and map_latent (k=8, 50 Euler steps),
    and score both together in a single paired report (pixel + latent space, Window-resampled bootstrap)."""
    lc.submit_command("train", sha, {}, time)
    if split != "val":
        raise ValueError("the experiment scores the validation split only")
    q = shlex.quote
    return f"""#!/bin/bash
# Change Map Diffusion: paired two-frame sample + score (change_map_pixels + map_latent) on one {PARTITION} GPU.
# Rendered by `python -m changemap.cluster render-two-frame-score`.
#SBATCH --job-name=changemap-tf-score
#SBATCH --partition={PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={home}/logs/%x.%j.out
#SBATCH --error={home}/logs/%x.%j.err

set -uo pipefail
P={q(home)}
SHA={sha}
CODE=$P/code/$SHA
TIER={q(tier)}
PIXEL_RUN={q(pixel_run)}
LATENT_RUN={q(latent_run)}
SAMPLES={q(samples)}
OUT={q(out)}
LPIPS={q(lpips_dir)}
ARM={q(arm)}
SIF={SIF}
SIF_SHA256={SIF_SHA256}

mkdir -p "$P/logs" "$SAMPLES/pixel" "$SAMPLES/latent" "$OUT"
[ -e "$CODE/changemap/cli.py" ] || {{ echo "code not staged: $CODE"; exit 2; }}
[ -e "$TIER/val_targets.jsonl" ] && [ -e "$TIER/val_index.jsonl" ] || {{ echo "tier not staged: $TIER"; exit 2; }}
[ -e "$LPIPS/vgg.pth" ] || {{ echo "LPIPS weights not staged: $LPIPS"; exit 2; }}
ls "$P"/hf/hub/models--facebook--dinov3-vitb16-pretrain-lvd1689m/snapshots/*/model.safetensors >/dev/null 2>&1 \\
    || {{ echo "DINOv3 weights not staged under $P/hf/hub"; exit 2; }}
[ -e "$SIF" ] || {{ echo "missing image $SIF"; exit 2; }}
[ "$(sha256sum "$SIF" | cut -d' ' -f1)" = "$SIF_SHA256" ] || {{ echo "image checksum differs from the pinned one"; exit 2; }}

unset PYTHONHOME PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE
export PYTHONNOUSERSITE=1
export HF_HOME=$P/hf HF_HUB_OFFLINE=1 TORCH_COMPILE_DISABLE=1 USE_TF=0
export CHANGEMAP_LATENTDIT_ROOT=${{CHANGEMAP_LATENTDIT_ROOT:-$P/siblings/latentdit}}
export CHANGEMAP_DIT2VLM_ROOT=${{CHANGEMAP_DIT2VLM_ROOT:-$P/siblings/dit2vlm}}
export CHANGE_REGIONS_REPO=${{CHANGE_REGIONS_REPO:-$P/siblings/change_regions}}
export PYTHONPATH=$CODE
export CHANGEMAP_COMMIT=$SHA

echo "job $SLURM_JOB_ID sha $SHA node $(hostname) arm $ARM tier $TIER out $OUT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

for RUN in "$PIXEL_RUN" "$LATENT_RUN"; do
    apptainer exec "$SIF" python -c "import json,sys; sys.exit(json.load(open(sys.argv[1]))['status'] != 'finished')" \\
        "$RUN/run_record.json" || {{ echo "run $RUN is not finished"; exit 3; }}
done

cd "$OUT"
if [ ! -e "$SAMPLES/pixel/$ARM.npz" ]; then
    apptainer exec --nv "$SIF" python -m changemap.cli sample --target change_map_pixels --run "$PIXEL_RUN" \\
        --tier "$TIER" --split {split} --k {k} --steps {sample_steps} --seed {seed} \\
        --out "$SAMPLES/pixel" --device cuda || exit $?
fi
if [ ! -e "$SAMPLES/latent/$ARM.npz" ]; then
    apptainer exec --nv "$SIF" python -m changemap.cli sample --target map_latent --run "$LATENT_RUN" \\
        --tier "$TIER" --split {split} --k {k} --steps {sample_steps} --seed {seed} \\
        --out "$SAMPLES/latent" --device cuda || exit $?
fi
apptainer exec --nv "$SIF" python -m changemap.cli score --target change_map_pixels --tier "$TIER" --split {split} \\
    --samples "$SAMPLES/pixel/$ARM.npz" --latent-samples "$SAMPLES/latent/$ARM.npz" \\
    --out "$OUT" --lpips-dir "$LPIPS" --device cuda --encoder dinov3 --encoder-device cuda
RC=$?
cp "$P/logs/${{SLURM_JOB_NAME}}.${{SLURM_JOB_ID}}".* "$OUT/" 2>/dev/null
exit $RC
"""


def _add_two_frame_score_args(s) -> None:
    s.add_argument("--sha", required=True)
    s.add_argument("--tier", required=True)
    s.add_argument("--pixel-run", required=True)
    s.add_argument("--latent-run", required=True)
    s.add_argument("--arm", default="frame_caption")
    s.add_argument("--samples", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--lpips-dir", required=True)
    s.add_argument("--time", default="06:00:00")
    s.add_argument("--mem", default="64G")
    s.add_argument("--cpus", type=int, default=4)
    s.add_argument("--k", type=int, default=8)
    s.add_argument("--sample-steps", type=int, default=50)
    s.add_argument("--seed", type=int, default=35)


def _add_map_latents_args(s) -> None:
    s.add_argument("--sha", required=True)
    s.add_argument("--tier", required=True)
    s.add_argument("--splits", nargs="+", choices=("train", "val"), default=["train", "val"])
    s.add_argument("--encoder", default="dinov3")
    s.add_argument("--time", default="02:00:00")
    s.add_argument("--mem", default="64G")
    s.add_argument("--cpus", type=int, default=4)


def _add_score_args(s) -> None:
    s.add_argument("--sha", required=True)
    s.add_argument("--tier", required=True)
    s.add_argument("--runs", required=True, help="folder holding one finished run per arm: <runs>/<arm>")
    s.add_argument("--arms", nargs="+", required=True)
    s.add_argument("--samples", required=True, help="where the samples files are written")
    s.add_argument("--out", required=True, help="where report.json, report.md and panels are written")
    s.add_argument("--lpips-dir", required=True)
    s.add_argument("--latent-stats", required=True)
    s.add_argument("--time", default="06:00:00")
    s.add_argument("--mem", default="24G")
    s.add_argument("--cpus", type=int, default=4)
    s.add_argument("--k", type=int, default=8)
    s.add_argument("--sample-steps", type=int, default=50)
    s.add_argument("--seed", type=int, default=35)


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
    s.add_argument("--job-name", default=None, help="override Slurm job name, script stem and W&B run name")


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
    rs = sub.add_parser("render-score")
    _add_score_args(rs)
    rs.add_argument("--script", type=Path, help="where to write the sbatch file (default stdout)")
    ss = sub.add_parser("submit-score")
    _add_score_args(ss)
    ss.add_argument("--dependency", metavar="afterok:ID1:ID2...")
    ss.add_argument("--note", default="")
    ss.add_argument("--dry-run", action="store_true", help="print the sbatch command; render and submit nothing")
    rml = sub.add_parser("render-map-latents")
    _add_map_latents_args(rml)
    rml.add_argument("--script", type=Path, help="where to write the sbatch file (default stdout)")
    sml = sub.add_parser("submit-map-latents")
    _add_map_latents_args(sml)
    sml.add_argument("--note", default="")
    sml.add_argument("--dry-run", action="store_true", help="print the sbatch command; render and submit nothing")
    rtfs = sub.add_parser("render-two-frame-score")
    _add_two_frame_score_args(rtfs)
    rtfs.add_argument("--script", type=Path, help="where to write the sbatch file (default stdout)")
    stfs = sub.add_parser("submit-two-frame-score")
    _add_two_frame_score_args(stfs)
    stfs.add_argument("--dependency", metavar="afterok:ID1:ID2...")
    stfs.add_argument("--note", default="")
    stfs.add_argument("--dry-run", action="store_true", help="print the sbatch command; render and submit nothing")
    a = p.parse_args(argv)
    if a.cmd in ("render-score", "submit-score"):
        return _score_main(a, p)
    if a.cmd in ("render-two-frame-score", "submit-two-frame-score"):
        return _two_frame_score_main(a, p)
    if a.cmd in ("render-map-latents", "submit-map-latents"):
        return _map_latents_main(a, p)
    try:
        text = render_sbatch(a.arm, a.sha, a.tier, a.out, a.steps, a.time, a.mem, a.cpus, a.home, a.train_args,
                             job_name=a.job_name)
        jname = a.job_name or f"changemap-{a.arm}"
        stem = a.job_name or f"train_{a.arm}"
        script = Path(a.home) / "scripts" / f"{stem}.sbatch" if a.cmd == "submit" else a.script
        if a.cmd == "submit":
            cmd = lc.submit_command("train", a.sha, {}, a.time, a.home, a.mem, a.dependency)[:-1]
            cmd = [c for c in cmd if not c.startswith("--job-name=")] + [f"--job-name={jname}", str(script)]
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
              lc.ledger_row(job_id, jname, PARTITION, 1, a.sha, lc.IMAGE_TAG, lc._now(), a.note))
    print(job_id)
    return 0


def _score_main(a, p) -> int:
    import re
    try:
        text = render_score_sbatch(a.sha, a.tier, a.runs, a.arms, a.samples, a.out, a.lpips_dir, a.latent_stats, a.time,
                                   a.mem, a.cpus, a.home, a.k, a.sample_steps, a.seed)
        dep = getattr(a, "dependency", None)
        if dep is not None and not re.fullmatch(r"afterok(:\d+)+", dep):
            raise ValueError(f"--dependency must look like afterok:<jobid>[:<jobid>...], got {dep!r}")
        script = Path(a.home) / "scripts" / "score.sbatch" if a.cmd == "submit-score" else a.script
        if a.cmd == "submit-score":
            cmd = lc.submit_command("train", a.sha, {}, a.time, a.home, a.mem)[:-1]
            cmd = [c for c in cmd if not c.startswith("--job-name=")] + ["--job-name=changemap-score"]
            cmd += [f"--dependency={dep}"] if dep else []
            cmd.append(str(script))
    except ValueError as e:
        p.exit(2, f"changemap.cluster: error: {e}\n")
    if a.cmd == "render-score":
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
              lc.ledger_row(job_id, "changemap-score", PARTITION, 1, a.sha, lc.IMAGE_TAG, lc._now(), a.note))
    print(job_id)
    return 0


def _two_frame_score_main(a, p) -> int:
    import re
    try:
        text = render_two_frame_score_sbatch(
            a.sha, a.tier, a.pixel_run, a.latent_run, a.samples, a.out, a.lpips_dir, a.arm,
            a.time, a.mem, a.cpus, a.home, a.k, a.sample_steps, a.seed,
        )
        dep = getattr(a, "dependency", None)
        if dep is not None and not re.fullmatch(r"afterok(:\d+)+", dep):
            raise ValueError(f"--dependency must look like afterok:<jobid>[:<jobid>...], got {dep!r}")
        script = Path(a.home) / "scripts" / "two_frame_score.sbatch" if a.cmd == "submit-two-frame-score" else a.script
        if a.cmd == "submit-two-frame-score":
            cmd = lc.submit_command("train", a.sha, {}, a.time, a.home, a.mem)[:-1]
            cmd = [c for c in cmd if not c.startswith("--job-name=")] + ["--job-name=changemap-tf-score"]
            cmd += [f"--dependency={dep}"] if dep else []
            cmd.append(str(script))
    except ValueError as e:
        p.exit(2, f"changemap.cluster: error: {e}\n")
    if a.cmd == "render-two-frame-score":
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
              lc.ledger_row(job_id, "changemap-tf-score", PARTITION, 1, a.sha, lc.IMAGE_TAG, lc._now(), a.note))
    print(job_id)
    return 0


def _map_latents_main(a, p) -> int:
    try:
        text = render_map_latents_sbatch(a.sha, a.tier, a.splits, a.time, a.mem, a.cpus, a.home, a.encoder)
        script = Path(a.home) / "scripts" / "map_latents.sbatch" if a.cmd == "submit-map-latents" else a.script
        if a.cmd == "submit-map-latents":
            cmd = lc.submit_command("train", a.sha, {}, a.time, a.home, a.mem)[:-1]
            cmd = [c for c in cmd if not c.startswith("--job-name=")] + ["--job-name=changemap-map-latents", str(script)]
    except ValueError as e:
        p.exit(2, f"changemap.cluster: error: {e}\n")
    if a.cmd == "render-map-latents":
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
              lc.ledger_row(job_id, "changemap-map-latents", PARTITION, 1, a.sha, lc.IMAGE_TAG, lc._now(), a.note))
    print(job_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
