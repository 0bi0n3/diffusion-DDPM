"""End-to-end CLI tests for paired two-frame scoring, cluster two-frame jobs, and CPU panels (#21)."""
from __future__ import annotations

import json

from changemap.cli import main as cli

from test_change_map_pixels import SMALL, exits_nonzero, write_two_frame_tier


def test_paired_score_combines_pixel_and_map_latent_runs(tmp_path):
    tier = write_two_frame_tier(tmp_path / "tier")
    pix_run, lat_run = tmp_path / "run_pix", tmp_path / "run_lat"
    pix_smp, lat_smp = tmp_path / "smp_pix", tmp_path / "smp_lat"
    out = tmp_path / "report"

    for target, run, smp in (("change_map_pixels", pix_run, pix_smp), ("map_latent", lat_run, lat_smp)):
        assert cli(["train", "--tier", str(tier), "--target", target, "--arm", "frame_caption",
                    "--qk-norm", "on", "--steps", "2", "--batch", "2", "--out", str(run), *SMALL]) == 0
        assert cli(["sample", "--tier", str(tier), "--target", target, "--run", str(run),
                    "--k", "2", "--steps", "2", "--out", str(smp)]) == 0

    assert cli(["score", "--tier", str(tier), "--target", "change_map_pixels", "--split", "val",
                "--samples", str(pix_smp / "frame_caption.npz"),
                "--latent-samples", str(lat_smp / "frame_caption.npz"),
                "--out", str(out), "--encoder", "standin"]) == 0

    rep = json.loads((out / "report.json").read_text())
    md = (out / "report.md").read_text()

    assert set(rep["rows"]) == {
        "frame_caption", "frame_caption_map_latent",
        "all_black_floor", "mean_map_floor", "previous_target_floor",
    }
    first_id = rep["window_ids"][0]
    lat_metrics = set(rep["rows"]["frame_caption_map_latent"]["per_window"][first_id])
    assert lat_metrics == {"token_cosine", "region_token_cosine", "token_mse"}
    pix_metrics = set(rep["rows"]["frame_caption"]["per_window"][first_id])
    assert {"lpips", "region_lpips", "mse", "spill", "token_cosine", "region_token_cosine", "token_mse"} <= pix_metrics

    by_key = {(c["a"], c["b"], c["metric"]): c for c in rep["comparisons"]}
    # H1, H2, H3, H4 all present with Window-resampled bootstrap
    h1 = by_key[("frame_caption:single", "previous_target_floor:single", "region_lpips")]
    h2 = by_key[("frame_caption_map_latent:single", "previous_target_floor:single", "region_token_cosine")]
    h3 = by_key[("frame_caption_map_latent:single", "frame_caption:single", "region_token_cosine")]
    h4_pix = by_key[("frame_caption:best_of_k", "frame_caption:single", "region_lpips")]
    h4_lat = by_key[("frame_caption_map_latent:best_of_k", "frame_caption_map_latent:single", "region_token_cosine")]
    for c in (h1, h2, h3, h4_pix, h4_lat):
        assert "n_source_windows" in c and c["ci_lo"] <= c["ci_hi"]

    assert "frame_caption_map_latent" in md
    assert "not decoded to pixels" in md.lower()

    # Refusals
    exits_nonzero(["score", "--tier", str(tier), "--target", "change_map_pixels", "--split", "val",
                   "--samples", str(pix_smp / "frame_caption.npz"),
                   "--latent-samples", str(pix_smp / "frame_caption.npz"),
                   "--out", str(tmp_path / "bad1"), "--encoder", "standin"])
    exits_nonzero(["score", "--tier", str(tier), "--target", "change_map_pixels", "--split", "val",
                   "--samples", str(pix_smp / "frame_caption.npz"),
                   "--latent-samples", str(lat_smp / "frame_caption.npz"),
                   "--token-metrics", "off", "--out", str(tmp_path / "bad2")])


def test_cluster_two_frame_train_job_name_and_paired_score_sbatch(tmp_path):
    from changemap import cluster

    sha = "a" * 40
    # 1. render with --job-name distinguishes pixel vs latent frame_caption runs
    pix_sbatch = tmp_path / "pix.sbatch"
    assert cluster.main(["render", "--arm", "frame_caption", "--job-name", "tf-pixels", "--sha", sha,
                         "--tier", "/scratch/two_frame_tier_1k", "--out", "/runs/pix", "--steps", "20000",
                         "--mem", "64G", "--train-args", "--target change_map_pixels --qk-norm on --grad-clip 1.0",
                         "--script", str(pix_sbatch)]) == 0
    txt = pix_sbatch.read_text()
    assert "#SBATCH --job-name=tf-pixels" in txt
    assert "--run-name tf-pixels" in txt
    assert "--target change_map_pixels --qk-norm on --grad-clip 1.0" in txt
    assert "#SBATCH --mem=64G" in txt

    assert cluster.main(["--home", str(tmp_path), "submit", "--dry-run", "--arm", "frame_caption",
                         "--job-name", "tf-latent", "--sha", sha, "--tier", "/scratch/two_frame_tier_1k",
                         "--out", "/runs/lat", "--steps", "20000", "--mem", "64G",
                         "--train-args", "--target map_latent --qk-norm on --grad-clip 1.0"]) == 0

    # 2. render-two-frame-score and submit-two-frame-score --dry-run
    score_sbatch = tmp_path / "two_frame_score.sbatch"
    assert cluster.main(["render-two-frame-score", "--sha", sha, "--tier", "/scratch/two_frame_tier_1k",
                         "--pixel-run", "/runs/pix", "--latent-run", "/runs/lat",
                         "--samples", "/samples", "--out", "/report", "--lpips-dir", "/lpips",
                         "--script", str(score_sbatch)]) == 0
    stxt = score_sbatch.read_text()
    assert "#SBATCH --partition=a100" in stxt
    assert "#SBATCH --mem=64G" in stxt
    assert cluster.SIF_SHA256 in stxt
    assert "--target change_map_pixels" in stxt
    assert "--target map_latent" in stxt
    assert "--latent-samples" in stxt
    assert "--latent-stats" not in stxt

    assert cluster.main(["--home", str(tmp_path), "submit-two-frame-score", "--dry-run", "--sha", sha,
                         "--tier", "/scratch/two_frame_tier_1k", "--pixel-run", "/runs/pix",
                         "--latent-run", "/runs/lat", "--samples", "/samples", "--out", "/report",
                         "--lpips-dir", "/lpips", "--dependency", "afterok:111:222"]) == 0

