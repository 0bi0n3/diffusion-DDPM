"""Map Latent target and latent-space scoring for both two-frame targets (#17) through the CLI seam."""
import hashlib
import json

import numpy as np
import pytest

from changemap.cli import main as cli
from test_change_map_pixels import (NO_CHANGE_IDS, SCORED_VAL, SMALL, VAL_IDS, exits_nonzero, true_map,
                                    write_two_frame_tier)

TOKEN_METRICS = {"token_cosine", "region_token_cosine", "token_mse"}
PIXEL_METRICS = {"lpips", "region_lpips", "mse", "spill"}


@pytest.fixture
def tftier(tmp_path):
    return write_two_frame_tier(tmp_path / "tier")


def _write_pixel_arm(tmp_path, name, ids, maps):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    np.savez(d / f"{name}.npz", ids=np.array(ids), maps=maps.astype(np.float32), arm=name, target="change_map_pixels")
    return d


def _score(tier, tmp_path, target, sample_dirs=(), out="rep", extra=()):
    argv = ["score", "--tier", str(tier), "--split", "val", "--target", target,
            "--lpips-dir", str(tmp_path), "--out", str(tmp_path / out)]
    for d in sample_dirs:
        argv += ["--samples", str(d)]
    assert cli(argv + list(extra)) == 0
    return json.loads((tmp_path / out / "report.json").read_text())


def test_pixel_run_reports_both_pixel_and_latent_metrics_and_reuses_cache(tftier, tmp_path, fake_encoder):
    truth_maps = []
    for r in SCORED_VAL:
        regs = [b for b, ok in zip(r["regions"], r["box_ok_50_20"]) if ok] if r["target_status"] == "target" else []
        truth_maps.append(true_map(regs))
    truth_arr = np.stack(truth_maps)[:, None]
    arm = _write_pixel_arm(tmp_path, "perfect", VAL_IDS, truth_arr)

    rep1 = _score(tftier, tmp_path, "change_map_pixels", [arm], out="r1")
    pw = rep1["rows"]["perfect"]["per_window"]
    for wid in VAL_IDS:
        assert set(pw[wid]) == PIXEL_METRICS | TOKEN_METRICS
        assert pw[wid]["lpips"] == [0.0] and pw[wid]["mse"] == [0.0]
        assert pw[wid]["token_cosine"] == [pytest.approx(1.0, abs=1e-5)]
        assert pw[wid]["token_mse"] == [pytest.approx(0.0, abs=1e-9)]
        if wid in NO_CHANGE_IDS:
            assert pw[wid]["region_lpips"] == [None] and pw[wid]["region_token_cosine"] == [None]
        else:
            assert pw[wid]["region_lpips"] == [0.0]
            assert pw[wid]["region_token_cosine"] == [pytest.approx(1.0, abs=1e-5)]

    # Cache and training Map Latent stats exist with valid sha256 sidecars
    cache = sorted((tftier / "token_cache").glob("val__*.npz"))
    assert len(cache) == 1
    side = json.loads(cache[0].with_suffix(".json").read_text())
    assert side["sha256"] == hashlib.sha256(cache[0].read_bytes()).hexdigest()

    # Second score run reuses cached val and train Map Latents (only encodes generated arm + floor)
    fake_encoder.reset()
    rep2 = _score(tftier, tmp_path, "change_map_pixels", [arm], out="r2")
    n_gen = len(VAL_IDS) * 3  # perfect (k=1) + all_black_floor + mean_map_floor (k=1 each)
    assert fake_encoder.frames_encoded() == n_gen
    assert rep2["rows"] == rep1["rows"]


def test_map_latent_train_sample_score_end_to_end_and_not_decoded_report(tftier, tmp_path):
    run, samples, rep_dir = tmp_path / "run", tmp_path / "samples", tmp_path / "report"
    assert cli(["train", "--tier", str(tftier), "--target", "map_latent", "--arm", "frame_caption",
                "--steps", "3", "--batch", "2", *SMALL, "--out", str(run)]) == 0
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["target"] == "map_latent"

    assert cli(["sample", "--run", str(run), "--tier", str(tftier), "--split", "val",
                "--target", "map_latent", "--k", "2", "--steps", "2", "--out", str(samples)]) == 0
    z = np.load(samples / "frame_caption.npz")
    assert str(z["target"]) == "map_latent"
    assert list(z["ids"]) == VAL_IDS
    assert z["latents"].shape == (len(VAL_IDS), 2, 28, 50, 8)

    # sample and score refuse another target
    exits_nonzero(["sample", "--run", str(run), "--tier", str(tftier), "--split", "val",
                   "--target", "change_map_pixels", "--k", "1", "--steps", "2", "--out", str(tmp_path / "bad_s")])
    exits_nonzero(["score", "--tier", str(tftier), "--split", "val", "--target", "change_map_pixels",
                   "--lpips-dir", str(tmp_path), "--samples", str(samples), "--out", str(tmp_path / "bad_r")])

    rep = _score(tftier, tmp_path, "map_latent", [samples], out="report")
    assert rep["target"] == "map_latent"
    assert list(rep["rows"]) == ["frame_caption", "all_black_floor"]
    fc = rep["rows"]["frame_caption"]
    for wid in VAL_IDS:
        assert set(fc["per_window"][wid]) == TOKEN_METRICS
        assert not (PIXEL_METRICS & set(fc["per_window"][wid]))
    s = fc["summary"]["change"]
    assert s["single"]["psnr"] is None
    assert s["best_of_k"]["token_cosine"] >= s["single"]["token_cosine"] - 1e-12
    assert s["best_of_k"]["region_token_cosine"] >= s["single"]["region_token_cosine"] - 1e-12
    assert s["best_of_k"]["token_mse"] <= s["single"]["token_mse"] + 1e-12
    md = (rep_dir / "report.md").read_text()
    assert "not decoded" in md.lower()
    assert "Masked Frames" not in md and "H1 to H5" not in md and "PSNR" not in md and "Panels:" not in md


def test_true_map_latent_scores_perfectly_normalises_with_train_stats_and_reuses_cache(tftier, tmp_path, fake_encoder):
    # First score (floors only) populates cached train/val Map Latents and train stats
    _score(tftier, tmp_path, "map_latent", out="r0")
    cache_dir = tftier / "token_cache"
    train_npz = next(cache_dir.glob("train__*.npz"))
    val_npz = next(cache_dir.glob("val__*.npz"))
    stats_npz = next(cache_dir.glob("train_stats__*.npz"))
    for p in (train_npz, val_npz, stats_npz):
        side = json.loads(p.with_suffix(".json").read_text())
        assert side["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()

    st = np.load(stats_npz)
    mean, std = st["mean"], st["std"]
    train_raw = np.load(train_npz)["tokens"].astype(np.float32)
    train_norm = (train_raw - mean) / std
    assert np.allclose(train_norm.mean(axis=(0, 1, 2)), 0.0, atol=1e-4)
    non_const = train_raw.std(axis=(0, 1, 2)) > 1e-6
    assert non_const.any()
    assert np.allclose(train_norm.std(axis=(0, 1, 2))[non_const], 1.0, atol=1e-3)

    # Normalise val Map Latents with TRAIN stats and score as a perfect arm
    vz = np.load(val_npz)
    by_id = {str(i): t for i, t in zip(vz["ids"], vz["tokens"])}
    val_raw = np.stack([by_id[i] for i in VAL_IDS]).astype(np.float32)
    val_norm = ((val_raw - mean) / std)[:, None]
    arm_dir = tmp_path / "perfect_latent"
    arm_dir.mkdir()
    np.savez(arm_dir / "perfect.npz", ids=np.array(VAL_IDS), latents=val_norm, arm="perfect", target="map_latent")

    fake_encoder.reset()
    rep = _score(tftier, tmp_path, "map_latent", [arm_dir], out="r_perfect")
    # Cached train/val latents are reused: only the single all-black floor map is encoded
    assert fake_encoder.frames_encoded() <= 1
    pw = rep["rows"]["perfect"]["per_window"]
    for wid in VAL_IDS:
        assert pw[wid]["token_cosine"] == [pytest.approx(1.0, abs=1e-5)]
        assert pw[wid]["token_mse"] == [pytest.approx(0.0, abs=1e-9)]
        if wid in NO_CHANGE_IDS:
            assert pw[wid]["region_token_cosine"] == [None]
        else:
            assert pw[wid]["region_token_cosine"] == [pytest.approx(1.0, abs=1e-5)]
