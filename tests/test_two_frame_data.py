"""Two-frame tier data preparation and duplicate Step Pair id handling (#20) through the CLI seam."""
import hashlib
import json

import numpy as np
import pytest

from changemap import data
from changemap.cli import main as cli
from latentdit.encode import StandInEncoder, encode_tutorial, read_shard
from test_frames_cli import make_video

DIM = 8
T = 7
BOX_BIG = [192, 108, 576, 324]
BOX_SMALL = [10, 10, 20, 20]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tf_row(pid, source_win, frames_ab, video, source, goal="Goal A", status="target",
           regions=(BOX_BIG,), ok=None, tutorial=T):
    ok = [True] * len(regions) if ok is None else list(ok)
    return {
        "id": pid, "tutorial": tutorial, "video": str(video), "frames": list(frames_ab),
        "source_window_id": source_win, "goal": goal, "subgoal": "Subgoal", "action": "Click",
        "evidence": "SECRET EVIDENCE", "target_key": pid, "target_source": str(source),
        "target_status": status, "regions": [list(b) for b in regions], "box_ok_50_20": ok,
    }


def index_row(r):
    regs = [b for b, ok in zip(r["regions"], r["box_ok_50_20"]) if ok] if r["target_status"] == "target" else []
    win = f"{r['id']}@{r['source_window_id']}"
    return {
        "id": f"pixel_difference/last_step/goal_subgoal_action/{win}",
        "window": win, "file": f"map_{win}.png", "n_regions": len(regs),
        "regions_px_1920x1080": [[float(v) for v in b] for b in regs],
    }


@pytest.fixture
def tf_ctx(tmp_path):
    video = tmp_path / "v.mp4"
    make_video(video, seconds=12)
    src = tmp_path / "pixel_difference.json"
    src.write_text(json.dumps({"pairs": {
        "t7_2-3": {"regions": [BOX_BIG, BOX_SMALL], "dropped": False, "no_change": False},
        "t7_3-4": {"regions": [BOX_BIG], "dropped": False, "no_change": False},
        "t7_4-5": {"regions": [], "dropped": False, "no_change": True},
    }}))
    tier = tmp_path / "tier"
    tier.mkdir()
    # Two Step Pairs share id "t7_3-4" under different source_window_ids and different Captions!
    train_rows = [
        tf_row("t7_3-4", "t7_2", [3, 4], video, src, goal="First window goal"),
        tf_row("t7_3-4", "t7_3", [3, 4], video, src, goal="Second window goal"),
        tf_row("t7_4-5", "t7_3", [4, 5], video, src, goal="Second window goal", status="no_change", regions=()),
        tf_row("t7_5-6", "t7_3", [5, 6], video, src, status="dropped"),
    ]
    # Val has second 6 (missing from store -> encoded fresh) and duplicate id "t7_3-4"
    val_rows = [
        tf_row("t7_3-4", "t7_2", [3, 4], video, src, goal="Val goal A"),
        tf_row("t7_3-4", "t7_3", [3, 4], video, src, goal="Val goal B"),
        tf_row("t7_6-7", "t7_5", [6, 7], video, src, goal="Val goal C", status="no_change", regions=()),
    ]
    for split, rows in (("train", train_rows), ("val", val_rows)):
        (tier / f"{split}_targets.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        idx_rows = [index_row(r) for r in rows if r["target_status"] != "dropped"]
        (tier / f"{split}_index.jsonl").write_text("\n".join(json.dumps(r) for r in idx_rows) + "\n")

    store = tmp_path / "store"
    enc = StandInEncoder(grid=(28, 50), dim=DIM)
    encode_tutorial(T, str(video), [3, 4], enc, store)          # seconds 3 and 4 in store; 6 is missing
    stats = tmp_path / "stats.npz"
    g = np.random.default_rng(7)
    np.savez(stats, mean=g.normal(size=DIM), std=g.uniform(0.5, 2.0, size=DIM))
    return {"tier": tier, "store": store, "stats": stats, "enc": enc, "video": video}


def test_context_and_previous_support_two_frame_input_frame_and_duplicate_ids(tf_ctx, fake_encoder):
    tier = tf_ctx["tier"]
    assert cli(["context", "--tier", str(tier), "--split", "train", "--out", str(tier),
                "--store", str(tf_ctx["store"]), "--stats", str(tf_ctx["stats"])]) == 0
    z = np.load(tier / "train_context.npz")
    assert list(z["ids"]) == ["t7_3-4@t7_2", "t7_3-4@t7_3", "t7_4-5@t7_3"]
    assert z["f8"].shape == (3, 28, 50, DIM) and z["f8"].dtype == np.float16
    # Input Frame (frames[0] = second 3) matches the normalised store entry for second 3
    sec, lat, _ = read_shard(tf_ctx["store"], T)
    s = np.load(tf_ctx["stats"])
    want3 = (np.asarray(lat[list(sec).index(3)], np.float32) - s["mean"].astype(np.float32)) / s["std"].astype(np.float32)
    assert np.allclose(z["f8"][0].astype(np.float32), want3, atol=2e-3)
    assert np.allclose(z["f8"][1].astype(np.float32), want3, atol=2e-3)
    # Duplicate Step Pair ids keep their distinct Captions
    assert not np.allclose(z["caption"][0], z["caption"][1])
    assert fake_encoder.frames_encoded() == 0

    # Val has 2 store hits (second 3) and 1 fresh encode (second 6)
    assert cli(["context", "--tier", str(tier), "--split", "val", "--out", str(tier),
                "--store", str(tf_ctx["store"]), "--stats", str(tf_ctx["stats"])]) == 0
    m = json.loads((tier / "context_manifest.json").read_text())["val"]["context"]
    assert (m["windows"], m["from_store"], m["encoded_fresh"], m["fresh_ids"]) == (3, 2, 1, ["t7_6-7@t7_5"])
    assert fake_encoder.frames_encoded() == 1

    # Previous records preserve composite keys and load cleanly via load_windows
    assert cli(["previous", "--tier", str(tier), "--split", "train", "--out", str(tier)]) == 0
    prev = data.read_previous(tier, "train")
    assert list(prev) == ["t7_3-4@t7_2", "t7_3-4@t7_3", "t7_4-5@t7_3"]
    wins = data.load_windows(tier, "train")
    assert len(wins) == 3 and wins[0].has_previous and wins[1].has_previous


def test_duplicate_step_pair_ids_train_sample_score_end_to_end(tf_ctx, tmp_path, fake_encoder):
    """A two-frame tier whose train and val splits contain duplicate Step Pair ids under different source_window_ids
    trains, samples, and scores all rows without overwriting entries in context, samples, or report per_window."""
    tier = tf_ctx["tier"]
    # Write 768-dim context keyed by composite Step Pair key (as real `changemap context` produces for DINOv3 + CLIP)
    g = np.random.default_rng(19)
    for split, keys in (("train", ["t7_3-4@t7_2", "t7_3-4@t7_3", "t7_4-5@t7_3"]),
                        ("val", ["t7_3-4@t7_2", "t7_3-4@t7_3", "t7_6-7@t7_5"])):
        np.savez(tier / f"{split}_context.npz",
                 ids=np.array(keys),
                 f8=g.standard_normal((len(keys), 28, 50, 768)).astype(np.float16),
                 caption=g.standard_normal((len(keys), 77, 768)).astype(np.float16))

    small = ["--hidden", "32", "--depth", "2", "--heads", "2"]
    for target in ("change_map_pixels", "map_latent"):
        run, samples, rep_dir = tmp_path / f"run_{target}", tmp_path / f"smp_{target}", tmp_path / f"rep_{target}"
        assert cli(["train", "--tier", str(tier), "--target", target, "--arm", "frame_caption",
                    "--steps", "2", "--batch", "2", *small, "--out", str(run)]) == 0
        assert cli(["sample", "--run", str(run), "--tier", str(tier), "--split", "val",
                    "--target", target, "--k", "2", "--steps", "2", "--out", str(samples)]) == 0
        z = np.load(samples / "frame_caption.npz")
        assert list(z["ids"]) == ["t7_3-4@t7_2", "t7_3-4@t7_3", "t7_6-7@t7_5"]
        assert cli(["score", "--tier", str(tier), "--split", "val", "--target", target,
                    "--lpips-dir", str(tmp_path), "--samples", str(samples), "--out", str(rep_dir)]) == 0
        rep = json.loads((rep_dir / "report.json").read_text())
        assert rep["n_windows"] == 3
        assert rep["window_ids"] == ["t7_3-4@t7_2", "t7_3-4@t7_3", "t7_6-7@t7_5"]
        assert list(rep["rows"]["frame_caption"]["per_window"]) == ["t7_3-4@t7_2", "t7_3-4@t7_3", "t7_6-7@t7_5"]
        assert rep["rows"]["frame_caption"]["summary"]["all"]["n"] == 3


def test_map_latents_cli_prebuilds_caches_and_stats_and_cluster_renders_sbatch(tf_ctx, tmp_path, fake_encoder):
    from changemap import cluster

    tier = tf_ctx["tier"]
    assert cli(["map-latents", "--tier", str(tier), "--splits", "train", "val"]) == 0
    cache_dir = tier / "token_cache"
    train_npz = next(cache_dir.glob("train__*.npz"))
    val_npz = next(cache_dir.glob("val__*.npz"))
    stats_npz = next(cache_dir.glob("train_stats__*.npz"))
    for p in (train_npz, val_npz, stats_npz):
        side = json.loads(p.with_suffix(".json").read_text())
        assert side["sha256"] == sha(p)
    # Duplicate Step Pair ids are encoded once, not twice (train has 2 unique pairs, val has 2 unique pairs)
    assert fake_encoder.frames_encoded() == 4

    # Refuses test split without --verdict
    with pytest.raises(SystemExit) as e:
        cli(["map-latents", "--tier", str(tier), "--splits", "test"])
    assert e.value.code == 2

    # Subsequent map_latent train + score hits the pre-built cache (0 encodes in train; only floor maps in score)
    fake_encoder.reset()
    assert cli(["train", "--tier", str(tier), "--target", "map_latent", "--arm", "unconditional",
                "--steps", "2", "--batch", "2", "--hidden", "32", "--depth", "2", "--heads", "2",
                "--out", str(tmp_path / "run_ml")]) == 0
    assert fake_encoder.frames_encoded() == 0
    assert cli(["score", "--tier", str(tier), "--split", "val", "--target", "map_latent",
                "--out", str(tmp_path / "rep_ml")]) == 0
    assert fake_encoder.frames_encoded() <= 2

    # Cluster render-map-latents and submit-map-latents --dry-run
    script = tmp_path / "map_latents.sbatch"
    sha40 = "a" * 40
    assert cluster.main(["render-map-latents", "--sha", sha40, "--tier", "/scratch/tier",
                         "--script", str(script)]) == 0
    txt = script.read_text()
    assert "#SBATCH --partition=a100" in txt
    assert cluster.SIF_SHA256 in txt
    assert "HF_HUB_OFFLINE=1" in txt
    assert "python -m changemap.cli map-latents" in txt
    assert "--device cuda" in txt
    assert cluster.main(["submit-map-latents", "--sha", sha40, "--tier", "/scratch/tier", "--dry-run"]) == 0


