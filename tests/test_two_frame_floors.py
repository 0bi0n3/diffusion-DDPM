"""Two-frame floors (Mean-Map and Previous-Target) for both `change_map_pixels` and `map_latent` (#18)."""
import json

import numpy as np
import pytest

from changemap.cli import main as cli
from test_change_map_pixels import H, W, pair_row, true_map, write_two_frame_tier

BOX_A = [192, 108, 576, 324]        # 80x45 = 3,600 px at 224x400 (total 89,600 px)
BOX_B = [960, 540, 1344, 756]       # 80x45 = 3,600 px at 224x400, disjoint from BOX_A


def _write_pixel_arm(tmp_path, name, ids, maps):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    np.savez(d / f"{name}.npz", ids=np.array(ids), maps=maps.astype(np.float32), arm=name, target="change_map_pixels")
    return d


def _score(tier, tmp_path, target, sample_dirs=(), out="rep"):
    argv = ["score", "--tier", str(tier), "--split", "val", "--target", target,
            "--lpips-dir", str(tmp_path), "--out", str(tmp_path / out)]
    for d in sample_dirs:
        argv += ["--samples", str(d)]
    assert cli(argv) == 0
    return json.loads((tmp_path / out / "report.json").read_text())


def test_mean_map_floor_pixel_matches_independent_truth(tmp_path):
    """3 non-dropped train pairs (2 with BOX_A, 1 no_change) + 1 dropped full-screen pair -> independent mean map is
    2/3 inside BOX_A (3,600 px) and 0 outside (86,000 px)."""
    train_rows = [
        pair_row("tr_0-1", "tr0", [0, 1], "target", [BOX_A]),
        pair_row("tr_1-2", "tr0", [1, 2], "target", [BOX_A]),
        pair_row("tr_2-3", "tr0", [2, 3], "no_change"),
        pair_row("tr_3-4", "tr0", [3, 4], "dropped", [[0, 0, 1920, 1080]]),
    ]
    val_rows = [
        pair_row("va_0-1", "va0", [0, 1], "target", [BOX_A]),
        pair_row("va_1-2", "va0", [1, 2], "no_change"),
    ]
    tier = write_two_frame_tier(tmp_path / "tier", train_rows=train_rows, val_rows=val_rows)
    arm = _write_pixel_arm(tmp_path, "frame_caption", ["va_0-1", "va_1-2"],
                           np.stack([true_map([BOX_A]), np.zeros((H, W), np.float32)])[:, None])

    rep = _score(tier, tmp_path, "change_map_pixels", [arm], out="rep_mean")
    assert "mean_map_floor" in rep["rows"]
    mm = rep["rows"]["mean_map_floor"]
    assert mm["per_window"]["va_0-1"]["mse"] == [pytest.approx(1 / 224, rel=1e-6)]
    assert mm["per_window"]["va_0-1"]["spill"] == [pytest.approx(0.0, abs=1e-12)]
    assert mm["per_window"]["va_1-2"]["mse"] == [pytest.approx(1 / 56, rel=1e-6)]
    assert mm["per_window"]["va_1-2"]["spill"] == [pytest.approx(1.0, abs=1e-12)]
    assert mm["summary"]["all"]["single"]["mse"] == pytest.approx(5 / 448, rel=1e-6)
    cmp_pairs = {(c["a"], c["b"], c["metric"]) for c in rep["comparisons"]}
    assert ("frame_caption:single", "mean_map_floor:single", "mse") in cmp_pairs
    assert ("frame_caption:single", "mean_map_floor:single", "lpips") in cmp_pairs


def test_previous_target_floor_same_window_fallback_and_missing_count(tmp_path):
    """Previous-Target takes Ft-1 -> Ft from the same Window when present (including dropped -> missing and
    no_change -> usable empty), falls back to `<split>_previous.jsonl` for the first pair of a Window, and counts
    Step Pairs with no usable previous pair in report.json and report.md."""
    train_rows = [
        pair_row("tr_0-1", "tr0", [0, 1], "target", [BOX_A]),
        pair_row("tr_1-2", "tr0", [1, 2], "no_change"),
    ]
    val_rows = [
        pair_row("wA_10-11", "wA", [10, 11], "target", [BOX_A]),
        pair_row("wA_11-12", "wA", [11, 12], "target", [BOX_A]),
        pair_row("wA_12-13", "wA", [12, 13], "dropped", [[0, 0, 1920, 1080]]),
        pair_row("wA_13-14", "wA", [13, 14], "target", [BOX_A]),
        pair_row("wB_20-21", "wB", [20, 21], "target", [BOX_B]),
        pair_row("wB_21-22", "wB", [21, 22], "no_change"),
        pair_row("wB_22-23", "wB", [22, 23], "no_change"),
    ]
    tier = write_two_frame_tier(tmp_path / "tier", train_rows=train_rows, val_rows=val_rows)
    prev_records = [
        {"id": "wA_10-11", "previous": {"regions": [BOX_A], "box_ok_50_20": [True]}},
        # Conflicting record for a non-first pair: same-Window previous pair (BOX_A) must win over tier record (BOX_B)
        {"id": "wA_11-12", "previous": {"regions": [BOX_B], "box_ok_50_20": [True]}},
    ]
    (tier / "val_previous.jsonl").write_text("\n".join(json.dumps(r) for r in prev_records) + "\n")

    scored_ids = [r["id"] for r in val_rows if r["target_status"] != "dropped"]
    arm = _write_pixel_arm(tmp_path, "frame_caption", scored_ids, np.zeros((len(scored_ids), 1, H, W), np.float32))
    rep = _score(tier, tmp_path, "change_map_pixels", [arm], out="rep_prev")

    assert list(rep["rows"]) == ["frame_caption", "all_black_floor", "mean_map_floor", "previous_target_floor"]
    assert rep["n_no_previous"] == 2
    pt = rep["rows"]["previous_target_floor"]
    assert pt["n_no_previous"] == 2
    pw = pt["per_window"]
    assert pw["wA_10-11"]["mse"] == [0.0] and pw["wA_10-11"]["lpips"] == [0.0]         # first pair -> tier record
    assert pw["wA_11-12"]["mse"] == [0.0] and pw["wA_11-12"]["lpips"] == [0.0]         # same-Window previous wins
    assert pw["wA_13-14"]["mse"] == [pytest.approx(9 / 224, rel=1e-6)]                 # dropped previous -> all black
    assert pw["wB_20-21"]["mse"] == [pytest.approx(9 / 224, rel=1e-6)]                 # absent previous -> all black
    assert pw["wB_21-22"]["mse"] == [pytest.approx(9 / 224, rel=1e-6)]                 # same-Window BOX_B vs no_change
    assert pw["wB_21-22"]["spill"] == [pytest.approx(1.0, abs=1e-12)]
    assert pw["wB_22-23"]["mse"] == [0.0]                                              # same-Window no_change -> usable

    md = (tmp_path / "rep_prev" / "report.md").read_text()
    assert "Previous-Target Floor:" in md and "2 of 6" in md
    cmp_pairs = {(c["a"], c["b"], c["metric"]) for c in rep["comparisons"]}
    for floor in ("all_black_floor", "mean_map_floor", "previous_target_floor"):
        assert ("frame_caption:single", f"{floor}:single", "mse") in cmp_pairs
        assert ("frame_caption:single", f"{floor}:single", "region_lpips") in cmp_pairs


def test_map_latent_floors_are_encoded_floor_maps_and_match_pixel_target(tmp_path, fake_encoder):
    """Both targets report the same three floors on identical Step Pair ids with matching latent-space scores;
    map_latent floors are the encoded and train-stats-normalised floor maps, and bootstrap compares the arm against
    every floor."""
    train_rows = [
        pair_row("tr_0-1", "tr0", [0, 1], "target", [BOX_A]),
        pair_row("tr_1-2", "tr0", [1, 2], "target", [BOX_A]),
        pair_row("tr_2-3", "tr0", [2, 3], "no_change"),
        pair_row("tr_3-4", "tr0", [3, 4], "dropped", [[0, 0, 1920, 1080]]),
    ]
    val_rows = [
        pair_row("wA_10-11", "wA", [10, 11], "target", [BOX_A]),
        pair_row("wA_11-12", "wA", [11, 12], "target", [BOX_B]),
        pair_row("wA_12-13", "wA", [12, 13], "dropped", [[0, 0, 1920, 1080]]),
        pair_row("wA_13-14", "wA", [13, 14], "target", [BOX_A]),
        pair_row("wB_20-21", "wB", [20, 21], "no_change"),
    ]
    tier = write_two_frame_tier(tmp_path / "tier", train_rows=train_rows, val_rows=val_rows)
    (tier / "val_previous.jsonl").write_text(
        json.dumps({"id": "wA_10-11", "previous": {"regions": [BOX_A], "box_ok_50_20": [True]}}) + "\n"
    )
    scored_ids = [r["id"] for r in val_rows if r["target_status"] != "dropped"]

    rep_pix = _score(tier, tmp_path, "change_map_pixels", out="rep_pix")

    # Independently encode the hand-built Mean-Map (2/3 inside BOX_A) through the Map Latent path and train stats
    stats_npz = next((tier / "token_cache").glob("train_stats__*.npz"))
    st = np.load(stats_npz)
    mean_img = np.repeat(np.rint(true_map([BOX_A]) * (2 / 3) * 255).astype(np.uint8)[..., None], 3, axis=-1)
    import cv2
    mean_up = cv2.resize(mean_img, (W * 4, H * 4), interpolation=cv2.INTER_CUBIC)
    mean_raw = np.asarray(fake_encoder.load("dinov3", "cpu").encode([mean_up])[0], np.float32)
    mean_norm = ((mean_raw - st["mean"]) / st["std"]).astype(np.float32)

    arm_dir = tmp_path / "arm_lat"
    arm_dir.mkdir()
    np.savez(arm_dir / "frame_caption.npz", ids=np.array(scored_ids),
             latents=np.broadcast_to(mean_norm, (len(scored_ids), 1, *mean_norm.shape)).copy(),
             arm="frame_caption", target="map_latent")
    rep_lat = _score(tier, tmp_path, "map_latent", [arm_dir], out="rep_lat")

    floors = ["all_black_floor", "mean_map_floor", "previous_target_floor"]
    assert rep_lat["window_ids"] == rep_pix["window_ids"] == scored_ids
    assert list(rep_pix["rows"]) == floors
    assert list(rep_lat["rows"]) == ["frame_caption", *floors]
    assert rep_lat["n_no_previous"] == rep_pix["n_no_previous"] == 2
    assert rep_lat["rows"]["previous_target_floor"]["n_no_previous"] == 2

    # Every floor's latent-space scores match across both targets on every Step Pair id
    for floor in floors:
        for wid in scored_ids:
            for m in ("token_cosine", "region_token_cosine", "token_mse"):
                pv = rep_pix["rows"][floor]["per_window"][wid][m][0]
                lv = rep_lat["rows"][floor]["per_window"][wid][m][0]
                if pv is None:
                    assert lv is None
                else:
                    assert lv == pytest.approx(pv, abs=1e-5)

    # Independent check: mean_map_floor matches our independently encoded mean-map arm
    for wid in scored_ids:
        for m in ("token_cosine", "region_token_cosine", "token_mse"):
            av = rep_lat["rows"]["frame_caption"]["per_window"][wid][m][0]
            mv = rep_lat["rows"]["mean_map_floor"]["per_window"][wid][m][0]
            if av is None:
                assert mv is None
            else:
                assert mv == pytest.approx(av, abs=1e-6)

    # Independent check: wA_10-11 previous pair equals truth (BOX_A) -> perfect latent scores
    pt_first = rep_lat["rows"]["previous_target_floor"]["per_window"]["wA_10-11"]
    assert pt_first["token_cosine"] == [pytest.approx(1.0, abs=1e-5)]
    assert pt_first["region_token_cosine"] == [pytest.approx(1.0, abs=1e-5)]
    assert pt_first["token_mse"] == [pytest.approx(0.0, abs=1e-9)]

    md = (tmp_path / "rep_lat" / "report.md").read_text()
    assert "Previous-Target Floor:" in md and "2 of 4" in md

    cmp_pairs = {(c["a"], c["b"], c["metric"]) for c in rep_lat["comparisons"]}
    for floor in floors:
        assert ("frame_caption:single", f"{floor}:single", "region_token_cosine") in cmp_pairs
        assert ("frame_caption:single", f"{floor}:single", "token_mse") in cmp_pairs


