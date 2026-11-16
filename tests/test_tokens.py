"""Token-space scoring (#12): generated and true Masked Frames encoded by the same (injected) encoder path."""
import json

import numpy as np
import pytest

from changemap.cli import main as cli
from test_end_to_end import IDS, NO_CHANGE
from test_masked_frame import (FLOORS, exits_nonzero, mtier, noisy_arm, score, truth_frames,  # noqa: F401
                               write_arm)

TOKEN = ("token_cosine", "region_token_cosine", "token_mse")


def test_true_masked_frame_scores_token_cosine_one_and_token_mse_zero(mtier, tmp_path):
    rgb, mask = truth_frames()
    rep = score(mtier, tmp_path, [write_arm(tmp_path, "perfect", rgb, mask)])
    for wid, w in rep["rows"]["perfect"]["per_window"].items():
        assert w["token_cosine"] == [pytest.approx(1.0, abs=1e-5)], wid
        assert w["token_mse"] == [pytest.approx(0.0, abs=1e-9)], wid
        if wid in NO_CHANGE:
            assert w["region_token_cosine"] == [None]            # absent for a No Change Window
        else:
            assert w["region_token_cosine"] == [pytest.approx(1.0, abs=1e-5)], wid


def test_floors_get_token_metrics_too(mtier, tmp_path):
    rep = score(mtier, tmp_path)
    for f in FLOORS:
        for wid, w in rep["rows"][f]["per_window"].items():
            assert set(TOKEN) <= set(w), (f, wid)
    cf = rep["rows"]["copy_f8_floor"]["per_window"]
    for wid in ("t1_10", "t1_20", "t1_30", "t1_60"):                 # F8 == F9 inside these Windows' regions
        assert cf[wid]["token_cosine"] == [pytest.approx(1.0, abs=1e-5)]
    bk = rep["rows"]["all_black_floor"]["per_window"]
    assert bk["t1_10"]["token_cosine"] == [1.0] and bk["t1_10"]["token_mse"] == [0.0]    # pure black truth
    assert bk["t1_20"]["token_cosine"][0] < 1 and bk["t1_20"]["token_mse"][0] > 0
    assert bk["t1_30"]["region_token_cosine"] == [None]


def test_token_metrics_off_removes_them_and_grid_map_refuses_on(mtier, tmp_path):
    rep = score(mtier, tmp_path, extra=["--token-metrics", "off"])
    for r in rep["rows"].values():
        assert not set(TOKEN) & set(r["per_window"]["t1_10"])
    assert "token" not in (tmp_path / "report" / "report.md").read_text().lower()
    assert "tokens" not in rep and all(c["metric"] != "region_token_cosine" for c in rep["comparisons"])
    exits_nonzero(["score", "--tier", str(mtier), "--target", "grid_map", "--split", "val", "--lpips-dir", str(tmp_path),
                   "--out", str(tmp_path / "g"), "--token-metrics", "on"])
    assert cli(["score", "--tier", str(mtier), "--target", "grid_map", "--split", "val", "--lpips-dir", str(tmp_path),
                "--out", str(tmp_path / "g2")]) == 0                  # default for grid_map: off
    assert "token_cosine" not in (tmp_path / "g2" / "report.json").read_text()


def test_true_frame_encodings_are_cached_and_reused_by_a_second_run(mtier, tmp_path, fake_encoder):
    rgb, mask = truth_frames()
    arm = write_arm(tmp_path, "arm", rgb, mask, k=2)
    score(mtier, tmp_path, [arm], out="r1")
    first = fake_encoder.frames_encoded()
    n_true, n_gen = len(IDS), len(IDS) * 2 + len(IDS) * 4          # arm k=2, then four floors with k=1
    assert first == n_true + n_gen
    cache = sorted((mtier / "token_cache").glob("*"))
    assert [c.suffix for c in cache] == [".json", ".npz"] and cache[1].name.startswith("val__standin__1")
    side = json.loads(cache[0].read_text())
    import hashlib
    assert side["sha256"] == hashlib.sha256(cache[1].read_bytes()).hexdigest() and side["encoder"] == "standin"
    z = np.load(cache[1])
    assert list(z["ids"]) == IDS and z["tokens"].shape == (len(IDS), 28, 50, 8) and z["tokens"].dtype == np.float16
    fake_encoder.reset()
    rep1 = json.loads((tmp_path / "r1" / "report.json").read_text())
    rep2 = score(mtier, tmp_path, [arm], out="r2")
    assert fake_encoder.frames_encoded() == n_gen                     # no true frame encoded again
    assert rep2["rows"] == rep1["rows"]


def test_a_corrupted_cache_is_refused_and_a_changed_true_frame_is_re_encoded(mtier, tmp_path, fake_encoder):
    score(mtier, tmp_path)
    npz = next((mtier / "token_cache").glob("*.npz"))
    good = npz.read_bytes()
    npz.write_bytes(good[:-1] + bytes([good[-1] ^ 1]))
    exits_nonzero(["score", "--tier", str(mtier), "--split", "val", "--lpips-dir", str(tmp_path), "--out", str(tmp_path / "x")])
    npz.write_bytes(good)
    f = np.load(mtier / "val_frames.npz")
    f9 = f["f9"].copy()
    f9[2] = 255 - f9[2]                                              # t1_30 is a No Change Window: truth RGB stays black
    f9[0] = 255 - f9[0]                                              # t1_10's region (black patch) turns white: its true frame changes
    np.savez(mtier / "val_frames.npz", ids=f["ids"], f8=f["f8"], f9=f9)
    fake_encoder.reset()
    score(mtier, tmp_path, out="y")
    assert fake_encoder.frames_encoded() == 4 * len(IDS) + 1          # the floors, and only t1_10's true frame again
