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


def write_stats(path, dim=8, mean=0.0, std=2.0):
    np.savez(path, mean=np.full(dim, mean, np.float32), std=np.full(dim, std, np.float32))
    return path


def test_token_mse_uses_latent_stats_and_the_report_says_which(mtier, tmp_path, fake_encoder):
    arm = noisy_arm(tmp_path, "arm", 60)
    raw = score(mtier, tmp_path, [arm], out="raw")
    stats = write_stats(tmp_path / "stats.npz")                       # (x - 0) / 2 -> MSE / 4
    norm = score(mtier, tmp_path, [arm], out="norm", extra=["--latent-stats", str(stats)])
    for wid in IDS:
        a, b = raw["rows"]["arm"]["per_window"][wid]["token_mse"], norm["rows"]["arm"]["per_window"][wid]["token_mse"]
        assert b == [pytest.approx(x / 4, rel=1e-3, abs=1e-12) for x in a]
        assert norm["rows"]["arm"]["per_window"][wid]["token_cosine"] == raw["rows"]["arm"]["per_window"][wid]["token_cosine"]
    assert raw["tokens"]["normalised"] is False and raw["tokens"]["latent_stats"] is None
    assert norm["tokens"]["normalised"] is True and norm["tokens"]["latent_stats"] == str(stats)
    assert norm["tokens"]["encoder"] == {"kind": "dinov3", "id": "standin", "revision": "1", "device": "cpu"}
    assert "unnormalised" in (tmp_path / "raw" / "report.md").read_text()
    assert "unnormalised" not in (tmp_path / "norm" / "report.md").read_text()
    assert fake_encoder.LOADED[0] == ("dinov3", "cpu")


def test_stats_with_the_wrong_dimension_are_refused(mtier, tmp_path):
    exits_nonzero(["score", "--tier", str(mtier), "--split", "val", "--lpips-dir", str(tmp_path), "--out", str(tmp_path / "x"),
                   "--latent-stats", str(write_stats(tmp_path / "s.npz", dim=5))])


def test_encoder_flag_and_device_reach_the_loader(mtier, tmp_path, fake_encoder):
    score(mtier, tmp_path, extra=["--encoder", "standin", "--encoder-device", "cpu"])
    assert fake_encoder.LOADED == [("standin", "cpu")]


def test_run_refuses_token_metrics_on_for_grid_map_before_training(mtier, tmp_path):
    out = tmp_path / "all"
    exits_nonzero(["run", "--tier", str(mtier), "--target", "grid_map", "--token-metrics", "on", "--steps", "1",
                   "--out", str(out)])
    assert not out.exists()


def test_bootstrap_on_region_token_cosine_for_the_h1_to_h5_pairs_with_higher_is_better(mtier, tmp_path):
    arms = {"unconditional": 90, "caption_only": 60, "frame_caption": 30}
    rep = score(mtier, tmp_path, [noisy_arm(tmp_path, a, v) for a, v in arms.items()])
    tok = {(c["a"], c["b"]): c for c in rep["comparisons"] if c["metric"] == "region_token_cosine"}
    pix = {(c["a"], c["b"]): c for c in rep["comparisons"] if c["metric"] == "region_lpips"}
    assert set(tok) == set(pix)                                       # same pairs as Region LPIPS
    assert {c["hypothesis"] for c in tok.values() if "hypothesis" in c} == {"H1", "H2", "H3", "H4", "H5"}
    assert all(tok[p].get("hypothesis") == pix[p].get("hypothesis") for p in pix)
    assert all((c["resamples"], c["seed"]) == (1000, 35) for c in tok.values())
    # less noise: lower LPIPS (negative diff) and higher cosine (positive diff)
    fc, co = ("frame_caption:single", "caption_only:single"), ("caption_only:single", "unconditional:single")
    for pair in (fc, co):
        assert pix[pair]["mean_diff"] < 0 < tok[pair]["mean_diff"]
    h4 = ("frame_caption:best_of_k", "frame_caption:single")
    assert pix[h4]["mean_diff"] <= 0 <= tok[h4]["mean_diff"]          # best of k: min LPIPS, max cosine
    r = rep["rows"]["frame_caption"]["summary"]["change"]
    assert r["best_of_k"]["region_token_cosine"] >= r["single"]["region_token_cosine"]
    assert r["best_of_k"]["token_mse"] <= r["single"]["token_mse"]
    md = (tmp_path / "report" / "report.md").read_text()
    assert "higher is better" in md and "Region Token Cosine" in md and "Pixel and token conclusions" in md


def test_hypothesis_readings_flag_disagreement_only():
    from changemap.score import hypothesis_readings

    def cmp(h, metric, lo, hi):
        return {"a": "x", "b": "y", "metric": metric, "hypothesis": h, "ci_lo": lo, "ci_hi": hi}
    got = hypothesis_readings([
        cmp("H1", "region_lpips", -.3, -.1), cmp("H1", "region_token_cosine", .1, .3),      # both support
        cmp("H2", "region_lpips", -.3, -.1), cmp("H2", "region_token_cosine", -.1, .3),     # pixel only
        cmp("H3", "region_lpips", -.3, .1), cmp("H3", "region_token_cosine", .1, .3),       # token only
        cmp("H4", "region_lpips", -.3, .1), cmp("H4", "region_token_cosine", -.1, .3),      # neither
        cmp("H5", "region_lpips", -.3, -.1)])                                               # no token reading: skipped
    assert [(r["hypothesis"], r["disagree"]) for r in got] == [("H1", False), ("H2", True), ("H3", True), ("H4", False)]
    zero = hypothesis_readings([cmp("H1", "region_lpips", -.3, 0.0), cmp("H1", "region_token_cosine", 0.0, .3)])
    assert zero[0]["pixel_supported"] is False and zero[0]["token_supported"] is False      # strict bounds


def test_report_lists_every_hypothesis_reading_and_names_the_disagreements(mtier, tmp_path):
    from changemap import report
    from changemap.score import hypothesis_readings
    arms = {"unconditional": 90, "caption_only": 60, "frame_caption": 30}
    rep = score(mtier, tmp_path, [noisy_arm(tmp_path, a, v) for a, v in arms.items()])
    assert rep["hypothesis_readings"] == hypothesis_readings(rep["comparisons"])
    assert [r["hypothesis"] for r in rep["hypothesis_readings"]] == ["H1", "H2", "H3", "H4", "H5"]
    md = (tmp_path / "report" / "report.md").read_text()
    n_dis = sum(r["disagree"] for r in rep["hypothesis_readings"])
    assert md.count("| DISAGREE |") == n_dis and md.count("| agree |") == 5 - n_dis
    # a forced disagreement is flagged and named, not resolved
    rep["hypothesis_readings"] = [dict(r, disagree=False) for r in rep["hypothesis_readings"]]
    rep["hypothesis_readings"][1] = dict(rep["hypothesis_readings"][1], pixel_supported=True, token_supported=False, disagree=True)
    out = report.render(rep)
    assert "Disagreement between pixel and token conclusions: H2." in out and "| H2 |" in out and "DISAGREE" in out
    rep["hypothesis_readings"][1]["disagree"] = False
    assert "No hypothesis has disagreeing" in report.render(rep)
