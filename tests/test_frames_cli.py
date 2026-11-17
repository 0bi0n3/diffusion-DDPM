"""`changemap frames`: decode F8/F9 of each Window from its video, through the CLI, on a tiny synthetic video."""
import json

import cv2
import numpy as np
import pytest

from changemap.cli import main as cli

FPS = 5
# one clearly different RGB colour per second (the video is written BGR)
COLOURS = [(250, 10, 10), (10, 250, 10), (10, 10, 250), (250, 250, 10), (250, 10, 250), (10, 250, 250),
           (130, 130, 130), (250, 130, 10), (10, 130, 250), (130, 10, 250), (60, 200, 90), (200, 60, 90)]


def make_video(path, size=(160, 90), seconds=12):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
    for s in range(seconds):
        for _ in range(FPS):
            w.write(np.full((size[1], size[0], 3), COLOURS[s][::-1], np.uint8))
    w.release()


def wrow(wid, video, start, status="target"):
    return {"id": wid, "video": str(video), "frames": list(range(start, start + 10)), "target_status": status,
            "regions": [], "box_ok_50_20": []}


@pytest.fixture
def ftier(tmp_path):
    v = tmp_path / "vid.mp4"
    make_video(v)
    d = tmp_path / "tier"
    d.mkdir()
    rows = {"train": [wrow("t1_a", v, 0), wrow("t1_dropped", v, 0, "dropped")],
            "val": [wrow("t2_a", v, 1), wrow("t2_gone", tmp_path / "nope.mp4", 0)],
            "test": [wrow("t3_a", v, 0)]}
    for s, r in rows.items():
        (d / f"{s}_targets.jsonl").write_text("\n".join(json.dumps(x) for x in r) + "\n")
    return d


def near(frame, colour):
    return np.abs(frame.astype(int) - np.array(colour)).max() <= 30


def test_frames_puts_the_right_second_in_f8_and_f9(ftier, tmp_path):
    out = tmp_path / "out"
    assert cli(["frames", "--tier", str(ftier), "--split", "train", "--out", str(out)]) == 0
    z = np.load(out / "train_frames.npz")
    assert list(z["ids"]) == ["t1_a"]                                  # the Dropped Pair is not decoded
    assert z["f8"].shape == (1, 224, 400, 3) and z["f9"].dtype == np.uint8
    assert near(z["f8"][0], COLOURS[8]) and near(z["f9"][0], COLOURS[9])   # frames[8] = 8, frames[9] = 9
    assert not near(z["f9"][0], COLOURS[8])


def test_frames_offset_window_and_undecodable_window_is_listed(ftier, tmp_path):
    out = tmp_path / "out"
    assert cli(["frames", "--tier", str(ftier), "--split", "val", "--out", str(out)]) == 0
    z = np.load(out / "val_frames.npz")
    assert list(z["ids"]) == ["t2_a"]
    assert near(z["f8"][0], COLOURS[9]) and near(z["f9"][0], COLOURS[10])  # frames start at 1: F8 = 9, F9 = 10
    missing = json.loads((out / "val_frames_missing.json").read_text())
    assert [m["id"] for m in missing] == ["t2_gone"]


def test_frames_seconds_past_the_end_are_undecodable(ftier, tmp_path):
    rows = [wrow("t9_late", json.loads((ftier / "train_targets.jsonl").read_text().splitlines()[0])["video"], 5)]
    (ftier / "train_targets.jsonl").write_text(json.dumps(rows[0]) + "\n")      # F9 = second 14 of a 12 s video
    out = tmp_path / "out"
    assert cli(["frames", "--tier", str(ftier), "--split", "train", "--out", str(out)]) == 0
    assert [m["id"] for m in json.loads((out / "train_frames_missing.json").read_text())] == ["t9_late"]


def test_frames_refuses_test_without_verdict(ftier, tmp_path):
    with pytest.raises(SystemExit) as e:
        cli(["frames", "--tier", str(ftier), "--split", "test", "--out", str(tmp_path / "o")])
    assert e.value.code == 2 and not (tmp_path / "o").exists()


def test_frames_downscales_large_video_like_the_regions(tmp_path):
    v = tmp_path / "big.mp4"
    make_video(v, size=(2560, 1440), seconds=2)
    d = tmp_path / "tier"
    d.mkdir()
    (d / "val_targets.jsonl").write_text(json.dumps(wrow("t4_a", v, 0)) .replace("[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]", "[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]") + "\n")
    out = tmp_path / "o"
    assert cli(["frames", "--tier", str(d), "--split", "val", "--out", str(out)]) == 0
    z = np.load(out / "val_frames.npz")
    assert near(z["f8"][0], COLOURS[0]) and near(z["f9"][0], COLOURS[1])
