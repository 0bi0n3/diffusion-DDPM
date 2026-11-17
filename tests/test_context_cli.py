"""`changemap context` and `changemap previous` through the CLI on a synthetic tier: tiny videos, a hand-made Frame
Latent store (written with the DiT study's own writer), fake CLIP and fake encoder (see conftest)."""
import hashlib
import json

import numpy as np
import pytest

from changemap import data
from changemap.cli import main as cli
from test_frames_cli import make_video
from latentdit.encode import StandInEncoder, encode_tutorial, read_shard

DIM = 8
T = 7


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair(regions, dropped=False):
    return {"regions": regions, "dropped": dropped, "no_change": not regions}


BOX_BIG, BOX_SMALL = [100, 100, 300, 200], [10, 10, 20, 20]       # pass / fail the 50/20 rule


def trow(wid, start, target_key, video, source, status="target", tutorial=T):
    return {"id": wid, "tutorial": tutorial, "video": str(video), "frames": list(range(start, start + 10)),
            "goal": "Make a scene", "subgoal": "Add a node.", "action": "Click Add", "evidence": "SECRET EVIDENCE",
            "target_key": target_key, "target_source": str(source), "target_status": status,
            "regions": [], "box_ok_50_20": []}


@pytest.fixture
def ctx(tmp_path):
    video = tmp_path / "v.mp4"
    make_video(video, seconds=12)
    src = tmp_path / "pixel_difference.json"
    src.write_text(json.dumps({"pairs": {
        "t7_7-8": pair([BOX_BIG, BOX_SMALL]), "t7_8-9": pair([BOX_BIG]), "t7_9-10": pair([], False),
        "t7_6-7": pair([BOX_BIG], dropped=True)}}))
    tier = tmp_path / "tier"
    tier.mkdir()
    rows = {"train": [trow("t7_a", 0, "t7_8-9", video, src),            # previous t7_7-8: two boxes, one passes
                      trow("t7_b", 1, "t7_9-10", video, src),           # previous t7_8-9
                      trow("t7_c", 0, "t7_7-8", video, src),            # previous t7_6-7: dropped
                      trow("t7_d", 0, "t7_6-7", video, src),            # previous t7_5-6: absent
                      trow("t7_gone", 0, "t7_9-10", video, src, "dropped")],
            "val": [trow("t7_v", 2, "t7_9-10", video, src)]}
    for s, r in rows.items():
        (tier / f"{s}_targets.jsonl").write_text("\n".join(json.dumps(x) for x in r) + "\n")
    # the store holds seconds 8 (window a, c, d: F8 = second 8) and 9 (window b: start 1) but not 10 (val window)
    store = tmp_path / "store"
    seconds = [8, 9]
    enc = StandInEncoder(grid=(28, 50), dim=DIM)
    encode_tutorial(T, str(video), seconds, enc, store)
    stats = tmp_path / "stats.npz"
    g = np.random.default_rng(3)
    np.savez(stats, mean=g.normal(size=DIM), std=g.uniform(0.5, 2, size=DIM))
    return {"tier": tier, "store": store, "stats": stats, "out": tmp_path / "out", "enc": enc, "video": video}


def run(ctx, split="train", *extra):
    return cli(["context", "--tier", str(ctx["tier"]), "--split", split, "--out", str(ctx["out"]),
                "--store", str(ctx["store"]), "--stats", str(ctx["stats"]), *extra])


def test_f8_comes_from_the_store_normalised_once(ctx):
    assert run(ctx) == 0
    z = np.load(ctx["out"] / "train_context.npz")
    assert list(z["ids"]) == ["t7_a", "t7_b", "t7_c", "t7_d"]          # the Dropped Pair has no row
    assert z["f8"].shape == (4, 28, 50, DIM) and z["f8"].dtype == np.float16
    sec, lat, _ = read_shard(ctx["store"], T)
    s = np.load(ctx["stats"])
    want = ((np.asarray(lat[list(sec).index(8)], np.float32) - s["mean"].astype(np.float32)) / s["std"].astype(np.float32))
    assert np.allclose(z["f8"][0].astype(np.float32), want, atol=2e-3)
    raw_b = np.asarray(lat[list(sec).index(9)], np.float32)
    assert not np.allclose(z["f8"][1].astype(np.float32), raw_b, atol=2e-3)   # not left raw
    assert np.allclose(z["f8"][1].astype(np.float32), (raw_b - s["mean"]) / s["std"], atol=2e-3)


def test_missing_window_is_encoded_fresh_with_the_same_encoder_and_normalised(ctx, fake_encoder):
    assert run(ctx, "val") == 0                                         # val window F8 = second 10: not in the store
    z = np.load(ctx["out"] / "val_context.npz")
    s = np.load(ctx["stats"])
    from latentdit.encode import decode_chunks
    frame = next(f for ks, f in decode_chunks(str(ctx["video"]), [10]))[0]
    want = (np.asarray(ctx["enc"].encode([frame])[0], np.float32) - s["mean"]) / s["std"]
    assert np.allclose(z["f8"][0].astype(np.float32), want, atol=2e-3)
    assert fake_encoder.frames_encoded() == 1
    m = json.loads((ctx["out"] / "context_manifest.json").read_text())["val"]["context"]
    assert (m["windows"], m["from_store"], m["encoded_fresh"], m["fresh_ids"]) == (1, 0, 1, ["t7_v"])


def test_manifest_counts_and_sha256(ctx, fake_encoder):
    assert run(ctx) == 0
    m = json.loads((ctx["out"] / "context_manifest.json").read_text())["train"]["context"]
    assert (m["windows"], m["from_store"], m["encoded_fresh"]) == (4, 4, 0)
    assert m["files"]["train_context.npz"]["sha256"] == sha(ctx["out"] / "train_context.npz")
    assert m["clip"] == {"id": "fake-clip", "revision": "rev1"}
    assert m["stats_sha256"] == sha(ctx["stats"])
    assert fake_encoder.frames_encoded() == 0                           # nothing needed encoding


def test_caption_is_goal_subgoal_action_and_never_evidence(ctx, fake_clip):
    assert run(ctx) == 0
    assert set(fake_clip.TEXTS) == {"Make a scene. Add a node. Click Add"}
    z = np.load(ctx["out"] / "train_context.npz")
    assert z["caption"].shape == (4, 77, DIM) and z["caption"].dtype == np.float16


def test_too_many_missing_windows_stop_with_the_count(ctx, capsys):
    with pytest.raises(SystemExit) as e:
        run(ctx, "val", "--max-fresh", "0")
    assert e.value.code == 2 and "1 Windows" in capsys.readouterr().err
    assert not (ctx["out"] / "val_context.npz").exists()


def test_a_store_made_by_another_encoder_is_refused(ctx, monkeypatch):
    info = ctx["store"] / "shards" / f"t{T}" / "shard.json"
    d = json.loads(info.read_text())
    d["encoder"] = "someone-else"
    info.write_text(json.dumps(d))
    with pytest.raises(SystemExit) as e:
        run(ctx, "val")                                                 # needs the encoder, ids differ
    assert e.value.code == 2


def test_test_split_needs_verdict(ctx, tmp_path):
    (ctx["tier"] / "test_targets.jsonl").write_text("")
    for cmd in ("context", "previous"):
        with pytest.raises(SystemExit) as e:
            cli([cmd, "--tier", str(ctx["tier"]), "--split", "test", "--out", str(ctx["out"])])
        assert e.value.code == 2


def test_previous_records_match_dit2vlm_lookup(ctx):
    assert cli(["previous", "--tier", str(ctx["tier"]), "--split", "train", "--out", str(ctx["out"])]) == 0
    prev = data.read_previous(ctx["out"], "train")
    assert list(prev) == ["t7_a", "t7_b", "t7_c", "t7_d"]
    assert prev["t7_a"] == {"regions": [BOX_BIG, BOX_SMALL], "box_ok_50_20": [True, False]}
    assert prev["t7_b"] == {"regions": [BOX_BIG], "box_ok_50_20": [True]}
    assert prev["t7_c"] == {"dropped": True}
    assert prev["t7_d"] is None
    m = json.loads((ctx["out"] / "context_manifest.json").read_text())["train"]["previous"]
    assert (m["windows"], m["with_previous"], m["dropped"], m["none"]) == (4, 2, 1, 1)
    assert m["files"]["train_previous.jsonl"]["sha256"] == sha(ctx["out"] / "train_previous.jsonl")
    # load_windows turns them into Grid Maps of the significant regions only
    (ctx["out"] / "train_targets.jsonl").write_text((ctx["tier"] / "train_targets.jsonl").read_text())
    w = {x.id: x for x in data.load_windows(ctx["out"], "train")}
    assert w["t7_a"].previous_regions == (tuple(map(float, BOX_BIG)),) and not w["t7_c"].has_previous


def test_both_commands_share_one_manifest(ctx):
    assert run(ctx) == 0
    assert cli(["previous", "--tier", str(ctx["tier"]), "--split", "train", "--out", str(ctx["out"])]) == 0
    m = json.loads((ctx["out"] / "context_manifest.json").read_text())
    assert set(m["train"]) == {"context", "previous"}
