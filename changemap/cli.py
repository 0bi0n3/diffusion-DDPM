"""`changemap train | sample | score | run | frames | context | previous | gate`: the one seam."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import targets, tokens

DEVICE_HELP = "torch device (default cpu; use cuda only for cluster jobs)"


def token_options(s) -> None:
    s.add_argument("--token-metrics", choices=("on", "off"), default=None,
                   help="token-space scoring (default: on for masked_frame; refused for grid_map)")
    s.add_argument("--encoder", choices=tokens.KINDS, default="dinov3",
                   help="frozen encoder for token metrics: the DiT study's DINOv3 path, or its stand-in")
    s.add_argument("--encoder-device", default="cpu", help="torch device of the encoder (default cpu)")
    s.add_argument("--latent-stats", type=Path,
                   help="the DiT study's stats.npz (mean, std); token MSE is raw tokens without it")


def token_kwargs(a) -> dict:
    return {"token_metrics": None if a.token_metrics is None else a.token_metrics == "on", "encoder": a.encoder,
            "encoder_device": a.encoder_device, "latent_stats": a.latent_stats}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="changemap")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s, split=False):
        s.add_argument("--tier", required=True, type=Path)
        s.add_argument("--target", choices=tuple(targets.TARGETS), default=targets.DEFAULT,
                       help="what the model generates (see changemap.targets)")
        if split:
            s.add_argument("--split", default="val")
            s.add_argument("--verdict", action="store_true", help="allow reading the test split")

    t = sub.add_parser("train")
    common(t)
    t.add_argument("--arm", default="unconditional")
    t.add_argument("--steps", type=int, required=True)
    t.add_argument("--out", required=True, type=Path)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--hidden", type=int, default=384)
    t.add_argument("--depth", type=int, default=12)
    t.add_argument("--heads", type=int, default=6)
    t.add_argument("--seed", type=int, default=35)
    t.add_argument("--device", default="cpu", help=DEVICE_HELP)
    from . import train as train_mod
    train_mod.add_arguments(t)

    s = sub.add_parser("sample")
    common(s, True)
    s.add_argument("--run", required=True, type=Path)
    s.add_argument("--k", type=int, default=8)
    s.add_argument("--steps", type=int, default=50)
    s.add_argument("--seed", type=int, default=35)
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--device", default="cpu", help=DEVICE_HELP)

    c = sub.add_parser("score")
    common(c, True)
    c.add_argument("--samples", action="append", default=[], type=Path,
                   help="a samples dir or .npz; repeat for several arms (none: floors only)")
    c.add_argument("--out", required=True, type=Path)
    c.add_argument("--lpips-dir", type=Path, help="folder with vgg16-397923af.pth and vgg.pth (env CHANGEMAP_LPIPS_DIR)")
    c.add_argument("--device", default="cpu", help=DEVICE_HELP)
    token_options(c)

    f = sub.add_parser("frames", help="decode the frame cache <out>/<split>_frames.npz from the tier's videos")
    f.add_argument("--tier", required=True, type=Path)
    f.add_argument("--split", choices=("train", "val", "test"), required=True)
    f.add_argument("--verdict", action="store_true", help="allow reading the test split")
    f.add_argument("--out", required=True, type=Path)
    f.add_argument("--workers", type=int, default=4, help="decoding processes, one video each (default 4)")

    for name, helptext in (("context", "write <out>/<split>_context.npz (F8 tokens, CLIP captions)"),
                           ("previous", "write <out>/<split>_previous.jsonl (the F7 -> F8 Step Pair per Window)")):
        x = sub.add_parser(name, help=helptext)
        x.add_argument("--tier", required=True, type=Path)
        x.add_argument("--split", choices=("train", "val", "test"), required=True)
        x.add_argument("--verdict", action="store_true", help="allow reading the test split")
        x.add_argument("--out", required=True, type=Path)
        if name == "context":
            x.add_argument("--store", action="append", required=True, type=Path,
                           help="DiT study Frame Latent store folder (shards/t<T>/...); repeat, first hit wins")
            x.add_argument("--stats", required=True, type=Path, help="the DiT study's stats.npz (mean, std)")
            x.add_argument("--encoder", choices=tokens.KINDS, default="dinov3", help="encoder for Windows missing from the store")
            x.add_argument("--max-fresh", type=int, default=50, help="most Windows encoded fresh (default 50)")
            x.add_argument("--device", default="cpu", help="torch device of the encoders (default cpu)")
            x.add_argument("--verify-store", action="store_true", help="sha256 every shard read (slow: the store is large)")

    r = sub.add_parser("run")
    common(r, True)
    r.add_argument("--arms", nargs="+", default=["unconditional"])
    r.add_argument("--steps", type=int, required=True)
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--batch", type=int, default=64)
    r.add_argument("--lr", type=float, default=1e-4)
    r.add_argument("--hidden", type=int, default=384)
    r.add_argument("--depth", type=int, default=12)
    r.add_argument("--heads", type=int, default=6)
    r.add_argument("--k", type=int, default=8)
    r.add_argument("--sample-steps", type=int, default=50)
    r.add_argument("--seed", type=int, default=35)
    r.add_argument("--lpips-dir", type=Path, help="folder with the LPIPS weights (env CHANGEMAP_LPIPS_DIR)")
    r.add_argument("--device", default="cpu", help=DEVICE_HELP)
    token_options(r)

    g = sub.add_parser("gate", help="check a run's training log for loss or gradient-norm collapse")
    g.add_argument("--run", required=True, type=Path, help="run directory (containing log.json) or path to log.json")
    g.add_argument("--max-loss-ratio", type=float, default=10.0,
                   help="fail if training loss exceeds this multiple of its running minimum (default 10)")
    g.add_argument("--max-grad-norm", type=float, default=100.0,
                   help="fail if pre-clip gradient norm exceeds this value (default 100)")
    return p


def _npz(paths):
    out = []
    for p in paths:
        out += sorted(p.glob("*.npz")) if p.is_dir() else [p]
    return out


def main(argv=None) -> int:
    p = _parser()
    a = p.parse_args(argv)
    from . import score as score_mod, sample as sample_mod, train as train_mod
    try:
        if a.cmd == "train":
            train_mod.train(a.tier, a.arm, a.steps, a.out, a.batch, a.lr, a.hidden, a.depth, a.heads, a.seed, a.device,
                            train_mod.options_from(a), target=a.target)
        elif a.cmd == "sample":
            sample_mod.sample(a.run, a.tier, a.split, a.k, a.steps, a.seed, a.out, a.verdict, a.device, target=a.target)
        elif a.cmd == "frames":
            from . import frames as frames_mod
            res = frames_mod.build_cache(a.tier, a.split, a.out, a.verdict, a.workers)
            print(f"{res['path']}: {res['n']} Windows, {len(res['missing'])} undecodable")
        elif a.cmd == "context":
            from . import context as context_mod
            e = context_mod.build_context(a.tier, a.split, a.out, a.store, a.stats, a.verdict, a.encoder, a.device,
                                          a.device, a.max_fresh, a.verify_store)
            print(f"{a.split}: {e['windows']} Windows, {e['from_store']} from the store, {e['encoded_fresh']} encoded fresh")
        elif a.cmd == "previous":
            from . import context as context_mod
            e = context_mod.build_previous(a.tier, a.split, a.out, a.verdict)
            print(f"{a.split}: {e['windows']} Windows, {e['with_previous']} with a previous pair, "
                  f"{e['dropped']} dropped, {e['none']} none")
        elif a.cmd == "score":
            score_mod.score(_npz(a.samples), a.tier, a.split, a.out, a.verdict, a.lpips_dir, a.device, target=a.target,
                            **token_kwargs(a))
        elif a.cmd == "gate":
            from . import gate as gate_mod
            res = gate_mod.check(a.run, a.max_loss_ratio, a.max_grad_norm)
            if not res.ok:
                print(f"FAIL: {res.reason}", file=sys.stderr)
                return 1
            print(f"PASS: {res.reason}")
        else:
            if a.split == "test" and not a.verdict:
                raise PermissionError("the test split is read only for a verdict: pass --verdict")
            if a.target == "grid_map" and a.token_metrics == "on":
                raise ValueError(f"--token-metrics on is not available for target {a.target!r}")
            if a.target == "map_latent" and a.token_metrics == "off":
                raise ValueError("--token-metrics off is not available for target 'map_latent'")
            samples = []
            for arm in a.arms:
                run = a.out / "runs" / arm
                train_mod.train(a.tier, arm, a.steps, run, a.batch, a.lr, a.hidden, a.depth, a.heads, a.seed, a.device,
                                target=a.target)
                samples.append(sample_mod.sample(run, a.tier, a.split, a.k, a.sample_steps, a.seed,
                                                 a.out / "samples" / arm, a.verdict, a.device, target=a.target))
            score_mod.score(samples, a.tier, a.split, a.out / "report", a.verdict, a.lpips_dir, a.device,
                            target=a.target, **token_kwargs(a))
    except (PermissionError, ValueError, FileNotFoundError) as e:
        p.exit(2, f"changemap: error: {e}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
