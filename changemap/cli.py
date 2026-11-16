"""`changemap train | sample | score | run`: the one seam."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import targets

DEVICE_HELP = "torch device (default cpu; use cuda only for cluster jobs)"


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
                            train_mod.options_from(a), a.target)
        elif a.cmd == "sample":
            sample_mod.sample(a.run, a.tier, a.split, a.k, a.steps, a.seed, a.out, a.verdict, a.device, target=a.target)
        elif a.cmd == "score":
            score_mod.score(_npz(a.samples), a.tier, a.split, a.out, a.verdict, a.lpips_dir, a.device, target=a.target)
        else:
            if a.split == "test" and not a.verdict:
                raise PermissionError("the test split is read only for a verdict: pass --verdict")
            samples = []
            for arm in a.arms:
                run = a.out / "runs" / arm
                train_mod.train(a.tier, arm, a.steps, run, a.batch, a.lr, a.hidden, a.depth, a.heads, a.seed, a.device,
                                target=a.target)
                samples.append(sample_mod.sample(run, a.tier, a.split, a.k, a.sample_steps, a.seed,
                                                 a.out / "samples" / arm, a.verdict, a.device, target=a.target))
            score_mod.score(samples, a.tier, a.split, a.out / "report", a.verdict, a.lpips_dir, a.device,
                            target=a.target)
    except (PermissionError, ValueError, FileNotFoundError) as e:
        p.exit(2, f"changemap: error: {e}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
