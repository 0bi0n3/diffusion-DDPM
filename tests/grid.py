"""`main` for tests of the Grid Map target: the CLI default is now `masked_frame`, so these tests ask for `grid_map`."""
from changemap.cli import main as _main


def main(argv=None):
    argv = list(argv)
    if "--target" not in argv:
        argv[1:1] = ["--target", "grid_map"]
    return _main(argv)
