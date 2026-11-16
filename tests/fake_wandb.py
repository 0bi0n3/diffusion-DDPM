"""A stand-in for the `wandb` module: records init kwargs, define_metric calls and logged dicts. No network.
`fail` names the calls that raise (any of "init", "define_metric", "log", "finish").

Copied from the DiT latent study: /mnt/hitchcock/scratch/oberon/DiT-experiments tests/fake_wandb.py,
commit 217a3630556fc0ca638298e6553c3898f84412a4.
"""
import types


class FakeWandb(types.ModuleType):
    def __init__(self, fail=()):
        super().__init__("wandb")
        self.fail = set(fail)
        self.inits, self.metrics, self.logged, self.finished = [], [], [], 0

    def init(self, **kw):
        self.inits.append(kw)
        if "init" in self.fail:
            raise ConnectionError("fake: init failed")
        return _Run(self)


class _Run:
    def __init__(self, w):
        self.w = w

    def define_metric(self, *a, **kw):
        if "define_metric" in self.w.fail:
            raise ConnectionError("fake: define_metric failed")
        self.w.metrics.append((a, kw))

    def log(self, data, **kw):
        assert not kw, "log must not be given step="
        if "log" in self.w.fail:
            raise ConnectionError("fake: log failed")
        self.w.logged.append(dict(data))

    def finish(self):
        if "finish" in self.w.fail:
            raise ConnectionError("fake: finish failed")
        self.w.finished += 1
