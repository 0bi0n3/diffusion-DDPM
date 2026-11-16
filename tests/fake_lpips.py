"""Tiny stand-in for `latentdit.lpips`, same call contract and structure, small random-init convolutions (seeded).

`load(dir)` ignores the folder. Selected in tests with CHANGEMAP_LPIPS_MODULE=fake_lpips; nothing is downloaded."""
import torch
import torch.nn as nn

CHANNELS = (4, 6)
STRIDES = (16, 2)


class FakeLpips(nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.manual_seed(0)
        self.register_buffer("shift", torch.tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer("scale", torch.tensor([.458, .448, .450])[None, :, None, None])
        self.convs = nn.ModuleList([nn.Conv2d(3, CHANNELS[0], 16, STRIDES[0], 0), nn.Conv2d(CHANNELS[0], CHANNELS[1], 3, STRIDES[1], 1)])
        self.lins = nn.ModuleList(nn.Sequential(nn.Identity(), nn.Conv2d(c, 1, 1, bias=False)) for c in CHANNELS)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.5, generator=g)
        for lin in self.lins:
            lin[1].weight.data.abs_()          # non-negative heads, as in LPIPS
        for p in self.parameters():
            p.requires_grad = False

    def _features(self, x):
        out = []
        for conv in self.convs:
            x = torch.relu(conv(x))
            out.append(x)
        return out

    def forward(self, x, y):
        fx, fy = self._features((x - self.shift) / self.scale), self._features((y - self.shift) / self.scale)
        total = 0
        for a, b, lin in zip(fx, fy, self.lins):
            a, b = a / (a.pow(2).sum(1, keepdim=True).sqrt() + 1e-10), b / (b.pow(2).sum(1, keepdim=True).sqrt() + 1e-10)
            total = total + lin((a - b) ** 2).mean([2, 3], keepdim=True)
        return total.mean()


def load(directory=None):
    return FakeLpips().eval()
