"""Freeze p4 group convs into plain convs for deployment.

models.py materialises rotated weights inside forward(), so the rot90/roll calls run on every
inference and block ONNX export. At inference the materialised weights are constant, so bake them
once into an ordinary nn.Conv2d.
"""
import torch, torch.nn as nn
import models as M


class BakedConv(nn.Module):
    def __init__(self, w, b, padding, groups):
        super().__init__()
        self.register_buffer("w", w)
        self.register_buffer("b", b if b is not None else torch.zeros(0))
        self.padding, self.groups, self.has_b = padding, groups, b is not None

    def forward(self, x):
        return nn.functional.conv2d(x, self.w, self.b if self.has_b else None,
                                    padding=self.padding, groups=self.groups)


def _bake_one(m):
    out = m.materialise()
    w, b = out if isinstance(out, tuple) else (out, None)
    if isinstance(m, M.P4Depthwise):
        return BakedConv(w.detach(), None, m.k // 2, M.G * m.c)
    if isinstance(m, M.P4Pointwise):
        return BakedConv(w.detach(), b.detach(), 0, 1)
    return BakedConv(w.detach(), b.detach(), m.k // 2, 1)


def bake(net):
    net = net.eval()
    with torch.no_grad():
        for parent in net.modules():
            for name, child in list(parent.named_children()):
                if hasattr(child, "materialise"):
                    setattr(parent, name, _bake_one(child))
    return net
