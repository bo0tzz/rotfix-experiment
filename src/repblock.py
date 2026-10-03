"""RepVGG-style structural re-parameterisation for C4 group convs.

Training uses two parallel equivariant branches (3x3 group conv + 1x1 group pointwise); at export
they fold into a single 3x3 group conv, so inference cost is unchanged. Both branches are
C4-equivariant and equivariance is closed under linear combination, so the sum is too.

The fused weights are produced by materialise(), which is the same hook bake.py already dispatches
on - so export needs no changes.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import models as M

G = M.G


class RepGroupConv(nn.Module):
    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        assert k == 3
        self.c_in, self.c_out, self.k = c_in, c_out, k
        self.conv3 = M.GroupConv(c_in, c_out, k)
        self.conv1 = M.P4Pointwise(c_in, c_out)

    def materialise(self):
        w3, b3 = self.conv3.materialise()
        w1, b1 = self.conv1.materialise()
        w = w3.clone()
        w[:, :, 1:2, 1:2] += w1
        return w, b3 + b1

    def forward(self, x):
        if self.training:
            return self.conv3(x) + self.conv1(x)
        w, b = self.materialise()
        return F.conv2d(x, w, b, padding=self.k // 2)


class RepP4Net(nn.Module):
    """P4Net with each GroupConv replaced by a RepGroupConv. Identical inference graph once baked."""
    def __init__(self, widths=(16, 32, 64, 96, 128), c_in=3):
        super().__init__()
        self.lift = M.LiftConv(c_in, widths[0])
        self.n1 = M.P4Norm(widths[0])
        blocks, norms = [], []
        for a, b in zip(widths, widths[1:]):
            blocks.append(RepGroupConv(a, b))
            norms.append(M.P4Norm(b))
        self.blocks, self.norms = nn.ModuleList(blocks), nn.ModuleList(norms)
        self.head = nn.Linear(widths[-1], 1, bias=False)
        self.widths = widths

    def forward(self, x):
        x = F.relu(self.n1(self.lift(x)))
        for blk, nrm in zip(self.blocks, self.norms):
            x = F.avg_pool2d(x, 2)
            x = F.relu(nrm(blk(x)))
        B, _, H, W = x.shape
        x = x.view(B, G, self.widths[-1], H, W).mean((-2, -1))
        return self.head(x).squeeze(-1)
