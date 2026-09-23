"""Plain CNN vs C4 (p4) group-equivariant CNN for 4-class rotation prediction.

Layout for group tensors: [B, G*C, H, W] with flat index g*C + c, G=4.
Group convs are ordinary convs whose weights are materialised from a smaller shared weight, which is
exactly what escnn's .export() does - so the exported graph is a plain CNN with no custom ops.
Exact equivariance needs: square even-sized inputs, stride-1 k=3 'same' convs, avg-pool by 2, global pool.
"""
import torch, torch.nn as nn, torch.nn.functional as F

G = 4


class LiftConv(nn.Module):
    """Z2 -> p4. Output group g uses the base filter rotated by g*90 CCW."""
    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        self.c_out, self.k = c_out, k
        self.weight = nn.Parameter(torch.empty(c_out, c_in, k, k))
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(c_out))

    def materialise(self):
        w = torch.cat([torch.rot90(self.weight, g, dims=(-2, -1)) for g in range(G)], 0)
        return w, self.bias.repeat(G)

    def forward(self, x):
        w, b = self.materialise()
        return F.conv2d(x, w, b, padding=self.k // 2)


class GroupConv(nn.Module):
    """p4 -> p4. Output group g: rotate filters by g AND cyclically shift the input-group axis by g."""
    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        self.c_in, self.c_out, self.k = c_in, c_out, k
        self.weight = nn.Parameter(torch.empty(c_out, G, c_in, k, k))
        nn.init.kaiming_normal_(self.weight.view(c_out, G * c_in, k, k), nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(c_out))

    def materialise(self):
        parts = []
        for g in range(G):
            wg = torch.roll(self.weight, shifts=g, dims=1)        # wg[:, g'] = w[:, (g'-g) % G]
            wg = torch.rot90(wg, g, dims=(-2, -1))
            parts.append(wg.reshape(self.c_out, G * self.c_in, self.k, self.k))
        return torch.cat(parts, 0), self.bias.repeat(G)

    def forward(self, x):
        w, b = self.materialise()
        return F.conv2d(x, w, b, padding=self.k // 2)


class P4Norm(nn.Module):
    """GroupNorm over the 4 rotation blocks, with an affine SHARED across the group axis.

    A standard per-channel affine has independent gamma/beta for every (rotation, channel) pair, which
    destroys equivariance as soon as training makes them differ. Sharing them over the rotation axis
    keeps it exact. (Per-block statistics are fine: under an input rotation the blocks permute, and so
    do their statistics.)
    """
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.norm = nn.GroupNorm(G, G * c, affine=False)
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        # layout is [g*C + c], so tiling the C-length affine G times lands weight[c] on channel g*C+c.
        # Doing it as a plain [1, G*C, 1, 1] broadcast avoids two 5-D reshapes and their intermediates.
        w = self.weight.repeat(G).view(1, -1, 1, 1)
        b = self.bias.repeat(G).view(1, -1, 1, 1)
        return self.norm(x) * w + b


class P4Net(nn.Module):
    """Logit vector IS the group axis, so a single pass is exactly equivariant: rotating the input
    cyclically shifts the logits. No TTA needed by construction."""
    def __init__(self, widths=(8, 16, 32, 48), c_in=3):
        super().__init__()
        self.lift = LiftConv(c_in, widths[0])
        self.n1 = P4Norm(widths[0])
        blocks, norms = [], []
        for a, b in zip(widths, widths[1:]):
            blocks.append(GroupConv(a, b))
            norms.append(P4Norm(b))
        self.blocks, self.norms = nn.ModuleList(blocks), nn.ModuleList(norms)
        self.head = nn.Linear(widths[-1], 1, bias=False)
        self.widths = widths

    def forward(self, x):
        x = F.relu(self.n1(self.lift(x)))
        for blk, nrm in zip(self.blocks, self.norms):
            x = F.avg_pool2d(x, 2)
            x = F.relu(nrm(blk(x)))
        B, _, H, W = x.shape
        x = x.view(B, G, self.widths[-1], H, W).mean((-2, -1))   # [B, G, C]
        return self.head(x).squeeze(-1)                           # [B, G] logits


class PlainNet(nn.Module):
    """Matched feature-map count and FLOPs; 4x the parameters (no weight sharing across rotations)."""
    def __init__(self, widths=(32, 64, 128, 192), c_in=3):
        super().__init__()
        self.stem = nn.Conv2d(c_in, widths[0], 3, padding=1)
        self.n1 = nn.GroupNorm(G, widths[0])
        blocks, norms = [], []
        for a, b in zip(widths, widths[1:]):
            blocks.append(nn.Conv2d(a, b, 3, padding=1))
            norms.append(nn.GroupNorm(G, b))
        self.blocks, self.norms = nn.ModuleList(blocks), nn.ModuleList(norms)
        self.head = nn.Linear(widths[-1], G)

    def forward(self, x):
        x = F.relu(self.n1(self.stem(x)))
        for blk, nrm in zip(self.blocks, self.norms):
            x = F.avg_pool2d(x, 2)
            x = F.relu(nrm(blk(x)))
        return self.head(x.mean((-2, -1)))


class P4Depthwise(nn.Module):
    """Depthwise p4 conv: channel (g,c) uses base filter c rotated by g. One filter per base channel."""
    def __init__(self, c, k=3):
        super().__init__()
        self.c, self.k = c, k
        self.weight = nn.Parameter(torch.empty(c, 1, k, k))
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")

    def materialise(self):
        return torch.cat([torch.rot90(self.weight, g, dims=(-2, -1)) for g in range(G)], 0)

    def forward(self, x):
        return F.conv2d(x, self.materialise(), None, padding=self.k // 2, groups=G * self.c)


class P4Pointwise(nn.Module):
    """1x1 p4 conv. Mixes channels AND rotations; no spatial rotation needed since 1x1 is
    spatially isotropic, so only the group axis is cyclically shifted."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.c_in, self.c_out = c_in, c_out
        self.weight = nn.Parameter(torch.empty(c_out, G, c_in))
        nn.init.kaiming_normal_(self.weight.view(c_out, G * c_in), nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(c_out))

    def materialise(self):
        parts = [torch.roll(self.weight, shifts=g, dims=1).reshape(self.c_out, G * self.c_in)
                 for g in range(G)]
        return torch.cat(parts, 0).view(G * self.c_out, G * self.c_in, 1, 1), self.bias.repeat(G)

    def forward(self, x):
        w, b = self.materialise()
        return F.conv2d(x, w, b)


class P4MobileNet(nn.Module):
    """Depthwise-separable C4-equivariant net: the cheap architecture the efficiency case needs.

    Matched-FLOP relationship to a plain depthwise net is the same as for dense convs: base width w
    gives 4w feature maps and costs like a plain net of width 4w, with 4x fewer free parameters.
    """
    def __init__(self, widths=(16, 24, 48, 64, 96), c_in=3, expand=3):
        super().__init__()
        self.lift = LiftConv(c_in, widths[0])
        self.n0 = P4Norm(widths[0])
        blocks = []
        for a, b in zip(widths, widths[1:]):
            mid = a * expand
            blocks.append(nn.ModuleList([
                P4Pointwise(a, mid), P4Norm(mid),      # expand
                P4Depthwise(mid), P4Norm(mid),         # spatial
                P4Pointwise(mid, b), P4Norm(b),        # project
            ]))
        self.blocks = nn.ModuleList(blocks)
        self.widths = widths
        self.head = nn.Linear(widths[-1], 1, bias=False)

    def forward(self, x):
        x = F.relu(self.n0(self.lift(x)))
        for pw1, n1, dw, n2, pw2, n3 in self.blocks:
            x = F.avg_pool2d(x, 2)
            x = F.relu(n1(pw1(x)))
            x = F.relu(n2(dw(x)))
            x = n3(pw2(x))                              # linear bottleneck, no activation
        B, _, H, W = x.shape
        x = x.view(B, G, self.widths[-1], H, W).mean((-2, -1))
        return self.head(x).squeeze(-1)


class P4PointwiseLocal(nn.Module):
    """1x1 that mixes channels WITHIN each rotation block only (block-diagonal, weights shared
    across blocks). G times cheaper than P4Pointwise, but cannot combine evidence across
    orientations, so a net needs some full-mixing layers to reason about relative orientation."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.c_in, self.c_out = c_in, c_out
        self.weight = nn.Parameter(torch.empty(c_out, c_in))
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(c_out))

    def forward(self, x):
        w = self.weight.unsqueeze(-1).unsqueeze(-1).repeat(G, 1, 1, 1)
        return F.conv2d(x, w, self.bias.repeat(G), groups=G)


class P4MobileNetV2(nn.Module):
    """Sweepable variant: stem downsampling, expansion, depth, and which pointwise layers mix
    across rotations (`mix`: 'all' = every block, 'alt' = every other, 'none' = never)."""
    def __init__(self, widths=(8, 16, 24, 32, 48), c_in=3, expand=2, stem_pool=1, mix="all"):
        super().__init__()
        self.stem_pool = stem_pool
        self.lift = LiftConv(c_in, widths[0])
        self.n0 = P4Norm(widths[0])
        blocks = []
        for i, (a, b) in enumerate(zip(widths, widths[1:])):
            mid = a * expand
            full = (mix == "all") or (mix == "alt" and i % 2 == 0)
            PW = P4Pointwise if full else P4PointwiseLocal
            blocks.append(nn.ModuleList([PW(a, mid), P4Norm(mid),
                                         P4Depthwise(mid), P4Norm(mid),
                                         PW(mid, b), P4Norm(b)]))
        self.blocks = nn.ModuleList(blocks)
        self.widths = widths
        self.head = nn.Linear(widths[-1], 1, bias=False)

    def forward(self, x):
        if self.stem_pool > 1:
            x = F.avg_pool2d(x, self.stem_pool)
        x = F.relu(self.n0(self.lift(x)))
        for pw1, n1, dw, n2, pw2, n3 in self.blocks:
            x = F.avg_pool2d(x, 2)
            x = F.relu(n1(pw1(x)))
            x = F.relu(n2(dw(x)))
            x = n3(pw2(x))
        B, _, H, W = x.shape
        x = x.view(B, G, self.widths[-1], H, W).mean((-2, -1))
        return self.head(x).squeeze(-1)
