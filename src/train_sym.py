"""Ambiguity-aware targets, SYMMETRY-MATCHED.

v2 of the objective fix.

For an image with no canonical orientation, hard cross-entropy against a randomly applied k teaches
the network to be confident about an arbitrary answer. Under C4 equivariance the only self-consistent
target for such an image is UNIFORM - rotating the input cyclically shifts the logits, so no single
class can be correct at all four rotations; only a tie is invariant.

So: hard CE for normal images, KL-to-uniform for the ambiguous set identified by out-of-sample
2-fold confident learning. Everything else matches v2-big12 exactly (p4_big, res 224, bs 64, 12
epochs, same 1.09M) so this isolates the objective as a single variable.

SOFT_W scales the ambiguous term; 1.0 weights each ambiguous image like any other.
"""
import json, os, random, time, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, "/src")
import models as M

W = Path("/work")
RES, EPOCHS, BS = int(os.environ.get("RES", 224)), int(os.environ.get("EPOCHS", 12)), int(os.environ.get("BS", 64))
SOFT_W = float(os.environ.get("SOFT_W", "1.0"))
ARCH = os.environ.get("ARCH", "p4_big")
UP_BIAS = os.environ.get("UP_BIAS", "0") == "1"
IMGDIR, MANIFEST = os.environ.get("IMGDIR", "oid/img_v2"), os.environ.get("MANIFEST", "oid/manifest_v2.jsonl")
RUN = os.environ.get("RUN_ID", "v2-soft12")
OUT = W / f"runs/{RUN}"; OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda"
MEAN = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(dev)
STD  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(dev)
def log(*a): print(*a, flush=True)

class UprightBias(torch.nn.Module):
    """One scalar on the upright logit - the class prior the equivariant head cannot otherwise hold."""
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.b = torch.nn.Parameter(torch.zeros(1))
    def forward(self, x):
        o = self.net(x)
        return o + F.pad(self.b.expand(1), (0, 3))

class OID(Dataset):
    """sym: 0 = normal (hard label), 1 = uniform (true C4 symmetry),
    2 = opposite-pair with offset 0 (plausible set {0,180}), 3 = pair with offset 1 ({90,270}).

    The target must rotate with the input: a pair {o, o+2} in the canonical frame becomes
    {o+k, o+2+k} once the image is rotated by k. Uniform is shift-invariant so needs no offset."""
    def __init__(self, rows, amb): self.rows, self.amb = rows, amb
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        im = Image.open(W / IMGDIR / f"{r['id']}.jpg").convert("RGB")
        if r["rot"]: im = im.rotate(r["rot"], expand=True)
        im = im.resize((RES, RES), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(im).copy()).permute(2,0,1)
        if random.random() < 0.5: x = torch.flip(x, dims=(-1,))
        k = random.randrange(4)
        return torch.rot90(x, k, dims=(-2,-1)).contiguous(), k, self.amb.get(r["id"], 0)

class LibStored(Dataset):
    def __init__(self, ids, truth): self.ids, self.truth = ids, truth
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        im = Image.open(W/"testset/prev_raw"/f"{self.ids[i]}.jpg").convert("RGB").resize((RES,)*2, Image.BICUBIC)
        return torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).contiguous(), self.truth.get(self.ids[i],0)//90

def norm(xb): return (xb.float().div(255) - MEAN) / STD

def main():
    amb = {}
    for l in open(W / os.environ.get("AMB_UNIFORM", "amb_uniform_ids.txt")):
        if l.strip(): amb[l.strip()] = 1
    for l in open(W / os.environ.get("AMB_PAIR", "amb_pair_ids.txt")):
        parts = l.split()
        if len(parts) == 2: amb[parts[0]] = 2 + int(parts[1])
    rows, seen = [], set()
    for l in open(W / MANIFEST):
        r = json.loads(l)
        if r["id"] not in seen: seen.add(r["id"]); rows.append(r)
    random.Random(0).shuffle(rows)
    tr = rows[5000:]
    truth = json.load(open(W/"testset/ground_truth.json"))["truth"]
    lib_ids = sorted(p.stem for p in (W/"testset/prev_raw").glob("*.jpg"))
    n_amb = sum(1 for r in tr if r["id"] in amb)
    n_uni = sum(1 for r in tr if amb.get(r["id"]) == 1)
    n_pair = n_amb - n_uni
    WIDTHS = {'p4_big': (16,32,64,96,128), 'p4_scratch': (8,16,32,48,64)}[ARCH]
    net = M.P4Net(widths=WIDTHS)
    if UP_BIAS: net = UprightBias(net)
    net = net.to(dev)
    log(f"{RUN}: train {len(tr)} (ambiguous {n_amb} = {100*n_amb/len(tr):.2f}%: "
        f"{n_uni} uniform, {n_pair} pair) "
        f"arch={ARCH} soft_w={SOFT_W} up_bias={UP_BIAS} bs {BS} ep {EPOCHS}")

    nw = min(16, os.cpu_count())
    tl = DataLoader(OID(tr, amb), batch_size=BS, shuffle=True, num_workers=nw, pin_memory=True,
                    drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=EPOCHS*len(tl), pct_start=0.25)
    scaler = torch.amp.GradScaler("cuda")
    hist = []
    for ep in range(EPOCHS):
        net.train(); t0, tot, cnt = time.time(), 0.0, 0
        for xb, yb, mb in tl:
            xb, yb, mb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True), mb.to(dev)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = net(norm(xb))
                hard = mb == 0
                loss = out.new_zeros(())
                if hard.any():
                    loss = loss + F.cross_entropy(out[hard], yb[hard]) * hard.sum() / len(xb)
                soft = ~hard
                if soft.any():
                    lsm = F.log_softmax(out[soft].float(), 1)
                    t = torch.zeros_like(lsm)
                    ks, ms = yb[soft], mb[soft]
                    uni = ms == 1
                    if uni.any():
                        t[uni] = 0.25
                    pr = ~uni
                    if pr.any():
                        # pair {o+k, o+2+k}; note yb IS k, so the applied rotation is already in it
                        off = (ms[pr] - 2).long()
                        a = (ks[pr] + off) % 4
                        idx = torch.arange(len(t), device=t.device)[pr]
                        t[idx, a] = 0.5
                        t[idx, (a + 2) % 4] = 0.5
                    loss = loss + SOFT_W * (-(t * lsm).sum(1).mean()) * soft.sum() / len(xb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()*len(xb); cnt += len(xb)
        torch.save(net.state_dict(), OUT/"model.pt")
        hist.append({"epoch": ep, "loss": tot/cnt, "sec": time.time()-t0})
        log(f"  ep{ep:>2} loss={tot/cnt:.4f} {time.time()-t0:.0f}s"
            + (f" up_bias={float(net.b.item()):+.3f}" if UP_BIAS else ""))

    log("final: dumping library posteriors (stored orientation)")
    net.eval(); P, Y = [], []
    for xb, yb in DataLoader(LibStored(lib_ids, truth), batch_size=128, shuffle=False, num_workers=nw):
        with torch.inference_mode():
            P.append(F.softmax(net(norm(xb.to(dev))).float(), 1).cpu()); Y.append(yb)
    P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy()
    (W/"probs").mkdir(exist_ok=True)
    np.savez(W/f"probs/{RUN}.npz", p=P, y=Y, ids=np.array(lib_ids))
    json.dump({"arch": "p4_big", "soft_w": SOFT_W, "up_bias": UP_BIAS, "res": RES, "bs": BS,
               "n_ambiguous": n_amb, "hist": hist}, open(OUT/"hist.json","w"))
    log(f"{RUN}: 4-way acc on stored = {float((P.argmax(1)==Y).mean()):.4f}")
    log("SOFT_DONE")

if __name__ == "__main__":
    main()
