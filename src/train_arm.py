"""Generic arm: baseline p4_big with optional label smoothing and a swappable manifest.

Matches v2-big12 exactly (p4_big widths, res 224, bs 64, 12 epochs, 1.09M v2 data); the only
change is that each GroupConv becomes a RepGroupConv (3x3 + 1x1, folded at export). Inference cost
is provably unchanged - verified max|diff| 0.00e+00 through bake(), same deployed tensor count.
"""
import json, os, random, time, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, "/src")
import models as M
from repblock import RepP4Net

W = Path("/work")
ARCH = os.environ.get("ARCH", "rep")
RES, EPOCHS, BS = int(os.environ.get("RES", 224)), int(os.environ.get("EPOCHS", 12)), int(os.environ.get("BS", 64))
IMGDIR, MANIFEST = os.environ.get("IMGDIR", "oid/img_v2"), os.environ.get("MANIFEST", "oid/manifest_v2.jsonl")
RUN = os.environ.get("RUN_ID", "v2-rep12")
LS = float(os.environ.get("LABEL_SMOOTH", "0.0"))
OUT = W / f"runs/{RUN}"; OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda"
MEAN = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(dev)
STD  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(dev)
def log(*a): print(*a, flush=True)

class OID(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        im = Image.open(W / IMGDIR / f"{r['id']}.jpg").convert("RGB")
        if r["rot"]: im = im.rotate(r["rot"], expand=True)
        im = im.resize((RES, RES), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(im).copy()).permute(2,0,1)
        if random.random() < 0.5: x = torch.flip(x, dims=(-1,))
        k = random.randrange(4)
        return torch.rot90(x, k, dims=(-2,-1)).contiguous(), k

class LibStored(Dataset):
    def __init__(self, ids, truth): self.ids, self.truth = ids, truth
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        im = Image.open(W/"testset/prev_raw"/f"{self.ids[i]}.jpg").convert("RGB").resize((RES,)*2, Image.BICUBIC)
        return torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).contiguous(), self.truth.get(self.ids[i],0)//90

def norm(xb): return (xb.float().div(255) - MEAN) / STD

def main():
    rows, seen = [], set()
    for l in open(W / MANIFEST):
        r = json.loads(l)
        if r["id"] not in seen: seen.add(r["id"]); rows.append(r)
    random.Random(0).shuffle(rows)
    val, tr = rows[:5000], rows[5000:]
    truth = json.load(open(W/"testset/ground_truth.json"))["truth"]
    lib_ids = sorted(p.stem for p in (W/"testset/prev_raw").glob("*.jpg"))
    WIDTHS = {"p4_big": (16,32,64,96,128), "p4_scratch": (8,16,32,48,64)}
    net = (RepP4Net() if ARCH == "rep" else M.P4Net(widths=WIDTHS[ARCH])).to(dev)
    log(f"{RUN}: arch={ARCH} ls={LS} manifest={MANIFEST} train {len(tr)} lib {len(lib_ids)} res {RES} bs {BS} ep {EPOCHS} "
        f"| train-params {sum(p.numel() for p in net.parameters()):,}")

    nw = min(16, os.cpu_count())
    tl = DataLoader(OID(tr), batch_size=BS, shuffle=True, num_workers=nw, pin_memory=True,
                    drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=EPOCHS*len(tl), pct_start=0.25)
    scaler = torch.amp.GradScaler("cuda")
    hist = []
    for ep in range(EPOCHS):
        net.train(); t0, tot, cnt = time.time(), 0.0, 0
        for xb, yb in tl:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(net(norm(xb)), yb, label_smoothing=LS)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()*len(xb); cnt += len(xb)
        torch.save(net.state_dict(), OUT/"model.pt")
        hist.append({"epoch": ep, "loss": tot/cnt, "sec": time.time()-t0})
        log(f"  ep{ep:>2} loss={tot/cnt:.4f} {time.time()-t0:.0f}s")

    log("final: dumping library posteriors (stored orientation)")
    net.eval(); P, Y = [], []
    dl = DataLoader(LibStored(lib_ids, truth), batch_size=128, shuffle=False, num_workers=nw)
    with torch.inference_mode():
        for xb, yb in dl:
            P.append(F.softmax(net(norm(xb.to(dev))).float(), 1).cpu()); Y.append(yb)
    P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy()
    (W/"probs").mkdir(exist_ok=True)
    np.savez(W/f"probs/{RUN}.npz", p=P, y=Y, ids=np.array(lib_ids))
    json.dump({"arch": ARCH, "label_smooth": LS, "manifest": MANIFEST, "params": sum(p.numel() for p in net.parameters()), "res": RES,
               "bs": BS, "hist": hist}, open(OUT/"hist.json","w"))
    log(f"{RUN}: 4-way acc on stored = {float((P.argmax(1)==Y).mean()):.4f}")
    log("REP_DONE")

if __name__ == "__main__":
    main()
