"""Score every checkpoint on the held-out OID validation split - the public-only evaluation.

Two variants from the same images:
  natural   - the image exactly as stored, label from Google's Rotation field.
              y = (4 - rot//90) % 4, because canonical = PIL.rotate(stored, rot) = rot90(stored, rot//90).
              Base rate ~1.3% rotated. This mirrors the private-library deployment eval.
  synthetic - canonicalise with Rotation, then apply a per-id deterministic k.
              Balanced 4-class, clean by construction. This mirrors the .9829 F1 number.
"""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, "/src")
import models as M
try:
    from repblock import RepP4Net
except Exception:
    RepP4Net = None

V = Path(os.environ.get("VALDIR", "/data/oidval")); IMG = V/"img"; OUT = V/"probs"; OUT.mkdir(exist_ok=True)
BS = int(os.environ.get("BS", 32))
RES = 224  # per-run, overwritten from hist.json below
dev = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(dev)
STD  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(dev)

rows = [json.loads(l) for l in open(V/"pool_val.jsonl")]
rows = [r for r in rows if (IMG/f"{r['id']}.jpg").exists()]
rows.sort(key=lambda r: r["id"])
print(f"{len(rows)} validation images present", flush=True)

def kfor(i):
    return int(hashlib.md5(("synth"+i).encode()).hexdigest()[:8], 16) % 4

class Nat(Dataset):
    def __len__(self): return len(rows)
    def __getitem__(self, i):
        r = rows[i]
        im = Image.open(IMG/f"{r['id']}.jpg").convert("RGB").resize((RES,RES), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).contiguous()
        return x, (4 - r["rot"]//90) % 4

class Syn(Dataset):
    def __len__(self): return len(rows)
    def __getitem__(self, i):
        r = rows[i]
        im = Image.open(IMG/f"{r['id']}.jpg").convert("RGB")
        if r["rot"]: im = im.rotate(r["rot"], expand=True)
        im = im.resize((RES,RES), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(im).copy()).permute(2,0,1)
        k = kfor(r["id"])
        return torch.rot90(x, k, dims=(-2,-1)).contiguous(), k

def build(sd, arch=None):
    """Dispatch on the arch name recorded in hist.json; fall back to shape inference."""
    if arch == "plain_scratch":
        n = M.PlainNet(widths=(32, 64, 128, 192, 256)); n.load_state_dict(sd); return n, arch
    if arch == "p4_deep6":
        n = M.P4MobileNetV2(widths=(8,12,16,24,32,48), expand=2, mix="all"); n.load_state_dict(sd); return n, arch
    if arch == "p4_res128":
        n = M.P4MobileNetV2(widths=(8,16,24,32,48), expand=2, mix="all"); n.load_state_dict(sd); return n, arch
    if arch == "p4_mobile":
        n = M.P4MobileNet(widths=(8,16,24,32,48), expand=2); n.load_state_dict(sd); return n, arch
    keys = set(sd)
    if RepP4Net is not None and any(".conv3." in k for k in keys):
        w = [sd["lift.weight"].shape[0]]; i = 0
        while f"blocks.{i}.conv3.weight" in sd:
            w.append(sd[f"blocks.{i}.conv3.weight"].shape[0]); i += 1
        n = RepP4Net(widths=tuple(w)); n.load_state_dict(sd); return n, f"rep{tuple(w)}"
    if "lift.weight" in keys:
        w = [sd["lift.weight"].shape[0]]; i = 0
        while f"blocks.{i}.weight" in sd:
            w.append(sd[f"blocks.{i}.weight"].shape[0]); i += 1
        n = M.P4Net(widths=tuple(w)); n.load_state_dict(sd); return n, f"p4{tuple(w)}"
    import timm
    n = timm.create_model("mobilenetv4_conv_small.e2400_r224_in1k", pretrained=False, num_classes=4)
    n.load_state_dict(sd); return n, "mnv4"

@torch.no_grad()
def run(net, ds):
    dl = DataLoader(ds, batch_size=BS, num_workers=8, pin_memory=True)
    P, Y = [], []
    for xb, yb in dl:
        xb = ((xb.to(dev, non_blocking=True).float().div(255)) - MEAN) / STD
        P.append(torch.softmax(net(xb).float(), 1).cpu().numpy()); Y.append(yb.numpy())
    return np.concatenate(P), np.concatenate(Y)

ids = np.array([r["id"] for r in rows])
for name in os.environ["RUNS"].split(","):
    name = name.strip()
    ck = Path(f"/work/runs/{name}/model.pt")
    if not ck.exists(): print(f"SKIP {name}: no checkpoint", flush=True); continue
    hj = ck.parent/"hist.json"
    meta = {}
    if hj.exists():
        try: meta = json.load(open(hj))
        except Exception: meta = {}
    RES = int(meta.get("res") or 224)
    try:
        net, tag = build(torch.load(ck, map_location="cpu"), meta.get("arch"))
    except Exception as e:
        print(f"SKIP {name}: {e}", flush=True); continue
    net = net.to(dev).eval()
    t0 = time.time()
    for variant, ds in (("natural", Nat()), ("synth", Syn())):
        p, y = run(net, ds)
        np.savez(OUT/f"{name}.{variant}.npz", p=p.astype(np.float32), y=y, ids=ids)
    print(f"{name:20s} {tag:22s} res={RES} {time.time()-t0:.0f}s  acc_nat={(np.load(OUT/f'{name}.natural.npz')['p'].argmax(1)==np.load(OUT/f'{name}.natural.npz')['y']).mean():.4f}", flush=True)
    del net
    torch.cuda.empty_cache()
print("VAL_SCORE_DONE", flush=True)
