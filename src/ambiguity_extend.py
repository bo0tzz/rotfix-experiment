"""Label the 401,183 images that manifest_15 adds over manifest_v2 with the SAME ambiguity protocol.

The original set came from 2-fold CV because every image in manifest_v2 is in-sample for any model
trained on it. The new images are genuinely out-of-sample for a fold model, so one pass suffices.

Protocol (from cl_train.py + the 09-30 split): score canonicalised-upright, normalised entropy over
4 classes; ambiguous iff ent >= 0.75; within those, dominant opposite-pair mass (p0+p2 vs p1+p3)
>= 0.60 -> pair-type with offset 0 ({0,180}) or 1 ({90,270}), else uniform.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, "/src")
import models as M

W = Path("/work")
old = set()
for l in open(W/"oid/manifest_v2.jsonl"): old.add(json.loads(l)["id"])
rows = []
for l in open(W/"oid/manifest_15.jsonl"):
    r = json.loads(l)
    if r["id"] not in old: rows.append(r)
print(f"{len(rows):,} images new in manifest_15", flush=True)

RES, BS = 224, int(os.environ.get("BS", 128))
dev = "cuda"
MEAN = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(dev)
STD  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(dev)

class DS(Dataset):
    def __len__(self): return len(rows)
    def __getitem__(self, i):
        r = rows[i]
        im = Image.open(W/"oid/img_v2"/f"{r['id']}.jpg").convert("RGB")
        if r["rot"]: im = im.rotate(r["rot"], expand=True)      # canonicalise: correct class is 0
        im = im.resize((RES,RES), Image.BICUBIC)
        return torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).contiguous(), i

sd = torch.load(W/"runs/fold0of2/model.pt", map_location="cpu")
net = M.P4Net(widths=(16,32,64,96,128)); net.load_state_dict(sd); net = net.to(dev).eval()
print(f"fold0of2 loaded, {sum(p.numel() for p in net.parameters()):,} params", flush=True)

P = np.zeros((len(rows),4), np.float32); t0=time.time()
with torch.no_grad():
    for xb, idx in DataLoader(DS(), batch_size=BS, num_workers=10, pin_memory=True):
        xb = ((xb.to(dev, non_blocking=True).float().div(255)) - MEAN) / STD
        P[idx.numpy()] = torch.softmax(net(xb).float(), 1).cpu().numpy()
print(f"scored in {time.time()-t0:.0f}s", flush=True)

p = P.astype(np.float64); p = p/p.sum(1, keepdims=True)
ent = -(p*np.log(np.clip(p,1e-12,1))).sum(1)/np.log(4)
AMB = ent >= 0.75
pair0, pair1 = p[:,0]+p[:,2], p[:,1]+p[:,3]
dom = np.maximum(pair0, pair1); off = (pair1 > pair0).astype(int)
PAIR = AMB & (dom >= 0.60); UNI = AMB & ~PAIR
print(f"ambiguous {int(AMB.sum()):,} ({100*AMB.mean():.2f}%)  ->  pair {int(PAIR.sum()):,} "
      f"(off0 {int((PAIR&(off==0)).sum()):,}, off1 {int((PAIR&(off==1)).sum()):,})  uniform {int(UNI.sum()):,}", flush=True)

out = Path("/work/ambext"); out.mkdir(exist_ok=True)
with open(out/"new_uniform_ids.txt","w") as f:
    for i in np.where(UNI)[0]: f.write(rows[i]["id"]+"\n")
with open(out/"new_pair_ids.txt","w") as f:
    for i in np.where(PAIR)[0]: f.write(f"{rows[i]['id']} {off[i]}\n")
np.savez(out/"new_scores.npz", ids=np.array([r["id"] for r in rows]), p=P, ent=ent)
print("AMBEXT_DONE", flush=True)
