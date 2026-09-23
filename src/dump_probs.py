"""Dump per-image softmax over the library in stored orientation, for offline rescoring.
Same protocol as train.py's LibraryStored: squash-to-square -> res BICUBIC, ImageNet norm.
Equivariant nets are single-pass (TTA is a no-op), so one forward per image.
"""
import json, os, sys, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, "/src")
import models as M

W = Path("/work"); TESTDIR = W / "testset/prev_raw"
dev = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(dev)
STD  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(dev)
truth = json.load(open(W / "testset/ground_truth.json"))["truth"]
ids = sorted(p.stem for p in TESTDIR.glob("*.jpg"))

RUNS = [("v2-224full12", "p4_scratch", 224, "runs/v2-224full12/model.pt"),
        ("v2-bigfull-s2", "p4_big", 224, "runs/v2-bigfull-s2/model.pt"),
        ("v2-big12", "p4_big", 224, "runs/v2-big12/model.pt"),
        ("v2-224full", "p4_scratch", 224, "runs/v2-224full/model.pt"),
        ("v2-bigfull", "p4_big", 224, "runs/v2-bigfull/model.pt"),
        ("big-full", "p4_big", 224, "runs/p4_big/model.pt"),
        ("v2-big",   "p4_big", 224, "runs/v2-big/model.pt"),
        ("v2-224",   "p4_scratch", 224, "runs/v2-224/model.pt"),
        ("v2-384",   "p4_scratch", 384, "runs/v2-384/model.pt")]

class Stored(Dataset):
    def __init__(self, res): self.res = res
    def __len__(self): return len(ids)
    def __getitem__(self, i):
        im = Image.open(TESTDIR / f"{ids[i]}.jpg").convert("RGB").resize((self.res,)*2, Image.BICUBIC)
        return torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).contiguous(), truth.get(ids[i],0)//90

(W / "probs").mkdir(exist_ok=True)
for name, arch, res, ckpt in RUNS:
    net = (M.P4Net(widths=(16,32,64,96,128)) if arch=="p4_big" else M.P4Net(widths=(8,16,32,48,64))).to(dev)
    net.load_state_dict(torch.load(W / ckpt, map_location=dev)); net.eval()
    P, Y = [], []
    with torch.inference_mode():
        for xb, yb in DataLoader(Stored(res), batch_size=64, num_workers=8):
            x = (xb.to(dev).float().div(255) - MEAN) / STD
            P.append(F.softmax(net(x).float(), 1).cpu()); Y.append(yb)
    P = torch.cat(P).numpy(); Y = torch.cat(Y).numpy()
    np.savez(W / f"probs/{name}.npz", p=P, y=Y, ids=np.array(ids))
    print(f"{name}: {P.shape} saved, argmax acc on stored={float((P.argmax(1)==Y).mean()):.4f}", flush=True)
print("DUMP_DONE")
