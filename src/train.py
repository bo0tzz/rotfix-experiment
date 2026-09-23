"""Four-arm rotation-model comparison on Open Images, evaluated on a human-verified Immich library.

Arms (ARCH env):
  plain_scratch  - plain CNN, random init, evaluated with and without 4-rotation TTA
  p4_scratch     - C4-equivariant counterpart of the same template, single pass (TTA is a no-op)
  mnv4_pretrained- timm MobileNetV4 (Apache-2.0 weights), ImageNet init, with TTA
  mnv4_consistency - same, plus a rotation-consistency loss so one pass suffices
"""
import json, os, random, time, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, DataLoader
import models as M

ARCH   = os.environ["ARCH"]
TEACHER= os.environ.get("TEACHER","")
KD_A   = float(os.environ.get("KD_ALPHA","0.3"))   # weight on the hard label; 1-a on the teacher
RES    = int(os.environ.get("RES", 224))
EPOCHS = int(os.environ.get("EPOCHS", 12))
BS     = int(os.environ.get("BS", 128))
LR     = float(os.environ.get("LR", 0))   # 0 = pick per arm below
N_TRAIN= int(os.environ.get("N_TRAIN", 0))          # 0 = all
N_VAL  = int(os.environ.get("N_VAL", 5000))
N_LIB  = int(os.environ.get("N_LIB", 3500))
FULL_LIB = os.environ.get("FULL_LIB", "1") == "1"
IMGDIR = os.environ.get("IMGDIR","oid/img")
TESTDIR= os.environ.get("TESTDIR","testset/thumbs")
TESTEXT= os.environ.get("TESTEXT","webp")
MANIFEST=os.environ.get("MANIFEST","oid/manifest.jsonl")
WIDTHSC=float(os.environ.get("WIDTH_SCALE","1.0"))
AUG    = os.environ.get("AUG","0") == "1"
W      = Path("/work"); RUNID  = os.environ.get("RUN_ID","") or ARCH
OUT = W / f"runs/{RUNID}"; OUT.mkdir(parents=True, exist_ok=True)
dev    = "cuda"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(dev)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(dev)
def log(*a): print(*a, flush=True)


def pad_square(im: Image.Image) -> Image.Image:
    """MEASURED: squash-to-square, NOT pad-to-square.

    Padding centres with integer division, so an odd size difference leaves the image 1 px
    off-centre and after a 90-degree rotation that offset sits on the other axis: max pixel diff
    223/255 vs the rotate-first path. Cropping to fix parity is not equivariant either (which edge?).
    Squashing resamples the two axes independently, so rotating merely swaps the two 1-D
    resamplings: exact in float (2.4e-5), and only uint8 rounding in PIL (mean 0.1/255 on real
    photos). Aspect distortion is consistent across rotations, which is what equivariance needs.
    """
    return im


def augment(im):
    """Randomise the low-level image statistics the 384px run was latching onto: JPEG artefacts,
    sharpening, colour and noise. All rotation-agnostic, so the label is untouched."""
    import io as _io
    if random.random() < 0.7:
        b = _io.BytesIO(); im.save(b, "JPEG", quality=random.randint(30, 95)); b.seek(0)
        im = Image.open(b).convert("RGB")
    if random.random() < 0.3:
        im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2)))
    if random.random() < 0.3:
        im = im.filter(ImageFilter.UnsharpMask(radius=2, percent=random.randint(50, 150)))
    if random.random() < 0.5:
        im = ImageEnhance.Brightness(im).enhance(random.uniform(0.7, 1.3))
        im = ImageEnhance.Contrast(im).enhance(random.uniform(0.7, 1.3))
        im = ImageEnhance.Color(im).enhance(random.uniform(0.6, 1.4))
    return im


class OID(Dataset):
    """Open Images, canonicalised upright via the dataset's own Rotation label."""
    def __init__(self, rows, res, train):
        self.rows, self.res, self.train = rows, res, train
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        im = Image.open(W / IMGDIR / f"{r['id']}.jpg").convert("RGB")
        if r["rot"]:
            # VERIFIED VISUALLY: Open Images Rotation is degrees COUNTER-CLOCKWISE to reach upright,
            # i.e. PIL rotate(+rot). The Immich library truth below uses the OPPOSITE convention
            # (clockwise degrees to fix), hence rotate(-fix) there. Do not unify these.
            im = im.rotate(r["rot"], expand=True)
        if self.train and AUG:
            im = augment(im)
        im = pad_square(im).resize((self.res, self.res), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(im).copy()).permute(2,0,1)
        if self.train and random.random() < 0.5:
            x = torch.flip(x, dims=(-1,))                # hflip preserves uprightness
        k = random.randrange(4) if self.train else (i % 4)
        return torch.rot90(x, k, dims=(-2,-1)).contiguous(), k


class Library(Dataset):
    """The Immich library: 13.3k thumbnails with human-verified orientation. Real-world test set."""
    def __init__(self, ids, truth, res):
        self.ids, self.truth, self.res = ids, truth, res
    def __len__(self): return len(self.ids) * 4
    def __getitem__(self, i):
        a, k = self.ids[i // 4], i % 4
        im = Image.open(W / TESTDIR / f"{a}.{TESTEXT}").convert("RGB")
        fix = self.truth.get(a, 0)
        if fix: im = im.rotate(-fix, expand=True)
        im = pad_square(im).resize((self.res, self.res), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(im).copy()).permute(2,0,1)
        return torch.rot90(x, k, dims=(-2,-1)).contiguous(), k


def build():
    if ARCH == "p4_scratch":
        w=tuple(max(2,round(x*WIDTHSC)) for x in (8,16,32,48,64))
        log(f"p4_scratch widths {w}")
        return M.P4Net(widths=w).to(dev), True
    if ARCH == "plain_scratch":
        return M.PlainNet(widths=(32, 64, 128, 192, 256)).to(dev), False
    if ARCH == "p4_big":
        return M.P4Net(widths=(16, 32, 64, 96, 128)).to(dev), True
    if ARCH == "p4_deep6":
        return M.P4MobileNetV2(widths=(8,12,16,24,32,48), expand=2, mix="all").to(dev), True
    if ARCH == "p4_res128":
        return M.P4MobileNetV2(widths=(8,16,24,32,48), expand=2, mix="all").to(dev), True
    if ARCH == "p4_mobile":
        # depthwise-separable equivariant: 0.247 GMACs single pass vs mnv4's 0.185 x4 = 0.74 with TTA
        return M.P4MobileNet(widths=(8, 16, 24, 32, 48), expand=2).to(dev), True
    import timm
    m = timm.create_model("mobilenetv4_conv_small.e2400_r224_in1k",
                          pretrained=(ARCH != "mnv4_scratch"), num_classes=4)
    return m.to(dev), False


def norm(x):  return (x.float().div(255) - MEAN) / STD

@torch.no_grad()
def evaluate(net, loader):
    net.eval(); single = tta = n = 0
    for xb, yb in loader:
        xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
        x = norm(xb)
        single += (net(x).argmax(1) == yb).sum().item()
        agg = torch.zeros(len(xb), 4, device=dev)
        for j in range(4):
            p = F.softmax(net(torch.rot90(x, j, dims=(-2,-1))), 1)
            for c in range(4): agg[:, c] += p[:, (c + j) % 4]
        tta += (agg.argmax(1) == yb).sum().item(); n += len(xb)
    net.train(); return single / n, tta / n


class LibraryStored(Dataset):
    """Library images in the orientation they are actually stored in, with the true fix angle.
    k = number of 90-degree CCW steps from upright = truth/90."""
    def __init__(self, ids, truth, res):
        self.ids, self.truth, self.res = ids, truth, res
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        a = self.ids[i]; fix = self.truth.get(a, 0)
        im = Image.open(W / TESTDIR / f"{a}.{TESTEXT}").convert("RGB")
        im = pad_square(im).resize((self.res, self.res), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)
        return x.contiguous(), fix // 90


@torch.no_grad()
def product_metrics(net, loader):
    """Precision/recall on genuinely-rotated images at a range of confidence thresholds."""
    net.eval(); P, Y = [], []
    for xb, yb in loader:
        xb = xb.to(dev, non_blocking=True)
        p = F.softmax(net(norm(xb)), 1)
        P.append(p.cpu()); Y.append(yb)
    P, Y = torch.cat(P), torch.cat(Y)
    conf, pred = P.max(1)
    out = []
    n_rot = int((Y != 0).sum())
    for thr in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        flag = (pred != 0) & (conf >= thr)
        tp = int((flag & (pred == Y)).sum())
        out.append({"thr": thr, "flagged": int(flag.sum()), "exact": tp,
                    "precision": round(tp / max(1, int(flag.sum())), 4),
                    "recall": round(tp / max(1, n_rot), 4)})
    net.train(); return {"n_images": len(Y), "n_rotated": n_rot, "curve": out}


def main():
    rows = [json.loads(l) for l in open(W / MANIFEST)]
    seen, uniq = set(), []
    for r in rows:
        if r["id"] not in seen: seen.add(r["id"]); uniq.append(r)
    random.Random(0).shuffle(uniq)
    n_val = N_VAL
    val, train = uniq[:n_val], uniq[n_val:]
    if N_TRAIN: train = train[:N_TRAIN]
    truth = json.load(open(W / "testset/ground_truth.json"))["truth"]
    lib_ids = sorted(p.stem for p in (W / TESTDIR).glob(f"*.{TESTEXT}"))
    log(f"{ARCH}: train {len(train)}  oid-val {len(val)}  library {len(lib_ids)} x4  res {RES}")

    nw = min(16, os.cpu_count())
    dl = lambda ds, sh: DataLoader(ds, batch_size=BS, shuffle=sh, num_workers=nw,
                                   pin_memory=True, drop_last=sh, persistent_workers=True)
    tl = dl(OID(train, RES, True), True)
    vl = dl(OID(val, RES, False), False)
    lib_sub = random.Random(7).sample(lib_ids, min(N_LIB, len(lib_ids)))   # per-epoch monitor
    ll = dl(Library(lib_sub, truth, RES), False)
    ll_full = dl(Library(lib_ids, truth, RES), False)                      # full sweep at the end

    net, is_eq = build()
    teacher = None
    if TEACHER:
        import copy
        tarch = TEACHER
        save_arch = ARCH
        globals()["ARCH"] = tarch
        teacher, _ = build()
        globals()["ARCH"] = save_arch
        teacher.load_state_dict(torch.load(W / f"runs/{tarch}/model.pt", map_location=dev))
        teacher.eval()
        for p in teacher.parameters(): p.requires_grad_(False)
        log(f"teacher {tarch} loaded, alpha={KD_A} (hard label) / {1-KD_A} (soft)")
    lr = LR or (3e-4 if ARCH.startswith("mnv4") and ARCH != "mnv4_scratch" else 3e-3)
    npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=EPOCHS*len(tl), pct_start=0.25)
    scaler = torch.amp.GradScaler("cuda")
    log(f"params {npar:,} | equivariant={is_eq} | lr {lr} | steps/epoch {len(tl)} | workers {nw}")

    hist = []
    for ep in range(EPOCHS):
        t0, tot, cnt = time.time(), 0.0, 0
        for xb, yb in tl:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                x = norm(xb)
                out = net(x)
                loss = F.cross_entropy(out, yb)
                if teacher is not None:
                    with torch.no_grad():
                        # TTA-aggregate the teacher: its 4-view consensus is a better target than 1 view
                        agg = torch.zeros(len(x), 4, device=dev)
                        for j in range(4):
                            tp = F.softmax(teacher(torch.rot90(x, j, dims=(-2,-1))), 1)
                            for c in range(4): agg[:, c] += tp[:, (c + j) % 4]
                        soft = (agg / 4).clamp_min(1e-6)
                    loss = KD_A * loss + (1 - KD_A) * F.kl_div(F.log_softmax(out, 1), soft, reduction="batchmean")
                if ARCH == "mnv4_consistency":
                    j = random.randrange(1, 4)
                    o2 = net(torch.rot90(x, j, dims=(-2,-1)))
                    loss = loss + 0.5 * F.kl_div(F.log_softmax(o2, 1),
                                                 F.softmax(torch.roll(out, j, dims=-1), 1).detach(),
                                                 reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()*len(xb); cnt += len(xb)
        vs, vt = evaluate(net, vl)
        ls, lt = evaluate(net, ll)
        hist.append({"epoch": ep, "loss": tot/cnt, "oid_single": vs, "oid_tta": vt,
                     "lib_single": ls, "lib_tta": lt, "sec": round(time.time()-t0)})
        log(f"  ep{ep:>2} loss={tot/cnt:.4f} | oid {vs:.4f}/{vt:.4f} | library {ls:.4f}/{lt:.4f} | {time.time()-t0:.0f}s")
        json.dump({"arch": ARCH, "params": npar, "res": RES, "hist": hist}, open(OUT/"hist.json","w"), indent=1)
        torch.save(net.state_dict(), OUT/"model.pt")   # per-epoch: node reboots kill long runs
    fs, ft = evaluate(net, ll_full) if FULL_LIB else (float("nan"), float("nan"))
    pm = product_metrics(net, dl(LibraryStored(lib_ids, truth, RES), False))
    hist.append({"epoch": "product_metrics", **pm})
    best = max(pm["curve"], key=lambda r: r["precision"] + r["recall"])
    log(f"  PRODUCT ({pm['n_images']} stored, {pm['n_rotated']} rotated): "
        + "  ".join(f"thr{c['thr']}: P={c['precision']:.3f} R={c['recall']:.3f}" for c in pm["curve"] if c["thr"] in (0.5, 0.8, 0.9, 0.95)))
    hist.append({"epoch": "final_full_library", "lib_single": fs, "lib_tta": ft})
    json.dump({"arch": ARCH, "params": npar, "res": RES, "hist": hist}, open(OUT/"hist.json","w"), indent=1)
    log(f"  FULL library ({len(lib_ids)} imgs x4): single={fs:.4f} tta={ft:.4f}")
    torch.save(net.state_dict(), OUT/"model.pt")
    last = [h for h in hist if "oid_single" in h][-1]
    log(f"{ARCH} DONE params={npar:,} | oid {last['oid_single']:.4f}/{last['oid_tta']:.4f}"
        f" | library(sub) {last['lib_single']:.4f}/{last['lib_tta']:.4f}"
        f" | library(full) {fs:.4f}/{ft:.4f}")

if __name__ == "__main__":
    main()
