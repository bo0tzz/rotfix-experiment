"""Can Immich's existing CLIP be used for rotation detection, with no new model?

Three ways, all using immich-app/ViT-B-32__openai exactly as immich-ml loads it:
  1. how rotation-invariant the image embedding actually is (cosine between rotations)
  2. zero-shot: stored embedding vs directional text prompts
  3. rotate-and-score: embed all 4 rotations, pick the one most like "an upright photo"
  4. trained linear probe on the pooled vector = ceiling for any linear readout
"""
import json, os, time, numpy as np, onnxruntime as ort
from pathlib import Path
from PIL import Image
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from concurrent.futures import ThreadPoolExecutor

W = Path("/work"); TESTDIR = W / "testset/prev_raw"
truth = json.load(open(W / "testset/ground_truth.json"))["truth"]
ids = sorted(p.stem for p in TESTDIR.glob("*.jpg"))
LIMIT = int(os.environ.get("LIMIT", "0"))
if LIMIT: ids = ids[:LIMIT]
y = np.array([truth.get(i, 0) // 90 for i in ids])

d = snapshot_download("immich-app/ViT-B-32__openai", cache_dir="/cache")
pp = json.load(open(Path(d) / "visual" / "preprocess_cfg.json"))
SIZE = pp["size"][0] if isinstance(pp["size"], list) else pp["size"]
MEAN = np.array(pp["mean"], np.float32); STD = np.array(pp["std"], np.float32)
print(f"preprocess: size={SIZE} mean={MEAN} std={STD}", flush=True)

def sess(p, threads):
    so = ort.SessionOptions(); so.inter_op_num_threads = 1; so.intra_op_num_threads = threads
    return ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])

T = int(os.environ.get("THREADS", "16"))
vis = sess(f"{d}/visual/model.onnx", T)
txt = sess(f"{d}/textual/model.onnx", T)
_tk = list(Path(d).rglob("tokenizer.json"))
print("repo files:", sorted(p.relative_to(d).as_posix() for p in Path(d).rglob("*") if p.is_file())[:30], flush=True)
tok = Tokenizer.from_file(str(_tk[0]))

def embed_text(strings):
    out = []
    for s in strings:
        e = tok.encode(s)
        n = len(txt.get_inputs()[0].shape)
        arr = np.array([e.ids], dtype=np.int32)
        v = txt.run(None, {txt.get_inputs()[0].name: arr})[0][0]
        out.append(v / np.linalg.norm(v))
    return np.stack(out)

# PIL transpose that applies k*90 CLOCKWISE (k is "degrees/90 clockwise needed to fix")
CW = {0: None, 1: Image.ROTATE_270, 2: Image.ROTATE_180, 3: Image.ROTATE_90}

def prep(im):
    w, h = im.size; s = SIZE / min(w, h)
    im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
    w, h = im.size; l, t = (w - SIZE) // 2, (h - SIZE) // 2
    a = np.asarray(im.crop((l, t, l + SIZE, t + SIZE)), np.float32) / 255.0
    return ((a - MEAN) / STD).transpose(2, 0, 1)[None]

vin = vis.get_inputs()[0].name
def embed_image_all_rot(i):
    im = Image.open(TESTDIR / f"{ids[i]}.jpg").convert("RGB")
    out = np.empty((4, 512), np.float32)
    for k in range(4):
        r = im if CW[k] is None else im.transpose(CW[k])
        v = vis.run(None, {vin: prep(r)})[0][0]
        out[k] = v / np.linalg.norm(v)
    return out

t0 = time.time()
E = np.empty((len(ids), 4, 512), np.float32)
with ThreadPoolExecutor(max_workers=int(os.environ.get("POOL", "8"))) as ex:
    for n, (i, e) in enumerate(zip(range(len(ids)), ex.map(embed_image_all_rot, range(len(ids))))):
        E[i] = e
        if n % 2000 == 0: print(f"  {n}/{len(ids)} {time.time()-t0:.0f}s", flush=True)
print(f"embedded {len(ids)} x4 in {time.time()-t0:.0f}s", flush=True)
np.savez(W / "probs/clip_embed.npz", E=E, y=y, ids=np.array(ids))

print("\n--- 1. how rotation-invariant is the CLIP image embedding? ---")
for k in (1, 2, 3):
    c = (E[:, 0] * E[:, k]).sum(1)
    print(f"  cos(stored, stored rotated {k*90:3d}CW) = {c.mean():.4f} +/- {c.std():.4f}")
rnd = (E[:, 0] * E[np.random.permutation(len(ids)), 0]).sum(1)
print(f"  cos(stored, random other image)      = {rnd.mean():.4f} +/- {rnd.std():.4f}")

PROMPTS = {
 "plain":    ["a photo", "a photo rotated 90 degrees counterclockwise", "an upside-down photo",
              "a photo rotated 90 degrees clockwise"],
 "sideways": ["an upright photo", "a sideways photo turned to the left", "an upside-down photo",
              "a sideways photo turned to the right"],
 "terse":    ["upright", "rotated left", "upside down", "rotated right"],
}
print("\n--- 2. zero-shot: stored embedding vs directional prompts ---")
for name, ps in PROMPTS.items():
    Tt = embed_text(ps)
    pred = (E[:, 0] @ Tt.T).argmax(1)
    print(f"  {name:9s} 4-way acc = {(pred==y).mean():.4f}   upright-kept = {(pred[y==0]==0).mean():.4f}")

UP = ["an upright photograph", "a correctly oriented photo", "a normal photo the right way up"]
print("\n--- 3. rotate-and-score: which rotation looks most 'upright'? ---")
for u in UP:
    t = embed_text([u])[0]
    pred = (E @ t).argmax(1)
    print(f"  acc = {(pred==y).mean():.4f}  upright-kept = {(pred[y==0]==0).mean():.4f}   <- \"{u}\"")

print("\n--- 4. trained linear probe on the pooled vector (the linear ceiling) ---")
from sklearn.linear_model import LogisticRegression
rs = np.random.RandomState(0); perm = rs.permutation(len(ids)); cut = int(.7*len(ids))
tr, va = perm[:cut], perm[cut:]
Xtr = np.concatenate([E[tr, k] for k in range(4)]); ytr = np.concatenate([(y[tr]-k) % 4 for k in range(4)])
clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1).fit(Xtr, ytr)
pv = clf.predict(E[va, 0])
print(f"  4-way acc = {(pv==y[va]).mean():.4f}   upright-kept = {(pv[y[va]==0]==0).mean():.4f}  (n_val={len(va)})")
print("CLIPZS_DONE")
