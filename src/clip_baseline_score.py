"""Parts 2-4 of the CLIP rotation test, reusing the embeddings saved by clipzs.py."""
import json, numpy as np, onnxruntime as ort
from pathlib import Path
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

W = Path("/work")
z = np.load(W / "probs/clip_embed.npz", allow_pickle=True)
E, y = z["E"], z["y"]
print(f"loaded {E.shape} embeddings, {int((y!=0).sum())} rotated of {len(y)}", flush=True)

d = snapshot_download("immich-app/ViT-B-32__openai", cache_dir="/cache")
tcfg = json.load(open(Path(d) / "textual" / "tokenizer_config.json"))
tok = Tokenizer.from_file(str(next(Path(d).rglob("tokenizer.json"))))
pad = tcfg["pad_token"]; pad = pad if isinstance(pad, str) else pad["content"]
tok.enable_padding(length=77, pad_token=pad, pad_id=tok.token_to_id(pad))
tok.enable_truncation(max_length=77)

so = ort.SessionOptions(); so.inter_op_num_threads = 1; so.intra_op_num_threads = 8
txt = ort.InferenceSession(f"{d}/textual/model.onnx", so, providers=["CPUExecutionProvider"])
tin = txt.get_inputs()[0].name

def embed_text(strings):
    out = []
    for s in strings:
        v = txt.run(None, {tin: np.array([tok.encode(s).ids], np.int32)})[0][0]
        out.append(v / np.linalg.norm(v))
    return np.stack(out)

PROMPTS = {
 "plain":    ["a photo", "a photo rotated 90 degrees counterclockwise", "an upside-down photo",
              "a photo rotated 90 degrees clockwise"],
 "sideways": ["an upright photo", "a sideways photo turned to the left", "an upside-down photo",
              "a sideways photo turned to the right"],
 "terse":    ["upright", "rotated left", "upside down", "rotated right"],
}
print("\n--- 2. zero-shot: stored embedding vs directional prompts ---")
for name, ps in PROMPTS.items():
    pred = (E[:, 0] @ embed_text(ps).T).argmax(1)
    print(f"  {name:9s} 4-way acc = {(pred==y).mean():.4f}   upright-kept = {(pred[y==0]==0).mean():.4f}")

print("\n--- 3. rotate-and-score: which of the 4 rotations looks most 'upright'? ---")
for u in ["an upright photograph", "a correctly oriented photo", "a normal photo the right way up"]:
    t = embed_text([u])[0]
    pred = (E @ t).argmax(1)
    print(f"  acc = {(pred==y).mean():.4f}  upright-kept = {(pred[y==0]==0).mean():.4f}   <- \"{u}\"")

print("\n--- 4. trained linear probe on the pooled vector (linear ceiling) ---")
from sklearn.linear_model import LogisticRegression
rs = np.random.RandomState(0); perm = rs.permutation(len(y)); cut = int(.7*len(y))
tr, va = perm[:cut], perm[cut:]
Xtr = np.concatenate([E[tr, k] for k in range(4)])
ytr = np.concatenate([(y[tr]-k) % 4 for k in range(4)])
clf = LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1).fit(Xtr, ytr)
pv = clf.predict(E[va, 0])
print(f"  4-way acc = {(pv==y[va]).mean():.4f}   upright-kept = {(pv[y[va]==0]==0).mean():.4f}  (n_val={len(va)})")
prob = clf.predict_proba(E[va, 0]); conf = prob.max(1); yv = y[va]
print("\n  product metric for the probe (flag when argmax != 0):")
for t in [0.0, 0.5, 0.9, 0.95]:
    f = (pv != 0) & (conf >= t); tp = int((f & (yv != 0) & (pv == yv)).sum()); fl = int(f.sum())
    print(f"    thr {t:<5} flagged {fl:5d}  P={tp/max(fl,1):.3f}  R={tp/max(int((yv!=0).sum()),1):.3f}")
print("CLIPZS2_DONE")
