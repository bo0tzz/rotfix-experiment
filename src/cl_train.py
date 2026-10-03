"""Split the out-of-sample fold posteriors into label errors vs orientation-ambiguity.

Every training image is scored canonicalised-upright, so the correct class is always 0. Because the
net is C4-equivariant the posterior at any rotation is a cyclic shift of this one, so a single pass
characterises the image completely. The two failure modes separate cleanly:

  confident and argmax != 0  ->  Google's Rotation column is probably wrong   (label error)
  high entropy               ->  the image has no recoverable canonical up    (ambiguity)
"""
import numpy as np, json, sys
from pathlib import Path

D = Path(sys.argv[1] if len(sys.argv) > 1 else "probs")
import glob
fs = sorted(glob.glob(str(D / "foldoos_*of2.npz")))
print(f"using {len(fs)} fold file(s): {[f.split('/')[-1] for f in fs]}")
zs = [np.load(f, allow_pickle=True) for f in fs]
p = np.concatenate([z["p"] for z in zs]).astype(np.float64)
ids = np.concatenate([z["ids"] for z in zs])
rot = np.concatenate([z["rot"] for z in zs])
p = p / p.sum(1, keepdims=True)
n = len(p)
pred, conf = p.argmax(1), p.max(1)
ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum(1) / np.log(4)     # 0 = certain, 1 = uniform

print(f"{n:,} out-of-sample training posteriors  (labels all class 0 = upright)")
print(f"  upright-accuracy               {float((pred==0).mean()):.4f}")
print(f"  mean normalised entropy        {ent.mean():.4f}")
print()
print("joint split of the error mass:")
for lo, hi in [(0,.25),(.25,.5),(.5,.75),(.75,1.01)]:
    m = (ent>=lo)&(ent<hi)
    if not m.any(): continue
    wrong = (pred[m]!=0).mean()
    print(f"  entropy {lo:.2f}-{hi:.2f}: {int(m.sum()):>9,} images ({100*m.mean():5.2f}%)  argmax!=0 in {100*wrong:5.1f}%")
print()
LAB = (pred!=0)&(conf>=0.90)
AMB = ent>=0.75
print(f"LABEL-ERROR candidates (argmax!=0, conf>=0.90): {int(LAB.sum()):>8,}  ({100*LAB.mean():.2f}%)")
print(f"  of which Google says rot!=0 already:          {int((LAB&(rot!=0)).sum()):>8,}")
print(f"AMBIGUOUS candidates (entropy>=0.75):           {int(AMB.sum()):>8,}  ({100*AMB.mean():.2f}%)")
print(f"  overlap between the two sets:                 {int((LAB&AMB).sum()):>8,}")
print()
print(f"Google-labelled rotated rate in this pool: {100*float((rot!=0).mean()):.2f}%")
print(f"implied true rate if every label-error candidate is real: "
      f"{100*float(((rot!=0)|LAB).mean()):.2f}%")

np.savez("cl_train_scores.npz", ids=ids, p=p.astype(np.float32), rot=rot, ent=ent,
         label_err=LAB, ambiguous=AMB)
for nm, m in [("labelerr", LAB), ("ambiguous", AMB)]:
    idx = np.argsort(-conf*m if nm=="labelerr" else -ent*m)[:200]
    json.dump({"ids": [str(ids[i]) for i in idx], "pred": [int(pred[i]) for i in idx],
               "conf": [float(conf[i]) for i in idx], "ent": [float(ent[i]) for i in idx],
               "rot": [int(rot[i]) for i in idx]}, open(f"cl_train_{nm}.json","w"))
print("\nwrote cl_train_scores.npz + cl_train_labelerr.json + cl_train_ambiguous.json")
