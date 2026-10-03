"""Public-only evaluation vs the private-library evaluation: do they pick the same model?"""
import glob, json, os, sys
import numpy as np

PUB = os.path.expanduser("~/rotfix-experiment/oidval/probs")
PRIV = os.path.expanduser("~/rotfix-experiment/handoff/results")

def metrics(p, y):
    pred, conf = p.argmax(1), p.max(1)
    rot = y != 0; nrot = int(rot.sum())
    flag = pred != 0
    if nrot == 0 or flag.sum() == 0:
        return dict(n=len(y), nrot=nrot, ap=0., r93=0., r97=0., onepress=0, missed=nrot, acc=float((pred==y).mean()))
    order = np.argsort(-conf[flag])
    hit = ((pred == y) & rot)[flag][order]
    tp = np.cumsum(hit); k = np.arange(1, len(hit)+1)
    P, R = tp/k, tp/nrot
    ap = float(np.sum(np.diff(np.concatenate([[0.], R])) * P))
    def r_at(m):
        ok = P >= m; return float(R[ok].max()) if ok.any() else 0.
    def band(m):
        ok = P >= m; return int(tp[ok].max()) if ok.any() else 0
    return dict(n=len(y), nrot=nrot, ap=ap, r93=r_at(.93), r97=r_at(.97),
                onepress=band(.95), missed=int((rot & (pred != y)).sum()),
                acc=float((pred == y).mean()))

def load(path):
    d = np.load(path, allow_pickle=True); return metrics(d["p"], d["y"])

def rankcorr(a, b):
    n = len(a)
    def rk(v):
        s = sorted(range(n), key=lambda i: -v[i]); r = [0]*n
        for p_, i in enumerate(s): r[i] = p_+1
        return r
    ra, rb = rk(a), rk(b)
    d2 = sum((x-y)**2 for x, y in zip(ra, rb))
    c = di = 0
    for i in range(n):
        for j in range(i+1, n):
            s = (a[i]-a[j])*(b[i]-b[j]); c += s > 0; di += s < 0
    return 1-6*d2/(n*(n*n-1)), (c-di)/(c+di), ra, rb

runs = sorted({os.path.basename(f).split(".")[0] for f in glob.glob(f"{PUB}/*.natural.npz")
                if os.path.exists(f.replace(".natural.", ".synth."))})
print(f"{len(runs)} runs scored on the held-out public set\n")
rows = []
for r in runs:
    nat = load(f"{PUB}/{r}.natural.npz")
    syn = load(f"{PUB}/{r}.synth.npz")
    pv = load(f"{PRIV}/{r}.npz") if os.path.exists(f"{PRIV}/{r}.npz") else None
    rows.append((r, nat, syn, pv))

print(f"{'run':18s} | {'PUBLIC natural':^30s} | {'PUB synth':^16s} | {'PRIVATE library':^22s}")
print(f"{'':18s} | {'AP':>6s} {'R@.93':>6s} {'1press':>7s} {'miss':>6s} | {'acc':>6s} {'F1ish':>8s} | {'AP':>6s} {'R@.93':>6s} {'miss':>6s}")
for r, nat, syn, pv in sorted(rows, key=lambda t: -t[1]["ap"]):
    pvs = f"{pv['ap']:>6.4f} {pv['r93']:>6.3f} {pv['missed']:>6d}" if pv else f"{'-':>6s} {'-':>6s} {'-':>6s}"
    print(f"{r:18s} | {nat['ap']:>6.4f} {nat['r93']:>6.3f} {nat['onepress']:>3d}/{nat['nrot']:<3d} {nat['missed']:>6d} | "
          f"{syn['acc']:>6.4f} {syn['ap']:>8.4f} | {pvs}")

both = [(r, nat["ap"], pv["ap"]) for r, nat, syn, pv in rows if pv]
if len(both) >= 3:
    rho, tau, ra, rb = rankcorr([b[1] for b in both], [b[2] for b in both])
    print(f"\nPUBLIC-natural AP vs PRIVATE AP over {len(both)} runs: Spearman {rho:.3f}, Kendall {tau:.3f}")
    print(f"{'run':18s} {'pub#':>5s} {'priv#':>6s} {'shift':>6s}   {'pubAP':>7s} {'privAP':>7s}")
    for (nm, a, b_), x, y in sorted(zip(both, ra, rb), key=lambda t: t[1]):
        print(f"{nm:18s} {x:>5} {y:>6} {x-y:>+6}   {a:>7.4f} {b_:>7.4f}{'  <<<' if abs(x-y)>=3 else ''}")

sa = [(r, syn["acc"], nat["ap"]) for r, nat, syn, pv in rows]
rho, tau, _, _ = rankcorr([s[1] for s in sa], [s[2] for s in sa])
print(f"\nSYNTHETIC acc vs NATURAL AP, both public, over {len(sa)} runs: Spearman {rho:.3f}, Kendall {tau:.3f}")
