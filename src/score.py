import numpy as np, sys, os

RESULTS = os.environ.get('RESULTS') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')

def score(name, key='p'):
    d = np.load(os.path.join(RESULTS, f'{name}.npz'), allow_pickle=True)
    p, y = d[key], d['y']
    pred = p.argmax(1); conf = p.max(1)
    rot = y != 0
    nrot = int(rot.sum())
    flag = pred != 0
    missed = int((rot & (pred != y)).sum())

    order = np.argsort(-conf[flag])
    hit = ((pred == y) & rot)[flag][order]
    tp = np.cumsum(hit); k = np.arange(1, len(hit) + 1)
    P = tp / k; R = tp / nrot
    ap = float(np.sum(np.diff(np.concatenate([[0.0], R])) * P))

    def r_at(pmin):
        ok = P >= pmin
        return float(R[ok].max()) if ok.any() else 0.0

    def band(pmin):
        ok = P >= pmin
        return int(tp[ok].max()) if ok.any() else 0

    return dict(n=len(y), nrot=nrot, ap=ap, r93=r_at(.93), r97=r_at(.97),
                onepress=band(.95), missed=missed)

for n in sys.argv[1:]:
    s = score(n)
    print(f"{n:22s} AP={s['ap']:.4f}  R@P>=.93={s['r93']:.3f}  R@P>=.97={s['r97']:.3f}  "
          f"one-press={s['onepress']}/{s['nrot']}  missed={s['missed']}")

def ens(names, key='p'):
    ds = [np.load(f'probs/{n}.npz', allow_pickle=True) for n in names]
    base = ds[0]['ids']
    idx = [np.argsort(d['ids']) for d in ds]
    ord0 = np.argsort(base)
    ps = [d['p'][i] for d, i in zip(ds, idx)]
    y = ds[0]['y'][idx[0]]
    p = np.mean(ps, 0)
    np.savez(os.path.join(RESULTS, '_ens.npz'), p=p, y=y, ids=base[ord0])
    return score('_ens')
