import h5py, numpy as np

def vlen(f, path, n):
    d = f[path][:n]
    return [np.asarray(a) for a in d]

def flat(lst):
    return np.concatenate(lst) if len(lst) else np.array([])

old = h5py.File('r_1t.h5', 'r')
new = h5py.File('n_8t.h5', 'r')
n1k = h5py.File('n_1t_5k.h5', 'r')

print("=== nouveau (8 threads) contre ancien, 25000 événements ===")
for frag in ['H', 'L']:
    for name in ['theta', 'phi', 'vx', 'vy', 'vz']:
        a = old['/%s/%s' % (frag, name)][:].astype(np.float64)
        b = new['/%s/%s' % (frag, name)][:].astype(np.float64)
        print(frag, name, "max |diff| =", np.max(np.abs(a - b)))
    for name in ['x', 'y', 'z', 'dE', 'px', 'py', 'pq', 'pt', 'pqt', 'pid', 'pnt']:
        p = '/%s/%s' % (frag, name)
        a = vlen(old, p, 25000)
        b = vlen(new, p, 25000)
        same_len = all(len(u) == len(v) for u, v in zip(a, b))
        fa = flat(a).astype(np.float64)
        fb = flat(b).astype(np.float64)
        if not same_len:
            print(frag, name, "LONGUEURS DIFFERENTES")
            continue
        err = np.abs(fa - fb)
        rel = err / np.maximum(np.abs(fa), 1e-30)
        print(frag, name, "max |diff| = %.3g   max rel = %.3g" % (err.max(), rel.max()))

print("\n=== somme de pq par fragment, ancien (double) contre nouveau ===")
for frag in ['H', 'L']:
    a = np.array([u.sum() for u in vlen(old, '/%s/pq' % frag, 25000)])
    b = np.array([v.astype(np.float64).sum() for v in vlen(new, '/%s/pq' % frag, 25000)])
    print(frag, "écart relatif max = %.3g" % np.max(np.abs(a - b) / a))

print("\n=== reproductibilité : 1 thread (5000) contre 8 threads (25000, 5000 premiers) ===")
ok = True
for frag in ['H', 'L']:
    for name in ['theta', 'phi', 'vx', 'vy', 'vz']:
        a = n1k['/%s/%s' % (frag, name)][:5000]
        b = new['/%s/%s' % (frag, name)][:5000]
        ok = ok and np.array_equal(a, b)
    for name in ['x', 'y', 'z', 'dE', 'px', 'py', 'pq', 'pt', 'pqt', 'pid', 'pnt']:
        p = '/%s/%s' % (frag, name)
        a = vlen(n1k, p, 5000)
        b = vlen(new, p, 5000)
        eq = all(np.array_equal(u, v) for u, v in zip(a, b))
        ok = ok and eq
        if not eq:
            print("DIFFERENT :", p)
print("identiques" if ok else "DIFFERENCES")
