"""cmp_mask.py : verifie le format etendu. Usage : python3 cmp_mask.py reponse_classique.bin reponse_fast.bin
(masque attendu pour l'evenement e : e % 3 + 1, comme ecrit par make_req.py ... fast)"""
import sys
import numpy as np
sys.path.insert(0, ".")
exec(open("tests/cmp_rsp.py").read().split("a, b = read")[0])
def read_ext(p):
    out = []
    with open(p, "rb") as f:
        while True:
            h = np.fromfile(f, dtype=np.int32, count=4)
            if len(h) < 4: break
            mask = int(h[0]) % 3 + 1
            nc = (int(h[1]) if mask & 1 else 0) + (int(h[2]) if mask & 2 else 0)
            out.append((h, mask, np.fromfile(f, dtype=np.float64, count=nc * int(h[3])).reshape(nc, int(h[3]))))
    return out
ref = read(sys.argv[1]); ext = read_ext(sys.argv[2]); worst = 0.0
assert len(ref) == len(ext)
for (hr, A), (he, mask, B) in zip(ref, ext):
    assert (hr == he).all()
    cols = []
    if mask & 1: cols.append(A[:hr[1]])
    if mask & 2: cols.append(A[hr[1]:hr[1] + hr[2]])
    E = np.vstack(cols) if cols else np.zeros((0, hr[3]))
    assert E.shape == B.shape, (E.shape, B.shape)
    worst = max(worst, np.abs(E - B).max() if E.size else 0.0)
print("format etendu : ecart max avec les colonnes correspondantes de la reponse classique = %.3e  -> %s" % (worst, "OK" if worst <= 1e-13 else "ECHEC (seuil 1e-13)"))
