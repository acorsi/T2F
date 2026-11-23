"""cmp_rsp.py : compare deux fichiers reponse (binaires) de profile_basis. Usage : python3 cmp_rsp.py a.bin b.bin"""
import sys
import numpy as np

def read(p):
    out = []
    with open(p, "rb") as f:
        while True:
            h = np.fromfile(f, dtype=np.int32, count=4)
            if len(h) < 4:
                break
            n = (h[1] + h[2]) * h[3]
            out.append((h, np.fromfile(f, dtype=np.float64, count=int(n)).reshape(int(h[1] + h[2]), int(h[3]))))
    return out

a, b = read(sys.argv[1]), read(sys.argv[2])
assert len(a) == len(b), "nombre d'evenements different"
worst = 0.0; ref = 0.0; nbad = 0
for (ha, A), (hb, B) in zip(a, b):
    assert (ha == hb).all(), "entetes differents"
    d = np.abs(A - B).max() if A.size else 0.0
    m = np.abs(A).max() if A.size else 0.0
    print("evt %d : M=(%d,%d) lignes=%d  max|A|=%.4e  ecart max=%.3e  relatif=%.2e" % (ha[0], ha[1], ha[2], ha[3], m, d, d / max(m, 1e-300)))
    worst = max(worst, d); ref = max(ref, m)
print("ECART MAX GLOBAL : %.3e  (relatif au max de la reponse : %.2e)  -> %s" % (worst, worst / max(ref, 1e-300), "OK" if worst <= 1e-13 * max(ref, 1e-300) else "ECHEC (seuil 1e-13 relatif)"))
