#!/usr/bin/env python3
"""Compare deux sorties .h5 de ./process (avant / apres modification).
usage : compare_outputs.py ref.h5 new.h5 [tol_rel]
 - datasets simples (KE, phi...) : entiers identiques, flottants a tol_rel pres
 - datasets a longueur variable (pid, pk0, pnt, pq, pqt...) : compares evenement par evenement
   * entiers (pid, pk0, pnt) : identiques
   * flottants (pq, pqt) : ecart max / max|ref| <= tol_rel (defaut 1e-5)
Tout dataset non compare est signale. Code retour 0 si conforme, 1 sinon."""
import sys
import numpy as np
import h5py


def collect(f):
    out = {}
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out[name] = obj
    f.visititems(visit)
    return out


def compare_vlen(name, x, y, tol):
    """x, y : tableaux d'objets (un sous-tableau par evenement)."""
    nev = len(x)
    n_shape = 0       # evenements dont la longueur differe
    n_diff = 0        # evenements dont le contenu differe (entiers) ou hors tolerance (flottants)
    err = 0.0
    scale = 0.0
    is_float = False
    for i in range(nev):
        a = np.asarray(x[i]); b = np.asarray(y[i])
        if a.shape != b.shape:
            n_shape += 1; continue
        if a.size == 0:
            continue
        if a.dtype.kind == "f":
            is_float = True
            scale = max(scale, float(np.max(np.abs(a))))
        elif not np.array_equal(a, b):
            n_diff += 1
    if is_float:
        for i in range(nev):
            a = np.asarray(x[i]); b = np.asarray(y[i])
            if a.shape == b.shape and a.size:
                err = max(err, float(np.max(np.abs(a - b))))
        rel = err / scale if scale > 0 else 0.0
        ok = (n_shape == 0 and rel <= tol)
        print("%s %-12s flottants : ecart max relatif a l'echelle = %.3g ; %d evenement(s) de longueur differente"
              % ("OK  " if ok else "DIFF", name, rel, n_shape))
    else:
        ok = (n_shape == 0 and n_diff == 0)
        print("%s %-12s entiers : %d evenement(s) differents, %d de longueur differente (sur %d)"
              % ("OK  " if ok else "DIFF", name, n_diff, n_shape, nev))
    return 0 if ok else 1


def main():
    if len(sys.argv) < 3:
        print(__doc__); return 2
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
    bad = 0
    with h5py.File(sys.argv[1], "r") as a, h5py.File(sys.argv[2], "r") as b:
        da, db = collect(a), collect(b)
        for k in sorted(set(da) ^ set(db)):
            print("DIFF  dataset present d'un seul cote :", k); bad += 1
        for k in sorted(set(da) & set(db)):
            x, y = da[k][()], db[k][()]
            if x.shape != y.shape:
                print("DIFF  forme differente %-30s %s vs %s" % (k, x.shape, y.shape)); bad += 1; continue
            if x.dtype.kind == "O":
                bad += compare_vlen(k, x, y, tol)
            elif x.dtype.kind in "iub":
                n = int(np.count_nonzero(x != y))
                print("%s %-12s entiers : %d differences" % ("OK  " if n == 0 else "DIFF", k, n))
                bad += (n != 0)
            elif x.dtype.kind == "f":
                scale = max(float(np.max(np.abs(x))) if x.size else 0.0, 1e-300)
                e = float(np.max(np.abs(x - y))) / scale if x.size else 0.0
                print("%s %-12s ecart max relatif a l'echelle = %.3g" % ("OK  " if e <= tol else "DIFF", k, e))
                bad += (e > tol)
            else:
                print("NON COMPARE  %-12s (type %s)" % (k, x.dtype)); bad += 1
        if "metadata" in a and "metadata" in b:
            for key in sorted(set(a["metadata"].attrs) & set(b["metadata"].attrs)):
                if key in ("created", "config_file", "config_text", "output_file"): continue
                va, vb = a["metadata"].attrs[key], b["metadata"].attrs[key]
                if not np.array_equal(va, vb):
                    print("INFO  metadata %-16s : %s -> %s" % (key, va, vb))
    print("RESULTAT :", "conforme" if bad == 0 else "%d difference(s)" % bad)
    return 0 if bad == 0 else 1


sys.exit(main())
