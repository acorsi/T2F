#!/usr/bin/env python3
"""
diag_band.py : diagnostic en LECTURE SEULE (ne modifie rien) de la largeur de bande de l'inverse du facteur de
Cholesky de la covariance du bruit, utilisee pour le blanchiment (Whitener.apply de fit_profile.py).

Si L^-1 (L = Cholesky de la covariance d'un pad) est quasi bande (largeur b << n), le blanchiment d'un pad coute
O(n b nb) au lieu de O(n^2 nb) : gain n/b. Ce script donne, pour plusieurs longueurs de fenetre n et plusieurs
echelles de bruit serie (ser_scale), la plus petite largeur b telle que tous les elements de L^-1 hors bande
soient inferieurs a tol * max|L^-1|.

Utilise exactement les appels de fit_profile.py / refine_geom.py : FitModel(at, check=...), model.cov(n, ser_scale),
model.ser_scale_pad(enc), load_pad_capacitances, event_pads.

Usage : python3 diag_band.py sortie.h5 [--n_events 20] [--n_extra 166]
A placer dans le dossier de refine_geom.py (imports t2f_io, view_fit).
"""
import argparse

import h5py
import numpy as np

from t2f_io import pad_geometry, event_pads, load_pad_capacitances
from view_fit import FitModel


def band_needed(M, tol):
    """plus petite largeur de bande b (diagonale comprise) telle que |M[j, k]| <= tol * max|M| pour j - k >= b"""
    n = M.shape[0]
    mx = float(np.abs(M).max())
    for d in range(n - 1, 0, -1):
        if float(np.abs(np.diagonal(M, -d)).max()) > tol * mx:
            return d + 1
    return 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("h5file")
    p.add_argument("--n_events", type=int, default=20, help="evenements lus pour la distribution des longueurs de fenetre")
    p.add_argument("--n_extra", type=str, default="166", help="longueurs de fenetre supplementaires (ex. fenetre des pads nuls), separees par des virgules")
    p.add_argument("--tols", type=str, default="1e-3,1e-4,1e-6")
    a = p.parse_args()
    tols = [float(s) for s in a.tols.split(",")]

    f = h5py.File(a.h5file, "r")
    at = f["/metadata"].attrs
    model = FitModel(at, check=False)
    geom = pad_geometry(f)
    caps = load_pad_capacitances(f)

    # echelles de bruit serie
    if caps is None:
        scales = [1.0]
        print("pas de capacites par pad dans le fichier : ser_scale = 1 pour tous les pads")
    else:
        sc = np.array(sorted(set(round(float(model.ser_scale_pad(v[1])), 5) for v in caps.values())))
        scales = sorted(set([float(sc.min()), float(np.median(sc)), float(sc.max())]))
        print("ser_scale sur les pads : min %.4f  mediane %.4f  max %.4f  (%d valeurs distinctes)" % (sc.min(), np.median(sc), sc.max(), len(sc)))

    # longueurs de fenetre
    nts = []
    N = f["/event_id"].shape[0]
    for i in range(min(N, a.n_events)):
        ev = event_pads(f, i, geom)
        nts += [int(v) for v in ev["pnt"] if v > 0]
    nts = np.array(nts)
    ns = sorted(set([int(np.percentile(nts, q)) for q in (10, 50, 90)] + [int(s) for s in a.n_extra.split(",") if s.strip()]))
    print("longueurs de fenetre (echantillons ecrits par pad, %d pads) : min %d  p10 %d  mediane %d  p90 %d  max %d"
          % (len(nts), nts.min(), np.percentile(nts, 10), np.median(nts), np.percentile(nts, 90), nts.max()))
    print()
    print("largeur de bande b de L^-1 (cout du blanchiment ~ n*b au lieu de n*n : gain n/b)")
    head = "   n   ser_scale  " + "  ".join("tol=%-7g" % t for t in tols)
    print(head)
    for n in ns:
        for s in scales:
            C = model.cov(n, s)
            L = np.linalg.cholesky(C)
            Li = np.linalg.inv(L)
            bs = [band_needed(Li, t) for t in tols]
            print("%4d   %8.4f   " % (n, s) + "  ".join("b=%3d (x%3.1f)" % (b, n / float(b)) for b in bs))


if __name__ == "__main__":
    main()
