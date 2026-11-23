#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic en lecture seule : fantomes et couronnes de pads non lus.

Pour chaque evenement :
  1. pads ecrits = P/pid
  2. fantomes retires = pads ecrits sans AUCUN autre pad ecrit a moins de 1.5*a
     (une seule passe, aucun drapeau de simulation utilise)
  3. pads touches = pads ecrits non retires
  4. couronne 1 = pads non touches dont le centre est a moins de 1.5*a
     du centre d'un pad touche
  5. couronne 2 = pads non touches, hors couronne 1, dont le centre est a
     moins de 1.5*a du centre d'un pad de couronne 1
Le drapeau P/pflag (0 = fantome) n'est utilise QUE pour comparer, dans la
sortie, le nombre de fantomes vrais retires / restants. Il n'intervient pas
dans la construction des couronnes.

Le pitch a est calcule a partir des positions : mediane de la distance au
plus proche voisin sur l'ensemble des pads de pads/x, pads/y.

Usage :
  python3 diag_couronnes.py ../drift_output/drift_pad5mm_new.h5 --n 1000 --out diag_cour
"""
import argparse
import sys
import numpy as np
import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--fac", type=float, default=1.5,
                    help="rayon de voisinage en unites de pitch (1.5)")
    ap.add_argument("--out", default="diag_cour",
                    help="prefixe du csv de sortie")
    args = ap.parse_args()

    f = h5py.File(args.h5, "r")
    pid_all = np.asarray(f["pads/id"][:])
    x = np.asarray(f["pads/x"][:], dtype=float)
    y = np.asarray(f["pads/y"][:], dtype=float)
    npad = len(pid_all)
    idx = {}
    for k in range(npad):
        idx[int(pid_all[k])] = k

    # matrice des distances entre tous les pads
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    D = np.hypot(dx, dy)
    D2 = D.copy()
    np.fill_diagonal(D2, np.inf)
    a = float(np.median(D2.min(axis=1)))
    R = args.fac * a
    near = (D2 < R)  # diagonale exclue
    print("pads : %d   pitch a = %.4f cm   rayon %.2f*a = %.4f cm"
          % (npad, a, args.fac, R))

    n_ev = len(f["P/pid"])
    i0 = args.start
    i1 = min(n_ev, args.start + args.n)
    rows = []
    for ev in range(i0, i1):
        pid = np.asarray(f["P/pid"][ev]).astype(int).ravel()
        pfl = np.asarray(f["P/pflag"][ev]).ravel()
        if len(pid) != len(pfl):
            print("evenement %d : len(pid)=%d != len(pflag)=%d, arret"
                  % (ev, len(pid), len(pfl)))
            sys.exit(1)
        w = np.zeros(len(pid), dtype=int)
        for k in range(len(pid)):
            key = int(pid[k])
            if key not in idx:
                print("evenement %d : pad id %d absent de pads/id" % (ev, key))
                sys.exit(1)
            w[k] = idx[key]
        n_w = len(w)
        if n_w == 0:
            rows.append([ev, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            continue

        nb = near[np.ix_(w, w)].sum(axis=1)
        keep = (nb > 0)
        touched = w[keep]
        gh = (pfl == 0)
        n_gh_true = int(gh.sum())
        n_gh_removed = int((gh & (~keep)).sum())
        n_gh_kept = int((gh & keep).sum())
        n_removed = int((~keep).sum())

        tmask = np.zeros(npad, dtype=bool)
        tmask[touched] = True
        written_mask = np.zeros(npad, dtype=bool)
        written_mask[w] = True

        if len(touched) > 0:
            r1 = (~tmask) & near[:, touched].any(axis=1)
            r1i = np.where(r1)[0]
            if len(r1i) > 0:
                r2 = (~tmask) & (~r1) & near[:, r1i].any(axis=1)
            else:
                r2 = np.zeros(npad, dtype=bool)
        else:
            r1 = np.zeros(npad, dtype=bool)
            r2 = np.zeros(npad, dtype=bool)

        rows.append([ev, n_w, n_removed, n_gh_true, n_gh_removed, n_gh_kept,
                     len(touched), int(r1.sum()), int(r2.sum()),
                     int((r1 & written_mask).sum() + (r2 & written_mask).sum())])

    tab = np.array(rows, dtype=float)
    names = ["event", "n_ecrits", "n_retires", "n_fantomes_vrais",
             "fantomes_retires", "fantomes_restants", "n_touches",
             "n_couronne1", "n_couronne2", "couronnes_ecrites(retires)"]
    fn = args.out + ".csv"
    np.savetxt(fn, tab, delimiter=",", header=",".join(names), comments="",
               fmt="%d")
    print("ecrit : %s  (%d evenements)" % (fn, tab.shape[0]))

    print("\n%-28s %8s %8s %8s %8s" % ("colonne", "min", "median", "95%", "max"))
    for j in range(1, tab.shape[1]):
        c = tab[:, j]
        print("%-28s %8.1f %8.1f %8.1f %8.1f"
              % (names[j], c.min(), np.median(c), np.percentile(c, 95), c.max()))

    tot_true = tab[:, 3].sum()
    if tot_true > 0:
        print("\nfantomes vrais : %d, retires par la regle : %d (%.1f %%), "
              "restants : %d" % (tot_true, tab[:, 4].sum(),
                                  100.0 * tab[:, 4].sum() / tot_true,
                                  tab[:, 5].sum()))
    print("evenements avec au moins un fantome restant : %d / %d"
          % (int((tab[:, 5] > 0).sum()), tab.shape[0]))
    print("evenements sans pad touche apres retrait : %d"
          % int((tab[:, 6] == 0).sum()))


if __name__ == "__main__":
    main()
