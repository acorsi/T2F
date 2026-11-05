#!/usr/bin/env python3
"""Comparaison de deux sorties de `process` (nouvelle version, avec induction) :
effet d'un changement de n_h (ou de tout paramètre numérique de la table) au niveau des pads.

Usage :
    python3 compare_nh.py reference.h5 test.h5 [--n 2000]

Condition : mêmes événements (1 thread, même graine, même fichier d'entrée) et mêmes seuils.
Tout écart vient alors du calcul de l'induction. Les écarts sont en électrons (e-),
test - référence.

Pour chaque fragment (H et L) et chaque pad écrit dans les deux fichiers :
  - charge finale pq ;
  - qt et Q aux bins gardés dans les deux fichiers ;
  - bins ou pads gardés dans un seul des deux fichiers (franchissement du seuil).
"""
import sys
import argparse
import numpy as np
import h5py


def unpack(f, frag, i, dt):
    """dict pad_id -> (k, qt, Q, q) ; k = indice de bin (entier)."""
    pid = np.asarray(f["/%s/pid" % frag][i], dtype=int)
    pnt = np.asarray(f["/%s/pnt" % frag][i], dtype=int)
    pq = np.asarray(f["/%s/pq" % frag][i])
    pt = np.asarray(f["/%s/pt" % frag][i])
    pqt = np.asarray(f["/%s/pqt" % frag][i])
    pQ = np.asarray(f["/%s/pQ" % frag][i])
    off = np.concatenate([[0], np.cumsum(pnt)])
    out = {}
    for k in range(len(pid)):
        s = slice(off[k], off[k + 1])
        out[int(pid[k])] = (np.rint(pt[s] / dt).astype(int), pqt[s], pQ[s], pq[k])
    return out


def pct(a, p):
    return np.percentile(np.abs(a), p) if len(a) else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()

    fr = h5py.File(a.ref, "r")
    ft = h5py.File(a.test, "r")
    mr = dict(fr["/metadata"].attrs)
    mt = dict(ft["/metadata"].attrs)
    dt = float(mr["delta_t"])
    s_pos = float(mr["seuil_qt_pos"])
    s_neg = float(mr["seuil_qt_neg"])
    for k in ("seuil_qt_pos", "seuil_qt_neg", "delta_t", "n_bins"):
        if k in mt and float(mt[k]) != float(mr[k]):
            print("ATTENTION : %s differe (%s contre %s)" % (k, mr[k], mt[k]))
    print("seuils : +%g / %g e-   delta_t = %g ns" % (s_pos, s_neg, dt))

    n_ev = min(a.n, fr["/H/dE"].shape[0], ft["/H/dE"].shape[0])
    for i in range(min(n_ev, 20)):
        if not np.array_equal(np.asarray(fr["/H/z"][i]), np.asarray(ft["/H/z"][i])):
            print("Les traces different (evenement %d) : comparaison impossible." % i)
            return 1

    d_pq, r_pq = [], []
    d_qt, d_Q = [], []
    worst_qt = (0.0, None)
    worst_Q = (0.0, None)
    n_frag = 0
    n_pad_common = 0
    n_pad_only_ref = n_pad_only_test = 0
    q_only_ref, q_only_test = [], []
    n_bin_common = n_bin_only_ref = n_bin_only_test = 0
    qt_only_ref, qt_only_test = [], []

    for i in range(n_ev):
        for frag in ("H", "L"):
            A = unpack(fr, frag, i, dt)
            B = unpack(ft, frag, i, dt)
            n_frag += 1
            for pid in set(A) | set(B):
                if pid in A and pid not in B:
                    n_pad_only_ref += 1
                    q_only_ref.append(A[pid][3])
                    continue
                if pid in B and pid not in A:
                    n_pad_only_test += 1
                    q_only_test.append(B[pid][3])
                    continue
                n_pad_common += 1
                kA, qtA, QA, qA = A[pid]
                kB, qtB, QB, qB = B[pid]
                d_pq.append(qB - qA)
                if qA > 0.0:
                    r_pq.append((qB - qA) / qA)
                common, ia, ib = np.intersect1d(kA, kB, return_indices=True)
                n_bin_common += len(common)
                if len(common):
                    e1 = qtB[ib] - qtA[ia]
                    e2 = QB[ib] - QA[ia]
                    d_qt.extend(e1.tolist())
                    d_Q.extend(e2.tolist())
                    j = int(np.argmax(np.abs(e1)))
                    if abs(e1[j]) > worst_qt[0]:
                        worst_qt = (abs(e1[j]), (i, frag, pid, int(common[j]) * dt, qtA[ia][j]))
                    j = int(np.argmax(np.abs(e2)))
                    if abs(e2[j]) > worst_Q[0]:
                        worst_Q = (abs(e2[j]), (i, frag, pid, int(common[j]) * dt, QA[ia][j]))
                only_a = np.setdiff1d(kA, kB)
                only_b = np.setdiff1d(kB, kA)
                n_bin_only_ref += len(only_a)
                n_bin_only_test += len(only_b)
                if len(only_a):
                    qt_only_ref.extend(qtA[np.isin(kA, only_a)].tolist())
                if len(only_b):
                    qt_only_test.extend(qtB[np.isin(kB, only_b)].tolist())

    d_pq = np.array(d_pq)
    r_pq = np.array(r_pq)
    d_qt = np.array(d_qt)
    d_Q = np.array(d_Q)

    print("\n%d fragments, %d pads communs, %d bins communs\n" % (n_frag, n_pad_common, n_bin_common))

    print("Charge finale pq (test - ref)")
    print("  absolu : max = %.3g e-   rms = %.3g e-" % (np.max(np.abs(d_pq)), np.sqrt(np.mean(d_pq ** 2))))
    print("  relatif : max = %.2e   rms = %.2e" % (np.max(np.abs(r_pq)), np.sqrt(np.mean(r_pq ** 2))))

    for nom, d, w in (("qt par bin", d_qt, worst_qt), ("Q cumulee", d_Q, worst_Q)):
        print("\n%s aux bins communs (test - ref), e-" % nom)
        print("  rms = %.3g   mediane|.| = %.3g   p90 = %.3g   p99 = %.3g   p99.9 = %.3g   max = %.3g"
              % (np.sqrt(np.mean(d ** 2)), pct(d, 50), pct(d, 90), pct(d, 99), pct(d, 99.9), pct(d, 100)))
        if w[1] is not None:
            print("  pire : evenement %d, %s, pad %d, t = %.0f ns (valeur ref = %.0f e-)" % w[1])

    print("\nFranchissements du seuil (par fragment)")
    print("  pads ecrits seulement dans ref  : %.3f   (charge mediane %s e-)"
          % (n_pad_only_ref / float(n_frag),
             "%.0f" % np.median(q_only_ref) if q_only_ref else "-"))
    print("  pads ecrits seulement dans test : %.3f   (charge mediane %s e-)"
          % (n_pad_only_test / float(n_frag),
             "%.0f" % np.median(q_only_test) if q_only_test else "-"))
    print("  bins gardes seulement dans ref  : %.3f par fragment (%.2e des bins communs)"
          % (n_bin_only_ref / float(n_frag), n_bin_only_ref / float(max(n_bin_common, 1))))
    print("  bins gardes seulement dans test : %.3f par fragment (%.2e des bins communs)"
          % (n_bin_only_test / float(n_frag), n_bin_only_test / float(max(n_bin_common, 1))))
    if qt_only_ref or qt_only_test:
        print("  (ces bins sont au voisinage du seuil : qt median |ref seul| = %s, |test seul| = %s e-)"
              % ("%.0f" % np.median(np.abs(qt_only_ref)) if qt_only_ref else "-",
                 "%.0f" % np.median(np.abs(qt_only_test)) if qt_only_test else "-"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
