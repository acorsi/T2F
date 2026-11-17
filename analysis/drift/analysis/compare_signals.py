#!/usr/bin/env python3
"""Compare les SIGNAUX de pads de deux sorties de ./process (meme evenements, meme geometrie),
en alignant les bins sur la grille absolue (t = k * delta_t) : pad par pad, evenement par evenement.

usage : compare_signals.py a.h5 b.h5 [--tol_q 5] [--tol_pq 1e-4]

Mesures, par fragment (H, L) :
  - pads presents d'un seul cote (avec leur charge pq et leur |qt| max)
  - pads communs : ecart max sur pq (e-, et relatif pour les pads > 1000 e-)
  - signal qt : ecart max (e-) apres alignement (bin absent = 0), et ou il se trouve ;
    ecart relatif au |qt| max du pad (pads > 500 e-)
  - bins "orphelins" : bin ecrit d'un seul cote (coupe par eps_qt / seuil) : leur |qt| max
Code retour 0 si ecart qt <= tol_q (e-) et ecart pq relatif <= tol_pq."""
import sys
import argparse
import numpy as np
import h5py


def frag_event(f, frag, i):
    pid = np.asarray(f["/%s/pid" % frag][i]).astype(int)
    pk0 = np.asarray(f["/%s/pk0" % frag][i]).astype(int)
    pnt = np.asarray(f["/%s/pnt" % frag][i]).astype(int)
    pq = np.asarray(f["/%s/pq" % frag][i], dtype=float)
    pqt = np.asarray(f["/%s/pqt" % frag][i], dtype=float)
    off = np.concatenate([[0], np.cumsum(pnt)])
    out = {}
    for k in range(len(pid)):
        out[int(pid[k])] = (int(pk0[k]), pqt[off[k]:off[k + 1]], float(pq[k]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--tol_q", type=float, default=5.0)
    ap.add_argument("--tol_pq", type=float, default=1e-4)
    args = ap.parse_args()
    bad = 0
    with h5py.File(args.a, "r") as fa, h5py.File(args.b, "r") as fb:
        n = min(fa["/H/dE"].shape[0], fb["/H/dE"].shape[0])
        if fa["/H/dE"].shape[0] != fb["/H/dE"].shape[0]:
            print("ATTENTION : nombres d'evenements differents, on compare les %d premiers" % n)
        for frag in ("H", "L"):
            n_pads_only = 0; only_pq = 0.0; only_qt = 0.0
            dpq_abs = 0.0; dpq_rel = 0.0
            dq_max = 0.0; dq_where = None
            rel_max = 0.0; rel_where = None
            orph_max = 0.0; n_orph = 0; n_pad_cmp = 0; n_k0_diff = 0
            for i in range(n):
                A = frag_event(fa, frag, i); B = frag_event(fb, frag, i)
                for p in set(A) ^ set(B):
                    n_pads_only += 1
                    k0, q, pq = A[p] if p in A else B[p]
                    only_pq = max(only_pq, abs(pq))
                    if q.size: only_qt = max(only_qt, float(np.max(np.abs(q))))
                for p in set(A) & set(B):
                    k0a, qa, pqa = A[p]; k0b, qb, pqb = B[p]
                    n_pad_cmp += 1
                    d = abs(pqa - pqb)
                    dpq_abs = max(dpq_abs, d)
                    if abs(pqa) > 1000.0: dpq_rel = max(dpq_rel, d / abs(pqa))
                    lo = min(k0a, k0b); hi = max(k0a + len(qa), k0b + len(qb))
                    xa = np.zeros(hi - lo); xb = np.zeros(hi - lo)
                    xa[k0a - lo:k0a - lo + len(qa)] = qa
                    xb[k0b - lo:k0b - lo + len(qb)] = qb
                    dd = np.abs(xa - xb)
                    amp = max(float(np.max(np.abs(xa))) if xa.size else 0.0, float(np.max(np.abs(xb))) if xb.size else 0.0)
                    if dd.size and amp > 500.0 and dd.max() / amp > rel_max:
                        rel_max = float(dd.max() / amp); rel_where = (i, p, amp)
                    if dd.size and dd.max() > dq_max:
                        dq_max = float(dd.max()); dq_where = (i, p, lo + int(dd.argmax()))
                    if k0a != k0b or len(qa) != len(qb):
                        n_k0_diff += 1
                    # bins ecrits d'un seul cote
                    ma = np.zeros(hi - lo, dtype=bool); mb = np.zeros(hi - lo, dtype=bool)
                    ma[k0a - lo:k0a - lo + len(qa)] = True; mb[k0b - lo:k0b - lo + len(qb)] = True
                    orph = (ma ^ mb)
                    if orph.any():
                        n_orph += int(orph.sum())
                        orph_max = max(orph_max, float(np.max(np.abs((xa + xb)[orph]))))
            ok = (dq_max <= args.tol_q) and (dpq_rel <= args.tol_pq)
            bad += (not ok)
            print("== %s : %d evenements, %d pads communs" % (frag, n, n_pad_cmp))
            print("   pads presents d'un seul cote : %d   (pq max %.3g e-, |qt| max %.3g e-)"
                  % (n_pads_only, only_pq, only_qt))
            print("   pq  : ecart max %.3g e-   ecart relatif max (pads > 1000 e-) %.3g" % (dpq_abs, dpq_rel))
            print("   qt  : ecart max %.3g e-   %s" % (dq_max, ("(evt %d, pad %d, bin %d)" % dq_where) if dq_where else ""))
            print("   qt relatif (pads dont |qt| max > 500 e-) : ecart max / |qt| max du pad = %.3g   %s"
                  % (rel_max, ("(evt %d, pad %d, |qt| max %.0f e-)" % rel_where) if rel_where else ""))
            print("   pads dont les bins ecrits different : %d ; bins orphelins : %d, |qt| max d'un bin orphelin = %.3g e-"
                  % (n_k0_diff, n_orph, orph_max))
            print("   %s" % ("OK" if ok else "DIFF"))
    return 0 if bad == 0 else 1


sys.exit(main())
