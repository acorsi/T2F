#!/usr/bin/env python3
"""Identité des pads écrits dans le fichier de référence mais absents du fichier test.

Usage :
    python3 missing_pads.py ref.h5 test.h5 [my_pad.txt] [--n 2000] [--list 40]

Pour chaque pad manquant : événement, fragment, pad_id, type (carré / transition / anneau),
position, charge finale pq, plus grand qt (et son instant), nombre de bins gardés,
distance minimale entre le centre du pad et la trace dans le plan (x, y).
Résumé : décompte par type, répartition des pad_id, distributions de pq et de qt max.
"""
import sys
import argparse
import numpy as np
import h5py


def unpack(f, frag, i, dt):
    pid = np.asarray(f["/%s/pid" % frag][i], dtype=int)
    pnt = np.asarray(f["/%s/pnt" % frag][i], dtype=int)
    pq = np.asarray(f["/%s/pq" % frag][i])
    px = np.asarray(f["/%s/px" % frag][i])
    py = np.asarray(f["/%s/py" % frag][i])
    pt = np.asarray(f["/%s/pt" % frag][i])
    pqt = np.asarray(f["/%s/pqt" % frag][i])
    off = np.concatenate([[0], np.cumsum(pnt)])
    out = {}
    for k in range(len(pid)):
        s = slice(off[k], off[k + 1])
        out[int(pid[k])] = (pt[s], pqt[s], pq[k], px[k], py[k])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("pad_file", nargs="?", default="my_pad.txt")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--list", type=int, default=40)
    a = ap.parse_args()

    fr = h5py.File(a.ref, "r")
    ft = h5py.File(a.test, "r")
    dt = float(dict(fr["/metadata"].attrs)["delta_t"])
    thr = float(dict(fr["/metadata"].attrs)["seuil_qt_pos"])

    geom = {}
    try:
        for row in np.loadtxt(a.pad_file):
            geom[int(row[0])] = row[1:]          # x y r_in r_out sx sy
    except Exception:
        print("pad_file illisible : types de pads non disponibles")

    def kind(pid):
        g = geom.get(pid)
        if g is None:
            return "?"
        if g[3] == 0.0:
            return "carre"
        return "transition" if g[2] < 8.15 else "anneau"

    n_ev = min(a.n, fr["/H/dE"].shape[0], ft["/H/dE"].shape[0])
    rows = []
    n_frag = 0
    for i in range(n_ev):
        for frag in ("H", "L"):
            A = unpack(fr, frag, i, dt)
            B = unpack(ft, frag, i, dt)
            n_frag += 1
            x = np.asarray(fr["/%s/x" % frag][i])
            y = np.asarray(fr["/%s/y" % frag][i])
            theta = float(fr["/%s/theta" % frag][i])
            for pid in A:
                if pid in B:
                    continue
                t, qt, q, px, py = A[pid]
                j = int(np.argmax(qt))
                if kind(pid) == "carre" and len(x):
                    dmin = float(np.min(np.hypot(x - px, y - py)))
                else:
                    dmin = float("nan")
                rows.append((i, frag, pid, kind(pid), px, py, q, qt[j], t[j], len(qt),
                             dmin, theta))

    print("%d fragments, %d pads seulement dans ref (%.3f par fragment)\n"
          % (n_frag, len(rows), len(rows) / float(max(n_frag, 1))))
    if not rows:
        return 0

    kinds = {}
    for r in rows:
        kinds[r[3]] = kinds.get(r[3], 0) + 1
    print("Par type :", ", ".join("%s = %d" % (k, v) for k, v in sorted(kinds.items())))

    q = np.array([r[6] for r in rows])
    m = np.array([r[7] for r in rows])
    d = np.array([r[10] for r in rows])
    print("pq        : min = %.0f  mediane = %.0f  max = %.0f e-" % (q.min(), np.median(q), q.max()))
    print("qt max    : min = %.0f  mediane = %.0f  max = %.0f e-   (seuil = %.0f)" % (m.min(), np.median(m), m.max(), thr))
    print("qt max / seuil : min = %.2f  mediane = %.2f" % (m.min() / thr, np.median(m) / thr))
    dd = d[~np.isnan(d)]
    if len(dd):
        print("distance pad-trace (carres) : min = %.2f  mediane = %.2f  max = %.2f cm" % (dd.min(), np.median(dd), dd.max()))

    ids = {}
    for r in rows:
        ids[r[2]] = ids.get(r[2], 0) + 1
    top = sorted(ids.items(), key=lambda kv: -kv[1])[:10]
    print("pad_id les plus frequents :", ", ".join("%d (x%d)" % kv for kv in top))

    print("\n%-6s %-2s %-6s %-10s %8s %8s %10s %9s %7s %4s %7s %6s"
          % ("evt", "fr", "pad", "type", "x", "y", "pq", "qt max", "t(ns)", "nb", "dmin", "theta"))
    rows.sort(key=lambda r: -r[6])
    for r in rows[:a.list]:
        print("%-6d %-2s %-6d %-10s %8.2f %8.2f %10.0f %9.0f %7.0f %4d %7.2f %6.2f" % r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
