#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Test "trace" : induction sur les pads voisins d'une trace courte (cas A, drift.cfg).
# Lit out/track_simu.csv, out/track_lc.csv, out/track_meta.txt (produits par ./test_track).
#
# Pour chaque cas (1 : trace horizontale ; 2 : trace a 45 degres dans le plan (x, d)) et chaque pad :
#   total simu     = somme sur les voxels de Psi lu dans la table du code
#   total exact    = meme somme, avec Psi calcule par la reference de test_level1.py (Fourier + Gauss)
#   collecte exact = signal des electrons qui FINISSENT sur le pad (Lc de test_track.cpp, meme modele
#                    temporel que le total)
#   induit exact   = total exact - collecte exact = signal des electrons qui finissent ailleurs
# Tolerance : 1e-3 en absolu, en fraction de la charge totale de la trace.
# ---------------------------------------------------------------------------
import csv
import sys
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import test_level1 as L1

OUT = "out/"
TOL = 1.0e-3
PADN = ["P0 (depart)", "P1 (voisin, meme rangee)", "P2 (quinconce, au-dessus du milieu)", "P3 (rangee du dessous)"]
CASEN = {1: "cas 1 : trace horizontale, d = 10 cm", 2: "cas 2 : trace a 45 degres (d de 9.74 a 10.26 cm)"}


class SqrtSpline(object):
    """Lc(h) interpole en fonction de sqrt(h) (le modele contient s_h = DT_ga sqrt(h))"""
    def __init__(self, h, lc):
        self.cs = CubicSpline(np.sqrt(h), lc)

    def __call__(self, h):
        return self.cs(np.sqrt(np.clip(h, 0.0, None)))


def read_meta():
    M = {}
    with open(OUT + "track_meta.txt") as fh:
        for line in fh:
            k, v = line.strip().split("=")
            M[k] = float(v)
    return M


def main():
    M = read_meta()
    N = int(M["n_vox"])
    t_ref = M["t_ref"]
    ve_kg = M["ve_kg"]
    dt = M["delta_t"]

    # ---- simu ----
    simu = {}
    tgrid = None
    with open(OUT + "track_simu.csv") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        simu.setdefault((int(r["case"]), int(r["pad"])), []).append(float(r["Q"]))
    tgrid = np.array([float(r["t_rel"]) for r in rows if int(r["case"]) == 1 and int(r["pad"]) == 0])
    simu = dict((k, np.array(v)) for k, v in simu.items())

    # ---- partie collectee : Lc(h) par (cas, pad, voxel) ----
    lc = {}
    info = {}
    with open(OUT + "track_lc.csv") as fh:
        for r in csv.DictReader(fh):
            key = (int(r["case"]), int(r["pad"]), int(r["k"]))
            lc.setdefault(key, []).append((float(r["h"]), float(r["Lc"])))
            info[key] = (float(r["u"]), float(r["v"]), float(r["d"]))

    # ---- references ----
    ref_tot, ref_col = {}, {}
    plat_exact = {}
    for cs in (1, 2):
        for ip in range(4):
            tot = np.zeros(len(tgrid))
            col = np.zeros(len(tgrid))
            pl = 0.0
            for k in range(N):
                u, v, d = info[(cs, ip, k)]
                tg_rel = d / ve_kg - t_ref
                sig_t = L1.sigma_t_of(M, d)
                spl_tot = L1.lambda_spline(M, u, v, d)
                h_arr = np.array([x[0] for x in lc[(cs, ip, k)]])
                l_arr = np.array([x[1] for x in lc[(cs, ip, k)]])
                spl_col = SqrtSpline(h_arr, l_arr)
                for it, t in enumerate(tgrid):
                    tau = t - tg_rel
                    tot[it] += L1.psi_exact(spl_tot, M, sig_t, tau) / N
                    col[it] += L1.psi_exact(spl_col, M, sig_t, tau) / N
                pl += L1.plateau_exact(M, u, v, d) / N
            ref_tot[(cs, ip)] = tot
            ref_col[(cs, ip)] = col
            plat_exact[(cs, ip)] = pl
            print("cas %d pad P%d : fait" % (cs, ip))
            sys.stdout.flush()

    # ---- verifications ----
    print("")
    print("%-5s %-4s %12s %12s %12s %12s %12s %12s" %
          ("cas", "pad", "max|simu-ex|", "plateau ex.", "tot(fin)-pl", "coll(fin)-pl", "induit(fin)", "max |ind|"))
    ok = True
    for cs in (1, 2):
        for ip in range(4):
            s, tt, cc = simu[(cs, ip)], ref_tot[(cs, ip)], ref_col[(cs, ip)]
            e = float(np.max(np.abs(s - tt)))
            ind = tt - cc
            pl = plat_exact[(cs, ip)]
            print("%-5d P%-3d %12.2e %12.4f %12.2e %12.2e %12.2e %12.4f" %
                  (cs, ip, e, pl, tt[-1] - pl, cc[-1] - pl, ind[-1], float(np.max(np.abs(ind)))))
            if e > TOL:
                ok = False
            if abs(tt[-1] - pl) > TOL or abs(cc[-1] - pl) > TOL or abs(ind[-1]) > TOL:
                ok = False
    print("")
    print("BILAN TRACE : " + ("tout est OK (ecart simu-exact < %.0e, plateaux et induit final coherents)" % TOL if ok
                              else "ATTENTION : un critere depasse %.0e" % TOL))

    # ---- figure de geometrie ----
    p, a = M["pitch"], M["a"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for ip in range(4):
        x0, y0 = M["pad%d_x" % ip], M["pad%d_y" % ip]
        ax[0].add_patch(Rectangle((x0 - a, y0 - a), 2 * a, 2 * a, fc="#dfe8f3", ec="#4a6fa5", lw=1.2))
        ax[0].text(x0, y0 + 0.12, "P%d" % ip, ha="center", va="center", fontsize=12)
    ax[0].plot([0, p], [0, 0], "r-", lw=2.5, label="trace (projection)")
    ax[0].plot([0, p], [0, 0], "ko", ms=3)
    ax[0].set_aspect("equal")
    ax[0].set_xlim(-0.6, 1.1)
    ax[0].set_ylim(-0.9, 0.9)
    ax[0].set_xlabel("x (cm)")
    ax[0].set_ylabel("y (cm)")
    ax[0].set_title("Vue de dessus : pads du cas A et trace")
    ax[0].legend(loc="upper right", fontsize=8)
    xs = np.array([0.0, p])
    ax[1].plot(xs, [10.0, 10.0], "-o", color="C0", label="cas 1 : horizontale")
    ax[1].plot(xs, [10.0 - 0.5 * p, 10.0 + 0.5 * p], "-o", color="C3", label="cas 2 : 45 degres")
    ax[1].set_xlabel("x (cm)")
    ax[1].set_ylabel("d = distance de derive (cm)")
    ax[1].set_title("Vue de cote : d en fonction de x (arrivee a la grille a d / ve_kg)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT + "track_geom.png", dpi=110)
    plt.close(fig)

    # ---- figures de signaux ----
    edges = np.arange(tgrid[0], tgrid[-1] + 0.5, dt)
    idx = [int(round((e - tgrid[0]) / (tgrid[1] - tgrid[0]))) for e in edges]
    for cs in (1, 2):
        fig, ax = plt.subplots(3, 3, figsize=(16, 10), sharex="col")
        for ir, ip in enumerate((0, 1, 2)):
            s, tt, cc = simu[(cs, ip)], ref_tot[(cs, ip)], ref_col[(cs, ip)]
            ind = tt - cc
            a0 = ax[ir, 0]
            a0.plot(tgrid, tt, "k-", lw=1.8, label="total exact")
            a0.plot(tgrid[::5], s[::5], "o", color="k", mfc="none", ms=5, label="total simu")
            a0.plot(tgrid, cc, "-", color="C0", lw=1.6, label="collecte (finissent sur le pad)")
            a0.plot(tgrid, ind, "-", color="C3", lw=1.6, label="induit (finissent ailleurs)")
            a0.axhline(0, color="gray", lw=0.5)
            a0.set_ylabel(PADN[ip] + "\ncharge cumulee / charge de la trace", fontsize=8)
            if ir == 0:
                a0.set_title("Charge cumulee (= sortie d'une electronique ideale a integration)")
                a0.legend(fontsize=7, loc="upper left")
            a1 = ax[ir, 1]
            qb = lambda q: np.diff(np.array([q[i] for i in idx]))
            a1.step(edges[:-1], qb(tt), where="post", color="k", lw=1.6, label="total exact")
            a1.plot(edges[:-1] + 0.5 * dt, qb(s), "o", color="k", mfc="none", ms=5, label="total simu")
            a1.step(edges[:-1], qb(cc), where="post", color="C0", lw=1.4)
            a1.step(edges[:-1], qb(ind), where="post", color="C3", lw=1.4)
            a1.axhline(0, color="gray", lw=0.5)
            a1.set_ylabel("charge par bin de %g ns" % dt, fontsize=8)
            if ir == 0:
                a1.set_title("Charge par bin de lecture (ce que lit l'analyse)")
                a1.legend(fontsize=7, loc="upper left")
            a2 = ax[ir, 2]
            a2.plot(tgrid, s - tt, "-", color="k", lw=1.2, label="cumulee")
            a2.plot(edges[:-1] + dt, qb(s) - qb(tt), "o--", color="C1", ms=3, lw=0.8, label="par bin")
            a2.axhline(TOL, color="gray", ls="--", lw=0.8)
            a2.axhline(-TOL, color="gray", ls="--", lw=0.8)
            a2.set_ylim(-1.2 * TOL, 1.2 * TOL)
            a2.set_ylabel("simu - exact", fontsize=8)
            if ir == 0:
                a2.set_title("Ecart simu - exact (total) ; traits gris : tolerance +-1e-3")
                a2.legend(fontsize=7, loc="upper left")
        for c in range(3):
            ax[2, c].set_xlabel("t - 1000 ns (arrivee a la grille d'un voxel a d = 10 cm)")
        fig.suptitle("Trace courte, " + CASEN[cs] + "  (%d voxels, pas %.3g mm, charge totale 1)" % (N, M["step"] * 10), fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(OUT + "track_case%d.png" % cs, dpi=105)
        plt.close(fig)
    print("figures : out/track_geom.png  out/track_case1.png  out/track_case2.png")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
