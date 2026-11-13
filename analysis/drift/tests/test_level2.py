#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Niveau 2 (suite) : project_pair contre la reference exacte (Fourier + Gauss), et figures.
# Lit les fichiers de ./test_level2 : out/l2_meta.txt, l2_ref.csv, l2_sumbins.csv, l2_bound.csv.
#
#   (a) charge par pad et par bin de lecture : n_electrons = 1, donc q_bin = Psi(bord haut) - Psi(bord bas),
#       avec Psi exact (test_level1.py) en (u, v, d) du voxel et tau = k delta_t - d / ve_kg
#   (b) somme sur tous les pads, bin par bin : doit valoir l'increment de la rampe exacte
#   (c) borne dpsi_max (selection par seuil) contre le maximum reel : figure
# Tolerance : 1e-3 en absolu, en fraction de la charge du voxel.
# ---------------------------------------------------------------------------
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import test_level1 as L1

OUT = "out/"
TOL = 1.0e-3


def read_meta():
    M = {}
    with open(OUT + "l2_meta.txt") as fh:
        for line in fh:
            k, v = line.strip().split("=")
            try:
                M[k] = float(v)
            except ValueError:
                M[k] = v
    return M


def main():
    M = read_meta()
    dt = M["delta_t"]
    ve_kg = M["ve_kg"]

    # ---------------- (a) par pad et par bin ----------------
    pads = {}
    with open(OUT + "l2_ref.csv") as fh:
        for r in csv.DictReader(fh):
            key = (int(r["vox"]), float(r["pad_x"]), float(r["pad_y"]))
            e = pads.setdefault(key, {"x": float(r["x"]), "y": float(r["y"]), "d": float(r["d"]), "q": {}})
            e["q"][int(r["k"])] = float(r["q"])

    sim_all, ref_all, cum_err = [], [], []
    per_vox = {}
    for key in sorted(pads):
        vox, px, py = key
        e = pads[key]
        u, v, d = e["x"] - px, e["y"] - py, e["d"]
        tg = d / ve_kg
        spl = L1.lambda_spline(M, u, v, d)
        sig_t = L1.sigma_t_of(M, d)
        ks = sorted(e["q"])
        k_lo, k_hi = ks[0] - 2, ks[-1] + 2
        edges = np.arange(k_lo, k_hi + 2) * dt - tg
        psi = np.array([L1.psi_exact(spl, M, sig_t, t) for t in edges])
        q_ref = np.diff(psi)
        q_sim = np.array([e["q"].get(k, 0.0) for k in range(k_lo, k_hi + 1)])
        sim_all.append(q_sim)
        ref_all.append(q_ref)
        # charge cumulee : somme des bins depuis le debut de la fenetre = Psi(bord haut) - Psi(bord bas du premier bin)
        cum_err.append(np.max(np.abs(np.cumsum(q_sim) - (psi[1:] - psi[0]))))
        w = per_vox.setdefault(vox, [0.0, 0.0, 0])
        w[0] = max(w[0], float(np.max(np.abs(q_sim - q_ref))))
        w[1] = max(w[1], float(cum_err[-1]))
        w[2] += 1
        sys.stdout.flush()

    sim = np.concatenate(sim_all)
    ref = np.concatenate(ref_all)
    diff = sim - ref
    err_bin = float(np.max(np.abs(diff)))
    err_cum = float(np.max(cum_err))

    print("(a) charge par pad et par bin, 6 voxels, pads a moins de 1.2 cm du voxel (n_electrons = 1) :")
    print("    %-4s %-8s %-14s %-16s" % ("vox", "pads", "ecart max bin", "ecart max cumule"))
    for vox in sorted(per_vox):
        w = per_vox[vox]
        print("    %-4d %-8d %-14.2e %-16.2e" % (vox, w[2], w[0], w[1]))
    ok_a = (err_bin <= TOL) and (err_cum <= TOL)
    print("    maximum : bin %.2e  cumule %.2e   (tolerance %.0e)  %s" % (err_bin, err_cum, TOL, "OK" if ok_a else "ECHEC"))

    # ---------------- (b) somme des pads, bin par bin ----------------
    sb = {}
    with open(OUT + "l2_sumbins.csv") as fh:
        for r in csv.DictReader(fh):
            sb.setdefault(int(r["vox"]), {})[int(r["k"])] = (float(r["d"]), float(r["qsum"]))
    err_sum_bin, err_sum_cum = 0.0, 0.0
    for vox in sorted(sb):
        ks = sorted(sb[vox])
        d = sb[vox][ks[0]][0]
        tg = d / ve_kg
        k_lo, k_hi = ks[0] - 2, ks[-1] + 2
        edges = np.arange(k_lo, k_hi + 2) * dt - tg
        ramp = np.array([L1.ramp_exact(M, d, t) for t in edges])
        q_ref = np.diff(ramp)
        q_sim = np.array([sb[vox].get(k, (d, 0.0))[1] for k in range(k_lo, k_hi + 1)])
        err_sum_bin = max(err_sum_bin, float(np.max(np.abs(q_sim - q_ref))))
        err_sum_cum = max(err_sum_cum, float(np.max(np.abs(np.cumsum(q_sim) - (ramp[1:] - ramp[0])))))
    ok_b = (err_sum_bin <= TOL) and (err_sum_cum <= TOL)
    print("(b) somme sur tous les pads, bin par bin, contre la rampe exacte (6 voxels) : bin %.2e  cumule %.2e  %s" %
          (err_sum_bin, err_sum_cum, "OK" if ok_b else "ECHEC"))

    # ---------------- (c) borne ----------------
    act, bnd = [], []
    with open(OUT + "l2_bound.csv") as fh:
        for r in csv.DictReader(fh):
            a, b = float(r["actual"]), float(r["bound"])
            if a > 1.0:
                act.append(a)
                bnd.append(b)
    act, bnd = np.array(act), np.array(bnd)
    ratio = bnd / act
    ok_c = bool(np.min(ratio) * 1.3 >= 1.0)
    print("(c) borne dpsi_max / max reel : min %.3f, mediane %.3f, 95e centile %.3f (%d pads) ; borne x 1.3 >= reel partout : %s" %
          (np.min(ratio), np.median(ratio), np.percentile(ratio, 95), len(ratio), "OK" if ok_c else "ECHEC"))

    # ---------------- figures ----------------
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    ax[0].plot(ref, sim, ".", ms=3, color="C0")
    lim = [min(ref.min(), sim.min()), max(ref.max(), sim.max())]
    ax[0].plot(lim, lim, "k-", lw=0.6)
    ax[0].set_xlabel("charge d'un bin de 10 ns / n_electrons : exact")
    ax[0].set_ylabel("project_pair")
    ax[0].set_title("Charge par pad et par bin, 6 voxels")
    ax[1].plot(ref, diff, ".", ms=3, color="C0")
    ax[1].axhline(TOL, color="gray", ls="--", lw=0.8)
    ax[1].axhline(-TOL, color="gray", ls="--", lw=0.8)
    ax[1].set_ylim(-1.2 * TOL, 1.2 * TOL)
    ax[1].set_xlabel("exact")
    ax[1].set_ylabel("project_pair - exact")
    ax[1].set_title("Ecart (max %.1e) ; traits gris : +-1e-3" % err_bin)
    ax[2].loglog(act, bnd, ".", ms=3, color="C0", label="pads (24 traces)")
    xx = np.array([1.0, act.max() * 1.5])
    ax[2].loglog(xx, xx, "k-", lw=0.8, label="borne = reel")
    ax[2].loglog(xx, xx / 1.3, "r--", lw=0.8, label="limite pour la marge 1.3 (borne x 1.3 = reel)")
    for thr in (200.0, 2000.0):
        ax[2].axvline(thr, color="gray", ls=":", lw=0.8)
        ax[2].axhline(thr / 1.3, color="gray", ls=":", lw=0.8)
    ax[2].set_xlabel("max |qt| reel sur les bins du pad (electrons)")
    ax[2].set_ylabel("borne dpsi_max, marge 1 (electrons)")
    ax[2].set_title("Borne de selection / maximum reel (au-dessus de la rouge)", fontsize=10)
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT + "l2_exact.png", dpi=105)
    plt.close(fig)
    print("figure : out/l2_exact.png")

    ok = ok_a and ok_b and ok_c
    print("BILAN NIVEAU 2 (partie Python) : " + ("tout est OK" if ok else "ATTENTION : un critere echoue"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
