#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Niveau 4 : chaine complete (./process) contre la reference EXACTE (Fourier + Gauss, test_level1.py).
#
# usage : python3 test_level4.py EVENT pad_file.txt sortie1.h5 [sortie2.h5 ...]
#   EVENT        : indice de l'evenement dans les sorties
#   pad_file.txt : my_pad.txt ecrit par ./process (id x y ...)
#   sortie*.h5   : sorties de ./process lancees avec  Fano=0  seuil_qt_pos=0  seuil_qt_neg=0  eps_qt=0
#                  (Fano = 0 : n_electrons = round(dE / W), donc connu ; seuils nuls : tous les pads touches sont ecrits)
#
# Reference : pour chaque voxel de H et de L (x, y, z, dE lus dans la sortie), n_e = round(dE/W),
#   charge d'un pad dans le bin k = n_e [ Psi(tau_haut) - Psi(tau_bas) ],  tau = k delta_t - d/ve_kg,  d = dkg - z,
#   (u, v) = (x, y) du voxel - centre du pad, Psi exact de test_level1.py.
# Pads testes : les plus charges, des pads moyens, des pads tres peu charges (induction seule).
# Voxels pris en compte : a moins de 3.2 cm du pad (le code coupe a reach_cm = 2.32 cm : la coupure est testee aussi).
# Un pad ecrit par le code dans H et dans L porte la somme H+L : on compare donc a la somme des deux fragments.
#
# Tolerance : 1e-3 par voxel (comme aux niveaux 1 et 2), soit pour un pad  1e-3 * somme des n_e des voxels
# qui le concernent (cas le plus defavorable : erreurs de meme signe sur tous les voxels).
# Les ecarts sont aussi donnes relativement au pic du pad.
# ---------------------------------------------------------------------------
import os
import sys
import numpy as np
import h5py
from multiprocessing import Pool

import test_level1 as L1

TOL = 1.0e-3
CUT = 3.2
OUT = "out/"


def load_meta(h):
    A = h["/metadata"].attrs
    p = float(A["pad_size"]) + float(A["inter_pad"])
    M = {"a": 0.5 * p, "pitch": p, "g": float(A["dga"]), "dkg": float(A["dkg"]), "W": float(A["W"]),
         "reach": float(A["reach_cm"]), "dt": float(A["delta_t"]), "n_bins": int(A["n_bins"])}
    for k in ("DT_kg", "DT_ga", "DL_kg", "DL_ga", "ve_kg", "ve_ga"):
        M[k] = float(A[k])
    return M


def voxels(h, ev, M):
    xs, ys, ds, ns = [], [], [], []
    for g in ("H", "L"):
        x = np.asarray(h["/%s/x" % g][ev]); y = np.asarray(h["/%s/y" % g][ev]); z = np.asarray(h["/%s/z" % g][ev])
        n = np.round(np.asarray(h["/%s/dE" % g][ev]) / M["W"])
        xs.append(x); ys.append(y); ds.append(M["dkg"] - z); ns.append(n)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ds), np.concatenate(ns)


def sim_pads(h, ev, K):
    """charge par pad et par bin (dense, K bins), union de H et L ; verifie que les pads communs sont identiques"""
    res, same = {}, True
    for g in ("H", "L"):
        pid = np.asarray(h["/%s/pid" % g][ev]).astype(int)
        k0 = np.asarray(h["/%s/pk0" % g][ev]).astype(int)
        nt = np.asarray(h["/%s/pnt" % g][ev]).astype(int)
        qt = np.asarray(h["/%s/pqt" % g][ev]).astype(float)
        off = np.concatenate([[0], np.cumsum(nt)])
        for j, p in enumerate(pid):
            arr = np.zeros(K)
            arr[k0[j]:k0[j] + nt[j]] = qt[off[j]:off[j + 1]]
            if p in res:
                if np.max(np.abs(res[p] - arr)) > 1e-3:
                    same = False
            else:
                res[p] = arr
    return res, same


def exact_pad(args):
    M, px, py, vx, vy, vd, vn, K = args
    q = np.zeros(K)
    ncontrib = 0.0
    dt = M["dt"]
    for x, y, d, n in zip(vx, vy, vd, vn):
        u, v = x - px, y - py
        if np.hypot(u, v) > CUT or n <= 0:
            continue
        ncontrib += n
        tg = d / M["ve_kg"]
        k_lo = int(np.floor((tg - 30.0) / dt)); k_hi = int(np.ceil((tg + 150.0) / dt))
        spl = L1.lambda_spline(M, u, v, d)
        sig_t = L1.sigma_t_of(M, d)
        ks = np.arange(k_lo, k_hi + 2)
        psi = np.array([L1.psi_exact(spl, M, sig_t, k * dt - tg) for k in ks])
        dq = n * np.diff(psi)
        for j, k in enumerate(ks[:-1]):
            if 0 <= k < K:
                q[k] += dq[j]
    return q, ncontrib


def exact_total(M, vx, vy, vd, vn, K):
    q = np.zeros(K)
    dt = M["dt"]
    for d, n in zip(vd, vn):
        tg = d / M["ve_kg"]
        k_lo = int(np.floor((tg - 30.0) / dt)); k_hi = int(np.ceil((tg + 150.0) / dt))
        ks = np.arange(k_lo, k_hi + 2)
        ramp = np.array([L1.ramp_exact(M, d, k * dt - tg) for k in ks])
        for j, k in enumerate(ks[:-1]):
            if 0 <= k < K:
                q[k] += n * (ramp[j + 1] - ramp[j])
    return q


def main():
    ev = int(sys.argv[1])
    padfile = sys.argv[2]
    files = sys.argv[3:]
    pads = {}
    for line in open(padfile):
        if line.startswith("#"):
            continue
        t = line.split()
        pads[int(t[0])] = (float(t[1]), float(t[2]))

    hs = [h5py.File(f, "r") for f in files]
    M = load_meta(hs[0])
    K = M["n_bins"] + 3
    vx, vy, vd, vn = voxels(hs[0], ev, M)
    print("evenement %d : %d voxels (H+L), charge totale en entree %.0f e-" % (ev, len(vx), vn.sum()))
    sims = []
    for f, h in zip(files, hs):
        s, same = sim_pads(h, ev, K)
        sims.append(s)
        print("%s : %d pads ecrits ; pads communs a H et L identiques : %s" % (f, len(s), "oui" if same else "NON"))

    # ---- choix des pads : par charge decroissante (sortie de la premiere version) ----
    order = sorted(sims[0], key=lambda p: -np.sum(np.abs(sims[0][p])))
    sel = [order[i] for i in (0, 1, 2, 3, 4, 5, 10, 20, 40, 70, 110, len(order) - 1) if i < len(order)]
    cache = OUT + "l4_exact_ev%d.npz" % ev
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        ex = z["ex"].item()
        sel = list(ex.keys())                 # memes pads que lors du calcul de la reference
        print("reference exacte relue : " + cache)
    else:
        tasks = [(M, pads[p][0], pads[p][1], vx, vy, vd, vn, K) for p in sel]
        with Pool(2) as pool:
            res = pool.map(exact_pad, tasks)
        ex = dict((p, r) for p, r in zip(sel, res))
        np.savez(cache, ex=np.array(ex, dtype=object))
        print("reference exacte calculee : " + cache)

    # ---- comparaison par pad ----
    ok = True
    for f, s in zip(files, sims):
        print("\n=== %s ===" % f)
        print("%-6s %-9s %-12s %-12s %-12s %-12s %-10s" % ("pad", "pic (e-)", "ecart max/bin", "ecart cumule", "tol (e-)", "ecart/pic", "bilan"))
        worst = 0.0
        for p in sel:
            q_ex, nc = ex[p]
            q_si = s.get(p, np.zeros(K))
            e_bin = float(np.max(np.abs(q_si - q_ex)))
            e_cum = float(np.max(np.abs(np.cumsum(q_si) - np.cumsum(q_ex))))
            tol = TOL * nc
            peak = float(np.max(np.abs(q_ex)))
            good = (e_bin <= tol) and (e_cum <= tol)
            if not good:
                ok = False
            worst = max(worst, e_bin / max(tol, 1e-30))
            print("%-6d %-9.0f %-12.1f %-12.1f %-12.1f %-12.2e %-10s" % (p, peak, e_bin, e_cum, tol, e_bin / max(peak, 1e-30), "OK" if good else "ECHEC"))
        print("pire rapport ecart/tolerance : %.2f" % worst)

    # ---- charge totale et profil temporel de la somme des pads ----
    tot_ex = exact_total(M, vx, vy, vd, vn, K)
    print("\n=== somme sur tous les pads, bin par bin, contre la rampe exacte ===")
    for f, s in zip(files, sims):
        tot = np.zeros(K)
        for arr in s.values():
            tot += arr
        e_bin = float(np.max(np.abs(tot - tot_ex)))
        e_cum = float(np.max(np.abs(np.cumsum(tot) - np.cumsum(tot_ex))))
        tol = TOL * vn.sum()
        print("%s : charge finale code %.0f, exacte %.0f (ecart relatif %.2e) ; ecart max par bin %.1f e-, cumule %.1f e- ; tolerance 1e-3 x charge = %.0f e-  %s" %
              (f, tot.sum(), tot_ex.sum(), (tot.sum() - tot_ex.sum()) / tot_ex.sum(), e_bin, e_cum, tol, "OK" if max(e_bin, e_cum) <= tol else "ECHEC"))
        if max(e_bin, e_cum) > tol:
            ok = False
    print("\nBILAN NIVEAU 4 : " + ("tout est OK" if ok else "ATTENTION : un critere echoue"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
