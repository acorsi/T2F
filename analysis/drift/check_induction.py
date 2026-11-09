#!/usr/bin/env python3
"""Contrôles chiffrés de l'induction sur la sortie de `process`.

Usage :
    ./process 200 0 0            # IMPORTANT : seuils à 0 pour tout lire
    python3 check_induction.py sortie.h5 [my_pad.txt] [--old ancien.h5] [--n 200]

Trois contrôles indépendants du code C++ (recalcul en numpy à partir des voxels x, y, z, dE
écrits dans le fichier), fragment par fragment (H et L) :

  A. Conservation : somme des charges finales pq de tous les pads / somme des électrons.
     Attendu : 1 à ~1e-3 près pour une trace entièrement dans le plan actif.
  B. Courant total : somme sur les pads de la charge cumulée pQ à chaque instant,
     comparée à la rampe attendue (chaque voxel arrive à t_grid = d/ve_kg, puis traverse
     le gap pendant dga/ve_ga, avec l'étalement gaussien sigma_t). Vérifie le zéro du
     temps, les unités et la phase des bins.
  C. Charge finale par pad carré : comparée à Σ_voxels N · fx · fy (intégrale gaussienne sur
     la tuile avec sigma_anode). Le RMS de l'écart doit être celui des fluctuations
     statistiques (Fano/Poisson), et l'excès systématique << la charge de l'événement.
  F. (si --old) comparaison pad par pad de la charge finale avec l'ancienne version.

Seuls les événements dont toutes les traces restent à plus de 1 cm du bord actif sont
utilisés pour A, B et C.
"""
import sys
import argparse
import numpy as np
import h5py
from scipy.special import ndtr

SQ2PI = np.sqrt(2.0 * np.pi)


def unpack(f, frag, i):
    """Retourne la liste des pads d'un événement : dict pid, x, y, q, t, qt, Q."""
    pid = np.asarray(f[f"/{frag}/pid"][i], dtype=int)
    pnt = np.asarray(f[f"/{frag}/pnt"][i], dtype=int)
    px = np.asarray(f[f"/{frag}/px"][i]); py = np.asarray(f[f"/{frag}/py"][i])
    pq = np.asarray(f[f"/{frag}/pq"][i])
    pt = np.asarray(f[f"/{frag}/pt"][i]); pqt = np.asarray(f[f"/{frag}/pqt"][i])
    pQ = np.asarray(f[f"/{frag}/pQ"][i])
    off = np.concatenate([[0], np.cumsum(pnt)])
    pads = []
    for k in range(len(pid)):
        s = slice(off[k], off[k + 1])
        pads.append(dict(pid=pid[k], x=px[k], y=py[k], q=pq[k], t=pt[s], qt=pqt[s], Q=pQ[s]))
    return pads


def ramp(tau, sigma, T):
    """E[clip((tau - delta)/T, 0, 1)] pour delta ~ N(0, sigma^2)."""
    def g(m):
        z = m / sigma
        return m * ndtr(z) + sigma * np.exp(-0.5 * z * z) / SQ2PI
    return (g(tau) - g(tau - T)) / T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5file")
    ap.add_argument("pad_file", nargs="?", default="my_pad.txt")
    ap.add_argument("--old", default=None, help="sortie de l'ancienne version (même entrée)")
    ap.add_argument("--n", type=int, default=200, help="nombre d'événements examinés")
    a = ap.parse_args()

    f = h5py.File(a.h5file, "r")
    m = dict(f["/metadata"].attrs)
    for k in ("seuil_qt_pos", "seuil_qt_neg"):
        if k in m and abs(m[k]) > 0.0:
            print(f"ATTENTION : {k} = {m[k]} dans ce fichier. Les contrôles A, B, C supposent "
                  "des seuils à 0 : relance `./process N 0 0`.")
            return 1
    if "/H/pQ" not in f:
        print("Pas de dataset pQ : fichier produit par l'ancienne version ?")
        return 1
    dkg, dga, W, Fano = m["dkg"], m["dga"], m["W"], m["Fano"]
    DTk, DTg, DLk, DLg = m["DT_kg"], m["DT_ga"], m["DL_kg"], m["DL_ga"]
    vk, vg, dt = m["ve_kg"], m["ve_ga"], m["delta_t"]
    nb = int(m["n_bins"]); r_act = m["r_active"]
    half = 0.5 * (m["pad_size"] + m["inter_pad"])
    T_gap = dga / vg

    geom = np.loadtxt(a.pad_file)
    sq = geom[geom[:, 4] == 0.0]                    # pads carrés (r_out = 0)
    sq_id, sq_x, sq_y = sq[:, 0].astype(int), sq[:, 1], sq[:, 2]
    index_of = {int(p): k for k, p in enumerate(sq_id)}

    n_ev = min(a.n, f["/H/dE"].shape[0])
    t_start = np.arange(nb) * dt
    res_A, res_B, rel_big, tail_all = [], [], [], []
    dev_all, sd_all, ntot_all = [], [], []
    n_used = 0

    for i in range(n_ev):
        for frag in ("H", "L"):
            x = np.asarray(f[f"/{frag}/x"][i]); y = np.asarray(f[f"/{frag}/y"][i])
            z = np.asarray(f[f"/{frag}/z"][i]); dE = np.asarray(f[f"/{frag}/dE"][i])
            if len(x) == 0:
                continue
            if np.max(np.hypot(x, y)) > r_act - 1.0:
                continue                             # trace trop près du bord actif
            n_used += 1
            ne = dE / W
            d = dkg - z
            tg = d / vk
            sig_t = np.sqrt((DLk * np.sqrt(d) / vk) ** 2 + (DLg * np.sqrt(dga) / vg) ** 2)
            sig_a = np.sqrt(DTk ** 2 * d + DTg ** 2 * dga)
            pads = unpack(f, frag, i)
            Ntot = ne.sum()

            # --- A : conservation ---
            res_A.append(sum(p["q"] for p in pads) / Ntot)

            # --- B : somme des Q(t) sur les pads, aux fins de bins ---
            obs = np.zeros(nb)
            for p in pads:
                if len(p["t"]) == 0:
                    continue
                idx = np.searchsorted(p["t"], t_start, side="right") - 1
                obs += np.where(idx >= 0, p["Q"][np.clip(idx, 0, None)], 0.0)
            tau = (t_start[None, :] + dt) - tg[:, None]                # voxels x bins
            exp_ = (ne[:, None] * ramp(tau, sig_t[:, None], T_gap)).sum(axis=0)
            res_B.append(np.max(np.abs(obs - exp_)) / Ntot)

            # --- C : charge finale des pads carrés ---
            u_hi = (sq_x[None, :] + half - x[:, None]) / sig_a[:, None]
            u_lo = (sq_x[None, :] - half - x[:, None]) / sig_a[:, None]
            v_hi = (sq_y[None, :] + half - y[:, None]) / sig_a[:, None]
            v_lo = (sq_y[None, :] - half - y[:, None]) / sig_a[:, None]
            fxy = (ndtr(u_hi) - ndtr(u_lo)) * (ndtr(v_hi) - ndtr(v_lo))   # voxels x pads
            Fv = np.where(ne > 30.0, Fano, 1.0)
            E = (ne[:, None] * fxy).sum(axis=0)
            var = (Fv[:, None] * ne[:, None] * fxy ** 2).sum(axis=0)
            q_obs = np.zeros(len(sq_id))
            for p in pads:
                k = index_of.get(int(p["pid"]))
                if k is not None:
                    q_obs[k] = p["q"]
            sel = E > 1000.0
            dev_all.extend((q_obs - E)[sel].tolist())
            sd_all.extend(np.sqrt(var[sel]).tolist())
            ntot_all.extend([Ntot] * int(sel.sum()))
            big = E > 5000.0
            rel_big.extend(((q_obs - E)[big] / E[big]).tolist())
            tail = (E > 200.0) & (E <= 1000.0)          # queues : plancher de précision de la table
            tail_all.extend(((q_obs - E)[tail] / Ntot).tolist())

    print(f"{n_used} fragments utilisés (traces à plus de 1 cm du bord actif)\n")
    ok = True

    A = np.array(res_A)
    print("A. Conservation  Σpq / Σélectrons")
    print(f"   moyenne = {A.mean():.5f}   écart-type = {A.std():.5f}   min = {A.min():.5f}   max = {A.max():.5f}")
    okA = abs(A.mean() - 1) < 1e-3 and A.min() > 0.995 and A.max() < 1.005
    print("   ->", "OK" if okA else "A REGARDER (attendu 1 ± 1e-3)"); ok &= okA

    B = np.array(res_B)
    print("\nB. Courant total  max_t |Σ_pads Q(t) − attendu| / Σélectrons")
    print(f"   médiane = {np.median(B):.2e}   max = {B.max():.2e}")
    okB = B.max() < 5e-3
    print("   ->", "OK" if okB else "A REGARDER (attendu < 5e-3 ; un décalage en temps de 10 ns donnerait ~2e-2)"); ok &= okB

    dev = np.array(dev_all); sd = np.array(sd_all); nt = np.array(ntot_all)
    R = np.array(rel_big); Tl = np.array(tail_all)
    ratio = np.sqrt((dev ** 2).mean() / (sd ** 2).mean())
    exces = np.sqrt(max((dev ** 2).mean() - (sd ** 2).mean(), 0.0) / (nt ** 2).mean())
    print("\nC. Charge finale des pads carrés, comparée à Σ N·fx·fy (σ_anode)")
    print(f"   {len(dev)} pads (>1000 e⁻ attendus)")
    print(f"   RMS(écart) / RMS(fluctuation statistique Fano) = {ratio:.3f}   (attendu ~1)")
    print(f"   excès systématique (en quadrature) = {exces:.1e} de la charge de l'événement   (table : ~1e-5)")
    print(f"   pads > 5000 e⁻ : écart relatif moyen = {R.mean():+.2e}   RMS = {R.std():.2e}")
    print(f"   queues (200-1000 e⁻, {len(Tl)} pads) : écart RMS = {np.sqrt((Tl**2).mean()):.1e} de la charge de l'événement")
    okC = ratio < 1.3 and exces < 5e-5 and abs(R.mean()) < 2e-3
    print("   ->", "OK" if okC else "A REGARDER"); ok &= okC

    if a.old:
        g = h5py.File(a.old, "r")
        same = all(np.array_equal(np.asarray(f["/H/z"][i]), np.asarray(g["/H/z"][i])) for i in range(min(n_ev, 20)))
        print("\nF. Comparaison avec l'ancienne version, événement par événement")
        print("   traces identiques (mêmes voxels) :", "oui" if same else "NON : comparaison impossible")
        if same:
            CUT = 200.0
            num = den = 0.0
            n_frag = n_common = n_old_only = n_new_only = n_induced = 0
            rel = []
            q_new_only = []
            for i in range(n_ev):
                for frag in ("H", "L"):
                    pads = unpack(f, frag, i)
                    new = {int(p["pid"]): p["q"] for p in pads}
                    pid = np.asarray(g[f"/{frag}/pid"][i], dtype=int); pq = np.asarray(g[f"/{frag}/pq"][i])
                    old = dict(zip(pid.tolist(), pq.tolist()))
                    n_frag += 1
                    for k in set(new) | set(old):
                        num += abs(new.get(k, 0.0) - old.get(k, 0.0)); den += old.get(k, 0.0)
                    set_new = {k for k, v in new.items() if v > CUT}
                    set_old = {k for k, v in old.items() if v > CUT}
                    n_common += len(set_new & set_old)
                    n_old_only += len(set_old - set_new)
                    n_new_only += len(set_new - set_old)
                    q_new_only.extend(new[k] for k in (set_new - set_old))
                    for k in set_new & set_old:
                        if old[k] > 5000.0:
                            rel.append((new[k] - old[k]) / old[k])
                    # pads purement induits : lobe négatif profond mais charge finale faible
                    for p in pads:
                        if len(p["qt"]) and p["qt"].min() < -CUT and p["q"] < CUT:
                            n_induced += 1
            print(f"   Σ|nouveau − ancien| / Σ ancien = {num/den:.3f}")
            print(f"   pads avec charge finale > {CUT:.0f} e⁻, moyenne par fragment :")
            print(f"       communs = {n_common/n_frag:.1f}   seulement ancien = {n_old_only/n_frag:.2f}   seulement nouveau = {n_new_only/n_frag:.2f}")
            if q_new_only:
                print(f"       charge des pads seulement-nouveau : médiane = {np.median(q_new_only):.0f} e⁻   max = {np.max(q_new_only):.0f} e⁻")
            if rel:
                rel = np.array(rel)
                print(f"   pads communs > 5000 e⁻ : écart relatif moyen = {rel.mean():+.3f}   RMS = {rel.std():.3f}   ({len(rel)} pads)")
            print(f"   pads purement induits (lobe < −{CUT:.0f} e⁻, charge finale < {CUT:.0f} e⁻) : {n_induced/n_frag:.1f} par fragment")

    print("\n" + ("TOUS LES CONTROLES SONT OK" if ok else "AU MOINS UN CONTROLE A REGARDER"))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
