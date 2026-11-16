#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stats_pads2.py : version 2 des statistiques en LECTURE SEULE sur la sortie de ./process.

Usage :
    python3 stats_pads2.py fichier.h5 [n_events] [prefixe_sortie] [ENC_electrons]

    fichier.h5      sortie de ./process (groupes /H, /L, /metadata)
    n_events        nombre d'evenements lus depuis le debut (defaut 5000)
    prefixe_sortie  prefixe des PNG produits (defaut "stats_pads2")
    ENC_electrons   bruit rms suppose pour calculer ENC/charge (defaut 2000)

Par rapport a la version 1 :
  - tous les controles et statistiques de la version 1 sont conserves
  - (a) pour les pads dont |pq| < 200 e- : amplitude maximale de |pqt|, rapport
        |pq| / pic (petit = transitoire bipolaire de type induit ; proche de 1 =
        signal unipolaire de type queue de diffusion)
  - (b) charge par pad en fonction de la distance au vertex, separement pour les
        pads touches seulement par H, seulement par L, et par les deux
        (la somme H+L est portee par les pads partages)
  - controle du repere : px, py doivent etre dans les memes unites que vx, vy

Rien n'est ecrit dans le fichier HDF5.
"""

import sys
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEUILS = [200, 1000, 2000, 5000, 8000, 10000, 20000]   # electrons
SEUIL_REF = 8000.0        # seuil de reference pour "pad touche" (e-)
Q_MIN_MONTEE = 2.0e4      # on ne mesure la montee que sur les pads au-dessus (e-)
TOL_PQ_PQT = 0.02         # tolerance relative sur sum(pqt du pad) = pq
Q_QUASI_NUL = 200.0       # |pq| sous lequel un pad est dit "a charge nette quasi nulle"
BORNES_R = np.arange(0.0, 12.5, 1.0)    # bords des tranches de distance au vertex (cm)


def lire_attribut(f, nom):
    """Lit un attribut de /metadata ; arrete le script s'il n'existe pas."""
    meta = f["/metadata"]
    if nom not in meta.attrs:
        print("ERREUR : attribut /metadata/%s introuvable" % nom)
        sys.exit(1)
    return float(meta.attrs[nom])


def decouper(pnt, pqt):
    """Decoupe pqt en un tableau par pad, d'apres pnt.
    Renvoie None si sum(pnt) != len(pqt) (structure inattendue)."""
    if int(pnt.sum()) != len(pqt):
        return None
    fins = np.cumsum(pnt)
    debuts = fins - pnt
    segments = []
    for j in range(len(pnt)):
        segments.append(pqt[debuts[j]:fins[j]])
    return segments


def lire_groupe(f, g, n):
    """Lit les n premiers evenements des datasets utiles d'un groupe."""
    d = {}
    for nom in ("pid", "pk0", "pnt", "pq", "pqt", "px", "py"):
        d[nom] = f[g][nom][0:n]
    for nom in ("KE", "vx", "vy"):
        d[nom] = f[g][nom][0:n]
    return d


def percentiles(valeurs, liste):
    """Texte des percentiles demandes pour un tableau 1D."""
    morceaux = []
    for p in liste:
        morceaux.append("%d%% : %.4g" % (p, np.percentile(valeurs, p)))
    return " ; ".join(morceaux)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    chemin = sys.argv[1]
    n_demande = 5000
    if len(sys.argv) >= 3:
        n_demande = int(sys.argv[2])
    prefixe = "stats_pads2"
    if len(sys.argv) >= 4:
        prefixe = sys.argv[3]
    enc = 2000.0
    if len(sys.argv) >= 5:
        enc = float(sys.argv[4])

    f = h5py.File(chemin, "r")
    dt = lire_attribut(f, "delta_t")            # ns
    W = lire_attribut(f, "W")                   # MeV par paire
    pad = lire_attribut(f, "pad_size")          # cm
    n_tot = f["H/KE"].shape[0]
    n = min(n_demande, n_tot)

    print("Fichier           : %s" % chemin)
    print("Evenements lus    : %d sur %d" % (n, n_tot))
    print("delta_t = %g ns ; pad = %g cm ; W = %g eV ; ENC suppose = %g e- rms"
          % (dt, pad, W * 1.0e6, enc))

    donnees = {}
    donnees["H"] = lire_groupe(f, "H", n)
    donnees["L"] = lire_groupe(f, "L", n)
    f.close()

    # accumulateurs (version 1)
    tous_q = []
    n_pads_evt = []
    n_partages_evt = []
    n_neg_evt = []
    ecart_energie = []
    plus_grand_ecart_HL = 0.0
    evt_structure_ko = 0
    tous_T = []
    toutes_montees = []
    n_pq_pqt_ok = 0
    n_pq_pqt_tot = 0
    frac_evt = {}
    nb_sous_evt = {}
    q_sous_tot = {}
    for s in SEUILS:
        frac_evt[s] = []
        nb_sous_evt[s] = []
        q_sous_tot[s] = 0.0
    q_total_cumule = 0.0

    # accumulateurs (version 2)
    pics_nuls = []          # amplitude max de |pqt| des pads a |pq| < Q_QUASI_NUL
    ratios_nuls = []        # |pq| / pic pour ces pads
    n_nuls_sans_pqt = 0
    tous_r = []
    tous_q2 = []
    toutes_origines = []    # 0 = H seul, 1 = L seul, 2 = les deux
    dist_min_evt = []
    coord_ko = 0
    ecart_vertex_HL = 0.0
    px_min = 1.0e30
    px_max = -1.0e30
    py_min = 1.0e30
    py_max = -1.0e30
    vx_min = 1.0e30
    vx_max = -1.0e30
    vy_min = 1.0e30
    vy_max = -1.0e30
    n_evt_utiles = 0

    for i in range(n):
        liste_pid = []
        liste_q = []
        liste_pnt = []
        liste_seg = []
        segs_ok = True
        pid_par_frag = {}
        q_par_frag = {}
        for g in ("H", "L"):
            d = donnees[g]
            pid = np.rint(d["pid"][i]).astype(np.int64)
            q = d["pq"][i].astype(np.float64)
            pnt = np.rint(d["pnt"][i]).astype(np.int64)
            pid_par_frag[g] = pid
            q_par_frag[g] = q
            seg = decouper(pnt, d["pqt"][i])
            if seg is None:
                segs_ok = False
                seg = [None] * len(pid)
            liste_pid.append(pid)
            liste_q.append(q)
            liste_pnt.append(pnt)
            liste_seg.extend(seg)
        if not segs_ok:
            evt_structure_ko += 1

        pid_all = np.concatenate(liste_pid)
        q_all = np.concatenate(liste_q)
        pnt_all = np.concatenate(liste_pnt)
        if len(pid_all) == 0:
            continue
        n_evt_utiles += 1

        # fusion : une seule occurrence par pad
        pid_u, idx = np.unique(pid_all, return_index=True)
        q_u = q_all[idx]
        pnt_u = pnt_all[idx]
        n_partages_evt.append(len(pid_all) - len(idx))

        # pads partages : les valeurs de pq doivent etre identiques dans H et L
        communs, iH, iL = np.intersect1d(pid_par_frag["H"], pid_par_frag["L"],
                                         return_indices=True)
        if len(communs) > 0:
            qH = q_par_frag["H"][iH]
            qL = q_par_frag["L"][iL]
            ecart = np.abs(qH - qL) / np.maximum(np.abs(qH), 1.0)
            if ecart.max() > plus_grand_ecart_HL:
                plus_grand_ecart_HL = ecart.max()

        # origine de chaque pad distinct : H seul, L seul, ou les deux
        dans_H = np.isin(pid_u, pid_par_frag["H"])
        dans_L = np.isin(pid_u, pid_par_frag["L"])
        origine = np.zeros(len(pid_u), dtype=np.int64)
        origine[dans_L & (~dans_H)] = 1
        origine[dans_L & dans_H] = 2

        # conservation de l'energie (comme le titre du viewer)
        ke = float(donnees["H"]["KE"][i]) + float(donnees["L"]["KE"][i])
        e_pads = q_u.sum() * W
        ecart_energie.append(e_pads / ke - 1.0)

        n_pads_evt.append(len(pid_u))
        n_neg_evt.append(int((q_u <= 0).sum()))
        tous_q.append(q_u)
        tous_T.append(pnt_u * dt)

        # fractions de charge sous seuil (pads a charge positive seulement)
        q_pos = q_u[q_u > 0]
        total = q_pos.sum()
        q_total_cumule += total
        for s in SEUILS:
            masque = q_pos < s
            q_sous = q_pos[masque].sum()
            q_sous_tot[s] += q_sous
            if total > 0:
                frac_evt[s].append(q_sous / total)
            nb_sous_evt[s].append(int(masque.sum()))

        # distance au vertex (version 2)
        px_all = np.concatenate([donnees["H"]["px"][i], donnees["L"]["px"][i]]).astype(np.float64)
        py_all = np.concatenate([donnees["H"]["py"][i], donnees["L"]["py"][i]]).astype(np.float64)
        vxH = float(donnees["H"]["vx"][i])
        vyH = float(donnees["H"]["vy"][i])
        vxL = float(donnees["L"]["vx"][i])
        vyL = float(donnees["L"]["vy"][i])
        ev = max(abs(vxH - vxL), abs(vyH - vyL))
        if ev > ecart_vertex_HL:
            ecart_vertex_HL = ev
        vx_min = min(vx_min, vxH)
        vx_max = max(vx_max, vxH)
        vy_min = min(vy_min, vyH)
        vy_max = max(vy_max, vyH)
        if len(px_all) == len(pid_all) and len(py_all) == len(pid_all):
            px_u = px_all[idx]
            py_u = py_all[idx]
            px_min = min(px_min, px_u.min())
            px_max = max(px_max, px_u.max())
            py_min = min(py_min, py_u.min())
            py_max = max(py_max, py_u.max())
            r_u = np.sqrt((px_u - vxH) ** 2 + (py_u - vyH) ** 2)
            dist_min_evt.append(r_u.min())
            tous_r.append(r_u)
            tous_q2.append(q_u)
            toutes_origines.append(origine)
        else:
            coord_ko += 1

        # mesures qui demandent pqt (seulement si la structure est coherente)
        if segs_ok:
            for k in range(len(idx)):
                seg = liste_seg[idx[k]]
                if seg is None or len(seg) == 0:
                    continue
                qk = q_u[k]
                if abs(qk) < Q_QUASI_NUL:
                    pic = float(np.abs(seg).max())
                    if pic > 0:
                        pics_nuls.append(pic)
                        ratios_nuls.append(abs(qk) / pic)
                if qk > 1.0e4:
                    n_pq_pqt_tot += 1
                    if abs(seg.sum() - qk) <= TOL_PQ_PQT * qk:
                        n_pq_pqt_ok += 1
                if qk > Q_MIN_MONTEE:
                    cum = np.cumsum(seg.astype(np.float64))
                    tot = cum[-1]
                    if tot > 0:
                        i10 = int(np.argmax(cum >= 0.1 * tot))
                        i90 = int(np.argmax(cum >= 0.9 * tot))
                        toutes_montees.append((i90 - i10) * dt)
        else:
            for k in range(len(idx)):
                if abs(q_u[k]) < Q_QUASI_NUL:
                    n_nuls_sans_pqt += 1

    # ---------------------------------------------------------------- bilan v1
    q_cat = np.concatenate(tous_q)
    T_cat = np.concatenate(tous_T)
    q_pos_cat = q_cat[q_cat > 0]
    pads_evt = np.array(n_pads_evt)

    print("")
    print("=== Controles ===")
    ee = np.array(ecart_energie)
    print("Energie : (somme pq * W) / (KE_H + KE_L) - 1 :")
    print("   evenement 0 : %+.3e   (ton viewer donne -4.25e-05)" % ee[0])
    print("   moyenne %+.3e ; ecart-type %.3e ; min %+.3e ; max %+.3e"
          % (ee.mean(), ee.std(), ee.min(), ee.max()))
    print("Pads partages H/L : %.2f par evenement en moyenne" % np.mean(n_partages_evt))
    print("   plus grand ecart relatif de pq entre H et L sur un pad partage : %.2e"
          % plus_grand_ecart_HL)
    if evt_structure_ko > 0:
        print("ATTENTION : %d evenements ou sum(pnt) != len(pqt) ; "
              "mesures sur pqt ignorees pour eux" % evt_structure_ko)
    if n_pq_pqt_tot > 0:
        print("Coherence pq / somme de pqt (pads > 1e4 e-) : %.1f %% a moins de %.0f %%"
              % (100.0 * n_pq_pqt_ok / n_pq_pqt_tot, 100.0 * TOL_PQ_PQT))

    print("")
    print("=== Pads par evenement (pads distincts) ===")
    print("   moyenne %.1f ; mediane %.0f ; 5%%-95%% : %.0f - %.0f"
          % (pads_evt.mean(), np.median(pads_evt),
             np.percentile(pads_evt, 5), np.percentile(pads_evt, 95)))
    print("   pads a pq <= 0 par evenement : %.2f" % np.mean(n_neg_evt))

    print("")
    print("=== Charge par pad pq (e-), pads a pq > 0 ===")
    for p in (5, 25, 50, 75, 95):
        print("   percentile %2d %% : %10.0f" % (p, np.percentile(q_pos_cat, p)))

    print("")
    print("=== Duree de la fenetre ecrite pnt*dt (ns) ===")
    for p in (5, 50, 95):
        print("   percentile %2d %% : %6.0f" % (p, np.percentile(T_cat, p)))

    if len(toutes_montees) > 0:
        m = np.array(toutes_montees)
        print("")
        print("=== Temps de montee 10%%-90%% de la charge cumulee (ns), pads > %.0e e- ==="
              % Q_MIN_MONTEE)
        print("   nombre de pads : %d" % len(m))
        for p in (5, 25, 50, 75, 95):
            print("   percentile %2d %% : %6.0f" % (p, np.percentile(m, p)))

    print("")
    print("=== Charge portee par les pads sous un seuil (pads a pq > 0) ===")
    print("   seuil(e-)  charge sous seuil/totale   mediane/evt   90%/evt   pads sous seuil/evt")
    for s in SEUILS:
        fe = np.array(frac_evt[s])
        print("   %7d       %8.3f %%            %8.3f %%   %8.3f %%   %8.1f"
              % (s, 100.0 * q_sous_tot[s] / q_total_cumule,
                 100.0 * np.median(fe), 100.0 * np.percentile(fe, 90),
                 np.mean(nb_sous_evt[s])))

    # ---------------------------------------------------------------- bilan v2 (a)
    print("")
    print("=== (a) Pads a charge nette quasi nulle : |pq| < %.0f e- ===" % Q_QUASI_NUL)
    n_nuls_tot = int(np.sum(np.abs(q_cat) < Q_QUASI_NUL))
    print("   %d pads au total, soit %.1f par evenement (%.1f %% des pads)"
          % (n_nuls_tot, n_nuls_tot / float(n_evt_utiles), 100.0 * n_nuls_tot / len(q_cat)))
    if n_nuls_sans_pqt > 0:
        print("   dont %d sans pqt exploitable (evenements a structure incoherente)"
              % n_nuls_sans_pqt)
    if len(pics_nuls) > 0:
        pics = np.array(pics_nuls)
        rat = np.array(ratios_nuls)
        print("   amplitude maximale |pqt| par bin (e-) : " + percentiles(pics, (5, 25, 50, 75, 95)))
        print("   rapport |pq| / pic                     : " + percentiles(rat, (5, 25, 50, 75, 95)))
        print("   (rapport << 1 : transitoire bipolaire de type induit ;"
              " proche de 1 : signal unipolaire minuscule)")
        for s in (2000, 4000, 8000):
            print("   pads dont le pic depasse %5d e- : %6.2f %%" % (s, 100.0 * np.mean(pics > s)))
        print("   pads de rapport < 0.1 : %.1f %% ; de rapport > 0.5 : %.1f %%"
              % (100.0 * np.mean(rat < 0.1), 100.0 * np.mean(rat > 0.5)))
    else:
        print("   aucun pad exploitable")

    # ---------------------------------------------------------------- bilan v2 (b)
    print("")
    print("=== (b) Charge par pad en fonction de la distance au vertex ===")
    if len(tous_r) == 0:
        print("   pas de coordonnees exploitables (px, py de longueur differente de pid)")
        R = np.array([])
        Q = np.array([])
        O = np.array([])
    else:
        R = np.concatenate(tous_r)
        Q = np.concatenate(tous_q2)
        O = np.concatenate(toutes_origines)
        if coord_ko > 0:
            print("   ATTENTION : %d evenements sans coordonnees coherentes (ignores ici)" % coord_ko)
        print("   Controle du repere :")
        print("      px : de %.2f a %.2f ; py : de %.2f a %.2f" % (px_min, px_max, py_min, py_max))
        print("      vx : de %.2f a %.2f ; vy : de %.2f a %.2f" % (vx_min, vx_max, vy_min, vy_max))
        print("      ecart de vertex entre H et L (max) : %.2e" % ecart_vertex_HL)
        dm = np.array(dist_min_evt)
        print("      distance minimale pad-vertex par evenement : mediane %.3f ; 95%% : %.3f  (pad = %.2f)"
              % (np.median(dm), np.percentile(dm, 95), pad))
        print("      (si px, py et vx, vy sont dans les memes unites, ces distances sont de l'ordre"
              " d'un pad ou moins)")
        print("   Les distances sont des distances au vertex en plan (x, y), pas des longueurs de trace.")
        print("   Pads 'touches' = pq >= %.0f e-. Bruit relatif = ENC / mediane de pq (ENC = %g e-)."
              % (SEUIL_REF, enc))
        print("   Origine : H = pads vus seulement par H ; L = seulement par L.")
        print("")
        print("   r (cm)     pads/evt  touches/evt  %%pads<%.0f |  H : mediane pq   ENC/med |  L : mediane pq   ENC/med"
              % SEUIL_REF)
        for b in range(len(BORNES_R) - 1):
            m = (R >= BORNES_R[b]) & (R < BORNES_R[b + 1])
            n_tranche = int(m.sum())
            if n_tranche == 0:
                continue
            n_all = n_tranche / float(n_evt_utiles)
            m_hit = m & (Q >= SEUIL_REF)
            n_hit = int(m_hit.sum()) / float(n_evt_utiles)
            pct_sous = 100.0 * float(np.sum(m & (Q < SEUIL_REF))) / n_tranche
            texte = []
            for orig in (0, 1):
                mo = m_hit & (O == orig)
                if int(mo.sum()) > 0:
                    med = float(np.median(Q[mo]))
                    texte.append("%12.0f   %6.2f %%" % (med, 100.0 * enc / med))
                else:
                    texte.append("%12s   %8s" % ("-", "-"))
            print("   %4.1f-%4.1f  %8.2f  %10.2f  %8.1f |  %s |  %s"
                  % (BORNES_R[b], BORNES_R[b + 1], n_all, n_hit, pct_sous, texte[0], texte[1]))

    # ---------------------------------------------------------------- figure 1 (v1)
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    bornes = np.logspace(2, 6.3, 70)
    ax[0, 0].hist(q_pos_cat, bins=bornes, color="steelblue")
    ax[0, 0].set_xscale("log")
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_xlabel("pq par pad (e-)")
    ax[0, 0].set_ylabel("pads")
    ax[0, 0].set_title("Charge par pad")
    for s in (2000, 8000):
        ax[0, 0].axvline(s, color="red", linestyle="--", linewidth=1)

    ax[0, 1].hist(pads_evt, bins=40, color="steelblue")
    ax[0, 1].set_xlabel("pads distincts par evenement")
    ax[0, 1].set_ylabel("evenements")
    ax[0, 1].set_title("Pads touches")

    if len(toutes_montees) > 0:
        ax[1, 0].hist(toutes_montees, bins=40, color="steelblue")
        ax[1, 0].set_title("Montee 10-90 %% (pads > %.0e e-)" % Q_MIN_MONTEE)
    else:
        ax[1, 0].set_title("Montee : pas de donnees")
    ax[1, 0].set_xlabel("ns")
    ax[1, 0].set_ylabel("pads")

    f8 = np.array(frac_evt[8000]) * 100.0
    ax[1, 1].hist(f8, bins=40, color="steelblue")
    ax[1, 1].set_xlabel("% de la charge de l'evenement portee par des pads < 8000 e-")
    ax[1, 1].set_ylabel("evenements")
    ax[1, 1].set_title("Charge sous le seuil de 8000 e-")

    fig.tight_layout()
    nom_png = prefixe + ".png"
    fig.savefig(nom_png, dpi=120)
    print("")
    print("Figure 1 ecrite : %s" % nom_png)

    # ---------------------------------------------------------------- figure 2 (v2)
    fig2, ax2 = plt.subplots(1, 3, figsize=(17, 5.5))

    if len(R) > 0:
        centres = 0.5 * (BORNES_R[:-1] + BORNES_R[1:])
        couleurs = ["tab:red", "tab:blue"]
        noms = ["H seul", "L seul"]
        for orig in (0, 1):
            xs = []
            med = []
            bas = []
            haut = []
            for b in range(len(BORNES_R) - 1):
                mo = (R >= BORNES_R[b]) & (R < BORNES_R[b + 1]) & (Q >= SEUIL_REF) & (O == orig)
                if int(mo.sum()) >= 5:
                    xs.append(centres[b])
                    med.append(np.median(Q[mo]))
                    bas.append(np.percentile(Q[mo], 10))
                    haut.append(np.percentile(Q[mo], 90))
            if len(xs) > 0:
                ax2[0].plot(xs, med, "o-", color=couleurs[orig], label=noms[orig] + " (mediane)")
                ax2[0].fill_between(xs, bas, haut, color=couleurs[orig], alpha=0.2)
        ax2[0].axhline(enc, color="black", linestyle=":", label="ENC = %g e-" % enc)
        ax2[0].axhline(4.0 * enc, color="black", linestyle="--", label="4 x ENC")
        ax2[0].set_yscale("log")
        ax2[0].set_xlabel("distance au vertex (cm)")
        ax2[0].set_ylabel("pq des pads touches (e-)")
        ax2[0].set_title("Charge par pad touche (10-90 %)")
        ax2[0].legend()

        frac = []
        xs2 = []
        for b in range(len(BORNES_R) - 1):
            m = (R >= BORNES_R[b]) & (R < BORNES_R[b + 1])
            if int(m.sum()) >= 20:
                xs2.append(centres[b])
                frac.append(100.0 * float(np.sum(m & (Q < SEUIL_REF))) / float(m.sum()))
        ax2[1].plot(xs2, frac, "o-", color="tab:green")
        ax2[1].set_xlabel("distance au vertex (cm)")
        ax2[1].set_ylabel("%% des pads de la tranche avec pq < %.0f e-" % SEUIL_REF)
        ax2[1].set_title("Pads faibles selon la distance")
    else:
        ax2[0].set_title("pas de coordonnees")
        ax2[1].set_title("pas de coordonnees")

    if len(pics_nuls) > 0:
        ax2[2].hist(pics_nuls, bins=np.logspace(0, 5, 60), color="steelblue")
        ax2[2].set_xscale("log")
        ax2[2].set_yscale("log")
        ax2[2].set_xlabel("pic de |pqt| (e- par bin), pads a |pq| < %.0f e-" % Q_QUASI_NUL)
        ax2[2].set_ylabel("pads")
        ax2[2].set_title("Amplitude des pads a charge nette quasi nulle")
        ax2[2].axvline(enc, color="black", linestyle=":")
        ax2[2].axvline(4.0 * enc, color="black", linestyle="--")
    else:
        ax2[2].set_title("pas de pads quasi nuls")

    fig2.tight_layout()
    nom_png2 = prefixe + "_v2.png"
    fig2.savefig(nom_png2, dpi=120)
    print("Figure 2 ecrite : %s" % nom_png2)


if __name__ == "__main__":
    main()
