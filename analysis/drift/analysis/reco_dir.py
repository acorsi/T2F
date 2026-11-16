#!/usr/bin/env python3
"""
reco_dir.py : ETAPE 0 de la reconstruction par profil dE/dx(l) : estimer la DIRECTION de la trace (et un vertex
approximatif) a partir des seuls pads forts d'un evenement, puis comparer a la verite de la simulation.

Usage :
    python reco_dir.py sortie.h5 [--start 0] [--n 200] [--k_thr 10] [--f_max 0.1] [--min_range 2.0]
                       [--sign_z 1] [--y_plane 0] [--t_off NS] [--out reco_dir] [--show]

Fichiers necessaires dans le meme dossier : t2f_io.py, view_fit.py (et view_pads.py, que view_fit importe).
Sorties : tableau resume a l'ecran, <out>.csv (une ligne par evenement), <out>.png (4 figures).

Methode (rien de la verite n'est utilise pour l'estimation) :
  1. pads forts : pads non fantomes dont l'amplitude (maximum des echantillons) vaut au moins
     max(k_thr * seuil de lecture, f_max * amplitude maximale de l'evenement), et dont la somme des echantillons
     est du signe d'une charge collectee (somme dt / aire(h) > 0.5 * amplitude : ecarte les pads d'induction bipolaires) ;
  2. axe dans le plan (x, y) : axe principal des centres des pads forts, ponderes par leur amplitude -> azimut alpha dans [0, pi) ;
  3. pente en z : temps d'arrivee de chaque pad fort = barycentre temporel des echantillons ecrits moins le barycentre
     de h (le barycentre d'une convolution est la somme des barycentres) ; regression ponderee (amplitude) du temps
     d'arrivee en fonction de l'abscisse s du pad le long de l'axe -> dt/ds ; pente dz/ds = - sign_z * ve_kg * dt/ds.
     Convention : z croit vers la grille (cathode a z = 0, grille a z = dkg), donc les electrons de z plus grand
     arrivent plus tot. Si le resume montre un rapport m_est / m_vrai proche de -1, relancer avec --sign_z -1.
  4. vertex : le vertex est dans le plan y = y_plane (plan de symetrie de l'anode, perpendiculaire aux pads ; y_plane = 0).
     Dans le plan des pads, trois candidats : pad d'amplitude maximale ; barycentre des pads forts ; INTERSECTION de
     l'axe estime avec la droite y = y_plane (pas besoin du temps). Son erreur varie comme 1 / |e_y|, e_y = composante
     de l'axe perpendiculaire au plan (indeterminee pour une trace parallele au plan : |e_y| < 0.05 -> pas d'estimation).
     z du vertex : on prolonge la regression du temps d'arrivee jusqu'a l'abscisse de l'intersection ; le temps d'arrivee
     du barycentre d'un pad de cote z vaut (dkg - z)/ve_kg + t_off, t_off = (dga/ve_ga)/2 - delta_t/2 (milieu du profil
     d'arrivee dans le gap, charge placee au debut du bin) ; --t_off pour le changer. Suppose que le temps zero est celui
     de la fission. Le resume donne le t_off MESURE sur la simulation (verite de z des pads) : il doit etre proche de la
     valeur utilisee, avec une faible dispersion d'un evenement a l'autre.

Comparaison a la verite (fragment H, axe sans orientation) :
  - erreur d'azimut (degres), erreur d'angle 3D entre la direction estimee (cos alpha, sin alpha, m) et la vraie ;
  - pente estimee contre vraie ;
  - ecart du vertex estime au vertex vrai dans le plan, le long de l'axe estime (ds) et perpendiculairement (d_perp), en cm ;
    erreur sur z du vertex (dz, cm).
Les evenements sont aussi classes par longueur de trace projetee dans le plan (Lxy).

Limites : pas de verification que H et L sont colineaires (angle entre u_H et -u_L donne dans le resume) ;
les pads ne servent qu'a leur centre (precision limitee par le pas de pad) ; les pads forts proches du vertex
melangent H et L (leur barycentre temporel est une moyenne).
"""

import argparse
import sys

import numpy as np
import h5py
import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from t2f_io import pad_geometry, read_vertex, event_pads
from view_fit import Response


def true_direction(f, frag, i, axis):
    """vecteur unitaire (ux, uy, uz) du fragment, selon trace_xyz_description"""
    th = float(f["/%s/theta" % frag][i])
    ph = float(f["/%s/phi" % frag][i])
    ux = np.sin(th) * np.cos(ph)
    uy = np.cos(th) if axis == "Y" else np.sin(th) * np.sin(ph)
    uz = np.cos(th) if axis == "Z" else np.sin(th) * np.sin(ph)
    return np.array([ux, uy, uz])


def pad_waveforms(ev, phase, R, dt):
    """liste de (t, v) par pad ecrit ; t = (phase + j adc_ratio) dt avec j = pk0 + indice dans la fenetre ecrite"""
    out = []
    off = 0
    for k in range(len(ev["pid"])):
        n = int(ev["pnt"][k])
        v = ev["pqn"][off:off + n]
        off += n
        j = int(ev["pk0"][k]) + np.arange(n)
        out.append(((phase + j * R) * dt, v))
    return out


def estimate(wfs, flag, x, y, area_h, m_h, thr, a, pitch, ve, dkg, t_off):
    """Estimation de la direction. Renvoie (dict, None) ou (None, raison)."""
    n = len(wfs)
    if n == 0:
        return None, "aucun pad"
    amp = np.array([w[1].max() for w in wfs])
    qint = np.array([w[1].sum() * (w[0][1] - w[0][0]) / area_h if len(w[1]) > 1 else 0.0 for w in wfs])
    amin = max(a.k_thr * thr, a.f_max * amp.max())
    ok = (flag != 0) & (amp >= amin) & (qint > 0.5 * amp)
    idx = np.where(ok)[0]
    if len(idx) < 3:
        return None, "moins de 3 pads forts"
    w = amp[idx]
    X = np.vstack([x[idx], y[idx]]).T
    mean = (w[:, None] * X).sum(axis=0) / w.sum()
    D = X - mean
    C = (w[:, None, None] * D[:, :, None] * D[:, None, :]).sum(axis=0) / w.sum()
    lam, vec = np.linalg.eigh(C)
    alpha = np.arctan2(vec[1, 1], vec[0, 1]) % np.pi
    e = np.array([np.cos(alpha), np.sin(alpha)])
    s = D @ e
    srange = float(s.max() - s.min())
    res = {"n_strong": len(idx), "alpha": alpha, "e": e, "mean": mean, "srange": srange,
           "elong": float(np.sqrt(lam[1] / max(lam[0], 1.0e-12))),
           "P_max": X[int(np.argmax(w))], "m": np.nan, "P_int": np.array([np.nan, np.nan]), "z_v": np.nan,
           "idx": idx, "w": w, "X": X}
    # temps d'arrivee des pads forts (barycentre temporel - barycentre de h)
    tc = np.array([(wfs[k][0] * wfs[k][1]).sum() / wfs[k][1].sum() for k in idx]) - m_h
    if srange >= a.min_range * pitch:
        S = w.sum(); Sx = (w * s).sum(); Sy = (w * tc).sum()
        Sxx = (w * s * s).sum(); Sxy = (w * s * tc).sum()
        den = S * Sxx - Sx * Sx
        if den > 0.0:
            b = (S * Sxy - Sx * Sy) / den              # dt/ds, ns/cm
            res["m"] = -a.sign_z * ve * b
            a0 = (Sxx * Sy - Sx * Sxy) / den              # ordonnee a l'origine (s = 0 : barycentre des pads forts)
            res["a0"] = a0
            res["b"] = b
    res["tc"] = tc
    # intersection de l'axe avec le plan y = y_plane, et z du vertex par prolongement de la regression
    if abs(e[1]) >= 0.05:
        sv = (a.y_plane - mean[1]) / e[1]
        res["P_int"] = mean + sv * e
        if "b" in res:
            res["z_v"] = dkg - ve * (res["a0"] + res["b"] * sv - t_off)
    res["e_y"] = abs(float(e[1]))
    return res, None


def pct(v):
    """mediane, 68 %, 95 % de |v| (nan ignores)"""
    v = np.abs(np.asarray(v, dtype=float))
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    return tuple(np.percentile(v, [50, 68, 95]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("h5file")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n", type=int, default=200, help="nombre d'evenements traites")
    p.add_argument("--k_thr", type=float, default=10.0, help="pad fort : amplitude >= k_thr x seuil de lecture")
    p.add_argument("--f_max", type=float, default=0.1, help="pad fort : amplitude >= f_max x amplitude maximale")
    p.add_argument("--min_range", type=float, default=2.0, help="etendue minimale des pads forts (en pas de pad) pour la pente")
    p.add_argument("--sign_z", type=float, default=1.0, help="+1 : z croit vers la grille ; -1 pour inverser")
    p.add_argument("--y_plane", type=float, default=0.0, help="y du plan du vertex (cm) ; 0 : plan de symetrie de l'anode")
    p.add_argument("--t_off", type=float, default=None, help="decalage temporel (ns) ; defaut : (dga/ve_ga)/2 - delta_t/2")
    p.add_argument("--out", type=str, default="reco_dir")
    p.add_argument("--show", action="store_true")
    a = p.parse_args()

    f = h5py.File(a.h5file, "r")
    at = f["/metadata"].attrs
    dt = float(at["delta_t"])
    R = int(at["adc_ratio"])
    ve = float(at["ve_kg"])
    thr = float(at["read_thr_e"]) if int(at["noise_enable"]) == 1 else 0.0
    pitch = float(at["pad_size"]) + float(at["inter_pad"])
    axis = at["emission_axis"]
    axis = axis.decode() if isinstance(axis, bytes) else str(axis)
    dkg = float(at["dkg"])
    t_off = (0.5 * float(at["dga"]) / float(at["ve_ga"]) - 0.5 * dt) if a.t_off is None else a.t_off
    resp = Response(at, dt)
    hf = resp.hcont(resp.tf)
    m_h = float((resp.tf * hf).sum() / hf.sum())          # barycentre temporel de h (ns)
    geom = pad_geometry(f)
    N = f["/event_id"].shape[0]
    i1 = min(N, a.start + a.n)
    print("[reco_dir] %s : evenements %d a %d | reponse : %s" % (a.h5file, a.start, i1 - 1, resp.descr))
    print("[reco_dir] ve_kg = %.4f cm/ns, seuil de lecture = %.0f e-, pas de pad = %.3f cm, barycentre de h = %.1f ns"
          % (ve, thr, pitch, m_h))
    print("[reco_dir] vertex : dans le plan y = %.2f cm ; t_off = %.1f ns (pour z du vertex)" % (a.y_plane, t_off))

    rows = []
    skipped = {}
    vy_abs = []
    vz_all = []
    for i in range(a.start, i1):
        ev = event_pads(f, i, geom)
        phase = int(f["/P/phase"][i])
        wfs = pad_waveforms(ev, phase, R, dt)
        res, why = estimate(wfs, ev["pflag"], ev["x"], ev["y"], resp.area, m_h, thr, a, pitch, ve, dkg, t_off)
        if res is None:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        # verite
        uH = true_direction(f, "H", i, axis)
        uL = true_direction(f, "L", i, axis)
        vx, vy, vz = read_vertex(f, i)
        vy_abs.append(abs(vy))
        vz_all.append(vz)
        uxy = np.hypot(uH[0], uH[1])
        alpha_t = np.arctan2(uH[1], uH[0]) % np.pi
        e_t = np.array([np.cos(alpha_t), np.sin(alpha_t)])
        sgn = 1.0 if (uH[0] * e_t[0] + uH[1] * e_t[1]) >= 0.0 else -1.0
        m_true = sgn * uH[2] / max(uxy, 1.0e-9)
        dal = (res["alpha"] - alpha_t + np.pi / 2.0) % np.pi - np.pi / 2.0
        flip = 1.0 if np.cos(res["alpha"] - alpha_t) >= 0.0 else -1.0
        m_est = res["m"] * flip if np.isfinite(res["m"]) else np.nan
        ang3d = np.nan
        if np.isfinite(res["m"]):
            ve3 = np.array([res["e"][0], res["e"][1], res["m"]])
            ve3 = ve3 / np.linalg.norm(ve3)
            ang3d = np.degrees(np.arccos(min(abs(float(ve3 @ uH)), 1.0)))
        else:
            ve3 = None
        acol = np.degrees(np.arccos(max(min(-float(uH @ uL), 1.0), -1.0)))
        nH = len(f["/H/dE"][i]); nL = len(f["/L/dE"][i])
        step = float(at["step_size_eff"])
        lxy = (nH + nL) * step * uxy
        vtx = np.array([vx, vy])
        def comp(P):
            d = P - vtx
            return abs(float(d @ res["e"])), abs(float(d[0] * res["e"][1] - d[1] * res["e"][0]))
        ds_max, dp_max = comp(res["P_max"])
        ds_cen, dp_cen = comp(res["mean"])
        if np.isfinite(res["P_int"][0]):
            ds_int, dp_int = comp(res["P_int"])
        else:
            ds_int = dp_int = np.nan
        dz_v = (res["z_v"] - vz) if np.isfinite(res["z_v"]) else np.nan
        # t_off mesure : barycentre temporel des pads forts - temps de derive vrai (z vrai au centre du pad)
        lam = ((res["X"][:, 0] - vx) * uH[0] + (res["X"][:, 1] - vy) * uH[1]) / max(uxy * uxy, 1.0e-9)
        z_i = vz + lam * uH[2]
        toff_ev = float((res["w"] * (res["tc"] - (dkg - z_i) / ve)).sum() / res["w"].sum())
        rows.append((i, len(ev["pid"]), res["n_strong"], np.degrees(abs(dal)), ang3d, m_true, m_est,
                     ds_max, dp_max, ds_cen, dp_cen, lxy, res["srange"], res["elong"], acol, ds_int, dp_int, toff_ev, abs(m_true), res["e_y"], dz_v))

    rows = np.array(rows, dtype=float)
    cols = ["event", "n_pads", "n_strong", "alpha_err_deg", "ang3d_deg", "m_true", "m_est", "ds_pmax", "dperp_pmax",
            "ds_cent", "dperp_cent", "Lxy_cm", "srange_cm", "elong", "angle_HL_deg", "ds_int", "dperp_int", "toff_meas", "abs_m_true", "abs_ey", "dz_vertex"]
    if len(rows):
        np.savetxt(a.out + ".csv", rows, delimiter=",", header=",".join(cols), comments="")

    ntot = i1 - a.start
    print("\n=== Resume : %d evenements, %d evalues, ignores : %s ===" % (ntot, len(rows), skipped if skipped else "aucun"))
    if len(rows) == 0:
        sys.exit(0)
    c = dict((k, rows[:, j]) for j, k in enumerate(cols))
    print("verite : |vy| median = %.3g cm (si non nul, trace_xyz n'ajoute pas vy : a regarder) ; angle H / -L : median %.2f deg, 95 %% %.2f deg"
          % (np.median(vy_abs), np.median(c["angle_HL_deg"]), np.percentile(c["angle_HL_deg"], 95)))
    print("                          mediane     68 %      95 %")
    for name, key, unit in (("erreur azimut", "alpha_err_deg", "deg"), ("erreur angle 3D", "ang3d_deg", "deg"),
                            ("vertex (pad max) ds", "ds_pmax", "cm"), ("vertex (pad max) dperp", "dperp_pmax", "cm"),
                            ("vertex (barycentre) ds", "ds_cent", "cm"), ("vertex (barycentre) dperp", "dperp_cent", "cm"),
                            ("vertex (plan y) ds", "ds_int", "cm"), ("vertex (plan y) dperp", "dperp_int", "cm"),
                            ("vertex z : |dz|", "dz_vertex", "cm")):
        m, p68, p95 = pct(c[key])
        print("  %-26s %8.3f %8.3f %8.3f  %s" % (name, m, p68, p95, unit))
    vz_all = np.array(vz_all)
    print("verite : vertex z : min %.3f  mediane %.3f  max %.3f cm" % (vz_all.min(), np.median(vz_all), vz_all.max()))
    print("t_off mesure (evenement par evenement) : mediane %.1f ns, 16-84 %% [%.1f ; %.1f] ns ; utilise : %.1f ns"
          % (np.median(c["toff_meas"]), np.percentile(c["toff_meas"], 16), np.percentile(c["toff_meas"], 84), t_off))
    dm = c["m_est"] - c["m_true"]
    m, p68, p95 = pct(dm)
    print("  %-26s %8.3f %8.3f %8.3f" % ("|m_est - m_vrai|", m, p68, p95))
    big = np.isfinite(c["m_est"]) & (np.abs(c["m_true"]) > 0.3)
    if big.any():
        print("verification du signe : mediane de m_est / m_vrai (|m_vrai| > 0.3, %d evenements) = %.2f (attendu : +1)"
              % (big.sum(), np.median(c["m_est"][big] / c["m_true"][big])))
    print("\nPar longueur de trace projetee dans le plan (Lxy, cm) :")
    print("  Lxy          n   azimut 68% (deg)   angle3D 68% (deg)   n(pente)")
    edges = [0.0, 1.0, 2.0, 4.0, 1.0e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (c["Lxy_cm"] >= lo) & (c["Lxy_cm"] < hi)
        if sel.sum() == 0:
            continue
        print("  %4.1f-%-5s %4d   %12.2f      %12.2f      %4d" % (lo, ("%.1f" % hi) if hi < 1e8 else "inf", sel.sum(),
              pct(c["alpha_err_deg"][sel])[1], pct(c["ang3d_deg"][sel])[1], int(np.isfinite(c["m_est"][sel]).sum())))

    print("\nVertex dans le plan, |ds| 68 %% (cm), selon |e_y| (composante de l'axe perpendiculaire au plan y = %.1f) :" % a.y_plane)
    print("  |e_y|        n   pad max   barycentre   intersection")
    for lo, hi in ((0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)):
        sel = (c["abs_ey"] >= lo) & (c["abs_ey"] < hi)
        if sel.sum() == 0:
            continue
        print("  %3.1f-%-5.2f %4d   %7.2f   %9.2f   %11.2f" % (lo, hi, sel.sum(), pct(c["ds_pmax"][sel])[1],
              pct(c["ds_cent"][sel])[1], pct(c["ds_int"][sel])[1]))
    print("\nz du vertex, |dz| 68 % (cm), selon la pente vraie |dz/ds| :")
    print("  |m|          n   |dz| 68 %")
    for lo, hi in ((0.0, 0.3), (0.3, 1.0), (1.0, 3.0), (3.0, 1.0e9)):
        sel = (c["abs_m_true"] >= lo) & (c["abs_m_true"] < hi) & np.isfinite(c["dz_vertex"])
        if sel.sum() == 0:
            continue
        print("  %3.1f-%-5s %4d   %9.2f" % (lo, ("%.1f" % hi) if hi < 1e8 else "inf", sel.sum(), pct(c["dz_vertex"][sel])[1]))

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].hist(c["alpha_err_deg"], bins=40)
    ax[0, 0].set_xlabel("|erreur d'azimut| (deg)"); ax[0, 0].set_yscale("log")
    g = np.isfinite(c["ang3d_deg"])
    ax[0, 1].hist(c["ang3d_deg"][g], bins=40)
    ax[0, 1].set_xlabel("erreur d'angle 3D (deg)"); ax[0, 1].set_yscale("log")
    g = np.isfinite(c["m_est"])
    ax[1, 0].plot(c["m_true"][g], c["m_est"][g], ".", ms=4)
    lim = np.nanmax(np.abs(c["m_true"])) if g.any() else 1.0
    ax[1, 0].plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    ax[1, 0].set_xlabel("pente vraie dz/ds"); ax[1, 0].set_ylabel("pente estimee")
    ax[1, 1].hist(c["ds_pmax"], bins=40, alpha=0.6, label="pad max")
    ax[1, 1].hist(c["ds_cent"], bins=40, alpha=0.6, label="barycentre")
    ax[1, 1].hist(c["ds_int"][np.isfinite(c["ds_int"])], bins=40, alpha=0.6, label="intersection plan y = %.1f" % a.y_plane)
    ax[1, 1].set_xlabel("vertex : |ds| le long de l'axe (cm)"); ax[1, 1].legend()
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=120)
    print("\necrit : %s.csv et %s.png" % (a.out, a.out))
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
