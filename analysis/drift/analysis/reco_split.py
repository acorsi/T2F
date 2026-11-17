#!/usr/bin/env python3
"""
reco_split.py : ETAPE 0 (cas de traces NON COLINEAIRES) : une demi-droite par demi-plan de l'anode, issues d'un
vertex COMMUN situe dans le plan y = y_plane (plan de symetrie de l'anode), puis comparaison a la verite.

Usage :
    python reco_split.py sortie.h5 [--start 0] [--n 200] [--k_thr 10] [--f_max 0.1] [--min_half 3]
                         [--sign_z 1] [--y_plane 0] [--t_off NS] [--out reco_split] [--show]

Fichiers necessaires dans le meme dossier : reco_dir.py, t2f_io.py, view_fit.py (et view_pads.py).
Sorties : tableau resume a l'ecran, <out>.csv (une ligne par evenement), <out>.png (4 figures).

Methode (la verite n'est jamais utilisee pour l'estimation) :
  1. pads forts : memes criteres que reco_dir.py (amplitude >= max(k_thr x seuil, f_max x maximum), charge collectee).
  2. un demi-plan par signe de y - y_plane : un pad centre exactement sur le plan est ecarte (compte dans le resume).
     Il faut au moins min_half pads forts dans chaque demi-plan, sinon l'evenement est ecarte.
  3. vertex commun P_v = (x_v, y_plane) et une demi-droite par demi-plan, de direction (cos a, sin a) partant de P_v.
     Pour x_v fixe, la meilleure direction du demi-plan h est le vecteur propre de plus petite valeur propre de
     M_h(x_v) = somme_i w_i D_i D_i^T, D_i = X_i - P_v (poids w = amplitude), et le cout vaut cette valeur propre ;
     on minimise la somme des deux couts sur x_v (balayage puis section doree). L'orientation de chaque demi-droite
     va du vertex vers le nuage de pads.
  4. temps : t_i = t_v + b_h s_i avec s_i = D_i . e_h (distance au vertex le long de la demi-droite) et t_i = barycentre
     temporel du pad moins celui de h ; (t_v, b_H, b_L) par moindres carres ponderes. Pente dz/ds = - sign_z ve_kg b_h,
     et z_v = dkg - ve_kg (t_v - t_off), t_off = (dga/ve_ga)/2 - delta_t/2 (--t_off pour le changer).
  Pour comparaison, chaque demi-plan est aussi traite SEUL (axe principal, intersection avec y = y_plane, regression
  du temps avec son propre point de depart) : cela montre ce que la contrainte du vertex commun apporte.

Verite : chaque demi-plan est associe au fragment dont uy a le signe de ce demi-plan ; erreur d'angle 3D entre la
direction estimee (orientee) et la vraie direction du fragment ; erreur de x_v (cm) et de z_v (cm) ; l'angle entre
u_H et -u_L (non-colinearite vraie) sert a classer les evenements.
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
from reco_dir import true_direction, pad_waveforms, pct


def lam_min_curve(xv, S, Sx, Sy, Sxx, Sxy, Syy, yp):
    """plus petite valeur propre de M_h(x_v) pour un tableau de x_v ; coordonnees y relatives a yp (Sy, Sxy, Syy deja relatifs)"""
    Mxx = Sxx - 2.0 * xv * Sx + xv * xv * S
    Mxy = Sxy - xv * Sy
    Myy = Syy
    return 0.5 * (Mxx + Myy - np.sqrt((Mxx - Myy) ** 2 + 4.0 * Mxy ** 2))


def moments(X, w):
    x = X[:, 0]
    y = X[:, 1]
    return (w.sum(), (w * x).sum(), (w * y).sum(), (w * x * x).sum(), (w * x * y).sum(), (w * y * y).sum())


def half_direction(X, w, xv, yp):
    """direction (cos, sin) de la demi-droite de P_v = (xv, yp) vers le nuage, pour un demi-plan"""
    D = X - np.array([xv, yp])
    M = (w[:, None, None] * D[:, :, None] * D[:, None, :]).sum(axis=0)
    lam, vec = np.linalg.eigh(M)
    n = vec[:, 0]
    e = np.array([n[1], -n[0]])
    mean = (w[:, None] * D).sum(axis=0) / w.sum()
    if float(e @ mean) < 0.0:
        e = -e
    return e


def fit_vertex_x(Xp, wp, Xm, wm, yp):
    """x_v minimisant la somme des couts des deux demi-plans (y relatif a yp)"""
    Xp = Xp.copy(); Xm = Xm.copy()
    Xp[:, 1] -= yp; Xm[:, 1] -= yp
    mp = moments(Xp, wp)
    mm = moments(Xm, wm)
    allx = np.concatenate([Xp[:, 0], Xm[:, 0]])
    grid = np.linspace(allx.min() - 3.0, allx.max() + 3.0, 700)
    cost = lam_min_curve(grid, *mp, yp) + lam_min_curve(grid, *mm, yp)
    k = int(np.argmin(cost))
    step = grid[1] - grid[0]
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    g = 0.6180339887498949
    c = hi - g * (hi - lo); d = lo + g * (hi - lo)
    f = lambda xv: float(lam_min_curve(np.array([xv]), *mp, yp)[0] + lam_min_curve(np.array([xv]), *mm, yp)[0])
    fc, fd = f(c), f(d)
    for it in range(40):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - g * (hi - lo); fc = f(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + g * (hi - lo); fd = f(d)
    return 0.5 * (lo + hi)


def estimate(wfs, flag, x, y, area_h, m_h, thr, a, pitch, ve, dkg, t_off):
    n = len(wfs)
    if n == 0:
        return None, "aucun pad"
    amp = np.array([w[1].max() for w in wfs])
    qint = np.array([w[1].sum() * (w[0][1] - w[0][0]) / area_h if len(w[1]) > 1 else 0.0 for w in wfs])
    amin = max(a.k_thr * thr, a.f_max * amp.max())
    ok = (flag != 0) & (amp >= amin) & (qint > 0.5 * amp)
    on_plane = ok & (np.abs(y - a.y_plane) < 1.0e-9)
    ok = ok & ~on_plane
    ip = np.where(ok & (y > a.y_plane))[0]
    im = np.where(ok & (y < a.y_plane))[0]
    if len(ip) < a.min_half or len(im) < a.min_half:
        return None, "demi-plan avec moins de %d pads forts" % a.min_half
    tc_all = np.array([(w[0] * w[1]).sum() / w[1].sum() if w[1].sum() > 0 else np.nan for w in wfs]) - m_h
    out = {"n_on_plane": int(on_plane.sum()), "n_p": len(ip), "n_m": len(im)}
    halves = {}
    for name, idx in (("p", ip), ("m", im)):
        X = np.vstack([x[idx], y[idx]]).T
        halves[name] = (X, amp[idx], tc_all[idx])
    # --- vertex commun ---
    xv = fit_vertex_x(halves["p"][0], halves["p"][1], halves["m"][0], halves["m"][1], a.y_plane)
    out["xv"] = xv
    rows, rhs, wts = [], [], []
    for k, name in enumerate(("p", "m")):
        X, w, tc = halves[name]
        e = half_direction(X, w, xv, a.y_plane)
        s = (X - np.array([xv, a.y_plane])) @ e
        out["e_" + name] = e
        halves[name] = (X, w, tc, e, s)
        for si, ti, wi in zip(s, tc, w):
            r = [1.0, 0.0, 0.0]
            r[1 + k] = si
            rows.append(r); rhs.append(ti); wts.append(wi)
    A = np.array(rows) * np.sqrt(np.array(wts))[:, None]
    bvec = np.array(rhs) * np.sqrt(np.array(wts))
    sol, res_, rk, sv = np.linalg.lstsq(A, bvec, rcond=None)
    t_v, b_p, b_m = sol
    out["zv"] = dkg - ve * (t_v - t_off)
    out["m_p"] = -a.sign_z * ve * b_p
    out["m_m"] = -a.sign_z * ve * b_m
    out["u_p"] = np.array([out["e_p"][0], out["e_p"][1], out["m_p"]]); out["u_p"] /= np.linalg.norm(out["u_p"])
    out["u_m"] = np.array([out["e_m"][0], out["e_m"][1], out["m_m"]]); out["u_m"] /= np.linalg.norm(out["u_m"])
    # --- demi-plans traites seuls (sans vertex commun) ---
    for name, sg in (("p", 1.0), ("m", -1.0)):
        X, w, tc = halves[name][:3]
        mean = (w[:, None] * X).sum(axis=0) / w.sum()
        D = X - mean
        C = (w[:, None, None] * D[:, :, None] * D[:, None, :]).sum(axis=0) / w.sum()
        lam, vec = np.linalg.eigh(C)
        e = vec[:, 1]
        if sg * e[1] < 0.0:
            e = -e
        s = D @ e
        S = w.sum(); Sx = (w * s).sum(); Sy = (w * tc).sum(); Sxx = (w * s * s).sum(); Sxy = (w * s * tc).sum()
        den = S * Sxx - Sx * Sx
        mi = np.nan
        xi = np.nan
        if den > 0.0:
            b = (S * Sxy - Sx * Sy) / den
            mi = -a.sign_z * ve * b
        if abs(e[1]) >= 0.05:
            xi = mean[0] + ((a.y_plane - mean[1]) / e[1]) * e[0]
        out["ui_" + name] = np.array([e[0], e[1], mi]) / np.linalg.norm([e[0], e[1], mi]) if np.isfinite(mi) else None
        out["xi_" + name] = xi
    return out, None


def angle_deg(u, v):
    return float(np.degrees(np.arccos(max(min(float(u @ v), 1.0), -1.0))))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("h5file")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--k_thr", type=float, default=10.0)
    p.add_argument("--f_max", type=float, default=0.1)
    p.add_argument("--min_half", type=int, default=3, help="pads forts minimum dans chaque demi-plan")
    p.add_argument("--sign_z", type=float, default=1.0)
    p.add_argument("--y_plane", type=float, default=0.0)
    p.add_argument("--t_off", type=float, default=None, help="decalage temporel (ns) ; defaut : (dga/ve_ga)/2 - delta_t/2")
    p.add_argument("--out", type=str, default="reco_split")
    p.add_argument("--show", action="store_true")
    a = p.parse_args()

    f = h5py.File(a.h5file, "r")
    at = f["/metadata"].attrs
    dt = float(at["delta_t"]); R = int(at["adc_ratio"]); ve = float(at["ve_kg"]); dkg = float(at["dkg"])
    thr = float(at["read_thr_e"]) if int(at["noise_enable"]) == 1 else 0.0
    pitch = float(at["pad_size"]) + float(at["inter_pad"])
    axis = at["emission_axis"]
    axis = axis.decode() if isinstance(axis, bytes) else str(axis)
    t_off = (0.5 * float(at["dga"]) / float(at["ve_ga"]) - 0.5 * dt) if a.t_off is None else a.t_off
    resp = Response(at, dt)
    hf = resp.hcont(resp.tf)
    m_h = float((resp.tf * hf).sum() / hf.sum())
    geom = pad_geometry(f)
    ys = np.array([g[1] for g in geom.values()])
    N = f["/event_id"].shape[0]
    i1 = min(N, a.start + a.n)
    print("[reco_split] %s : evenements %d a %d | reponse : %s" % (a.h5file, a.start, i1 - 1, resp.descr))
    print("[reco_split] plan du vertex y = %.2f cm ; plus petit |y - y_plane| d'un centre de pad : %.4f cm (pas de pad %.3f cm) ; "
          "pads centres sur le plan : %d" % (a.y_plane, np.abs(ys - a.y_plane).min(), pitch, int((np.abs(ys - a.y_plane) < 1e-9).sum())))
    print("[reco_split] t_off = %.1f ns" % t_off)

    cols = ["event", "n_p", "n_m", "on_plane", "acol_deg", "ang_con_p", "ang_con_m", "ang_ind_p", "ang_ind_m",
            "dx_con", "dz_con", "dx_ind_p", "dx_ind_m", "xgap_ind"]
    rows = []
    skipped = {}
    for i in range(a.start, i1):
        ev = event_pads(f, i, geom)
        phase = int(f["/P/phase"][i])
        wfs = pad_waveforms(ev, phase, R, dt)
        res, why = estimate(wfs, ev["pflag"], ev["x"], ev["y"], resp.area, m_h, thr, a, pitch, ve, dkg, t_off)
        if res is None:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        uH = true_direction(f, "H", i, axis)
        uL = true_direction(f, "L", i, axis)
        vx, vy, vz = read_vertex(f, i)
        if uH[1] * uL[1] >= 0.0 or uH[1] == 0.0 or uL[1] == 0.0:
            skipped["fragments du meme cote du plan (verite)"] = skipped.get("fragments du meme cote du plan (verite)", 0) + 1
            continue
        u_true = {"p": uH if uH[1] > 0.0 else uL, "m": uH if uH[1] < 0.0 else uL}
        acol = angle_deg(uH, -uL)
        ang_con = dict((k, angle_deg(res["u_" + k], u_true[k])) for k in ("p", "m"))
        ang_ind = dict((k, angle_deg(res["ui_" + k], u_true[k]) if res["ui_" + k] is not None else np.nan) for k in ("p", "m"))
        xg = res["xi_p"] - res["xi_m"] if np.isfinite(res["xi_p"]) and np.isfinite(res["xi_m"]) else np.nan
        rows.append((i, res["n_p"], res["n_m"], res["n_on_plane"], acol, ang_con["p"], ang_con["m"], ang_ind["p"], ang_ind["m"],
                     res["xv"] - vx, res["zv"] - vz, res["xi_p"] - vx, res["xi_m"] - vx, xg))

    rows = np.array(rows, dtype=float)
    ntot = i1 - a.start
    print("\n=== Resume : %d evenements, %d evalues, ecartes : %s ===" % (ntot, len(rows), skipped if skipped else "aucun"))
    if len(rows) == 0:
        sys.exit(0)
    np.savetxt(a.out + ".csv", rows, delimiter=",", header=",".join(cols), comments="")
    c = dict((k, rows[:, j]) for j, k in enumerate(cols))
    print("non-colinearite vraie (angle entre u_H et -u_L) : mediane %.2f deg, 68 %% %.2f, 95 %% %.2f"
          % tuple(pct(c["acol_deg"])))
    ang_con = np.concatenate([c["ang_con_p"], c["ang_con_m"]])
    ang_ind = np.concatenate([c["ang_ind_p"], c["ang_ind_m"]])
    dx_ind = np.concatenate([c["dx_ind_p"], c["dx_ind_m"]])
    print("                                   mediane     68 %      95 %")
    for name, v, unit in (("angle 3D, vertex commun", ang_con, "deg"), ("angle 3D, demi-plan seul", ang_ind, "deg"),
                          ("x_v, vertex commun", c["dx_con"], "cm"), ("x_v, intersection d'un demi-plan seul", dx_ind, "cm"),
                          ("z_v, vertex commun", c["dz_con"], "cm"), ("ecart des 2 intersections seules", c["xgap_ind"], "cm")):
        m, p68, p95 = pct(v)
        print("  %-36s %8.3f %8.3f %8.3f  %s" % (name, m, p68, p95, unit))
    print("\nSelon la non-colinearite vraie (angle entre u_H et -u_L, degres) :")
    print("  intervalle     n   angle3D commun 68%   angle3D seul 68%   x_v 68% (cm)   z_v 68% (cm)")
    for lo, hi in ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 1.0e9)):
        sel = (c["acol_deg"] >= lo) & (c["acol_deg"] < hi)
        if sel.sum() == 0:
            continue
        ac = np.concatenate([c["ang_con_p"][sel], c["ang_con_m"][sel]])
        ai = np.concatenate([c["ang_ind_p"][sel], c["ang_ind_m"][sel]])
        print("  %4.1f-%-6s %4d   %16.3f   %16.3f   %12.3f   %12.3f" % (lo, ("%.1f" % hi) if hi < 1e8 else "inf", sel.sum(),
              pct(ac)[1], pct(ai)[1], pct(c["dx_con"][sel])[1], pct(c["dz_con"][sel])[1]))

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    top = max(np.nanpercentile(ang_ind, 99), np.nanpercentile(ang_con, 99))
    bins = np.linspace(0, top, 40)
    ax[0, 0].hist(ang_ind[np.isfinite(ang_ind)], bins=bins, alpha=0.6, label="demi-plan seul")
    ax[0, 0].hist(ang_con, bins=bins, alpha=0.6, label="vertex commun")
    ax[0, 0].set_xlabel("erreur d'angle 3D (deg)"); ax[0, 0].set_yscale("log"); ax[0, 0].legend()
    ax[0, 1].hist(c["dx_con"], bins=40, alpha=0.6, label="vertex commun")
    ax[0, 1].hist(dx_ind[np.isfinite(dx_ind)], bins=40, alpha=0.6, label="demi-plan seul")
    ax[0, 1].set_xlabel("x_v estime - vrai (cm)"); ax[0, 1].set_yscale("log"); ax[0, 1].legend()
    ax[1, 0].hist(c["dz_con"], bins=40)
    ax[1, 0].set_xlabel("z_v estime - vrai (cm)")
    g = np.isfinite(c["xgap_ind"])
    ax[1, 1].plot(c["acol_deg"][g], np.abs(c["xgap_ind"][g]), ".", ms=3)
    ax[1, 1].set_xlabel("non-colinearite vraie (deg)"); ax[1, 1].set_ylabel("|ecart des 2 intersections| (cm)")
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=120)
    print("\necrit : %s.csv et %s.png" % (a.out, a.out))
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
