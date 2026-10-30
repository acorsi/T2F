#!/usr/bin/env python3
"""
Inversion de dE/dx(l) pour T2F (type2) : NNLS + Tikhonov d'ordre 2.

Chaine (la verite n'est utilisee que pour la comparaison finale) :
  1. H+L sommes par pad ; bruit : charge pad N(0,sig_q), Q(t) bruit blanc sig_b=sig_q/sqrt(nref)
  2. selection des pads, temps moyen par pad, droite 3D (ACP ponderee q^wp) -> direction d, vertex v
  3. modele direct R (replique project() du C++) sur une grille en l SIGNEE, pas --h (defaut 1 mm),
     chaque cellule = s sous-points (defaut 4) pour ne pas dependre du pas de la simulation
  4. donnees : Q(t) bruites des pads retenus, fenetre = amas +- marge ; moindres carres ponderes
       min ||(R n - b)/sig_b||^2 + lam ||D2 n||^2   avec n >= 0     (NNLS sur systeme augmente)
     D2 = derivee seconde, appliquee separement de chaque cote du vertex (H et L sont independants)
     lam : principe de discrepance (plus grand lam avec chi2 <= max(n_rows, 1.1*chi2_min)) borne
       inferieurement par --lam-min (defaut 3), ou --lam fixe
  5. comparaison a la verite, cellule par cellule, et PDF evenement par evenement.

Controle du modele direct (--check, actif par defaut) : R appliquee a la VRAIE trace doit
reproduire les Q(t) non bruites du fichier. Si l'ecart n'est pas ~1e-6, R n'est pas fidele au C++.

Robustesse (eviter le "crime inverse") : --dve (erreur relative sur v_e, utilisee par
l'estimateur ET par R) et --dsig (erreur relative sur sigma_T, sigma_L dans R seulement).

Usage
  python invert_dedx.py drift_G4_40mu_merged.h5 --pads my_pad.txt -n 200 --plot 20
  python invert_dedx.py ... --dve 0.01 --dsig 0.05
"""
import argparse
import numpy as np
import h5py
from scipy.optimize import nnls, minimize
from scipy.special import ndtr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SQ2PI = np.sqrt(2.0 * np.pi)


# ----------------------------------------------------------------- utilitaires
def fill(dense, pid, pnt, pt, pqt, dt):
    pnt = np.asarray(pnt).astype(int)
    if pnt.size == 0:
        return
    rows = np.repeat(np.asarray(pid).astype(int), pnt)
    k = np.rint(np.asarray(pt) / dt).astype(int)
    dense[rows, k] += np.asarray(pqt)


def best_cluster(row, thr, gap):
    idx = np.flatnonzero(row > thr)
    if idx.size == 0:
        return None
    cuts = np.flatnonzero(np.diff(idx) > gap + 1)
    groups = np.split(idx, cuts + 1)
    best = max(groups, key=lambda g: row[g].sum())
    return int(best[0]), int(best[-1])


def fit3d(P, w, scale, kout=4.0):
    """ACP ponderee anisotrope avec rejet iteratif des points aberrants (distance a la droite,
    en unites de scale > kout). Retourne (d, vertex, masque des points gardes)."""
    keep = np.ones(len(P), bool)
    for _ in range(6):
        Pk, wk = P[keep], w[keep]
        c = (wk[:, None] * Pk).sum(0) / wk.sum()
        X = (Pk - c) / scale * np.sqrt(wk)[:, None]
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        e = Vt[0]
        if not kout:
            break
        Xs = (P - c) / scale
        r = np.linalg.norm(Xs - np.outer(Xs @ e, e), axis=1)
        new = r <= kout
        if new.sum() < 4 or (new == keep).all():
            break
        keep = new
    d = e * scale
    d /= np.linalg.norm(d)
    if d[1] < 0:
        d = -d
    if abs(d[1]) < 1e-6:
        return None, None, keep
    return d, c - (c[1] / d[1]) * d, keep


# ------------------------------------------------------------- modele direct
def response(P, padxy, rows_pad, rows_k, cfg, sscale=1.0):
    """Reponse par electron de chaque voxel P (nv,3, cm) sur les lignes (pad, bin).
    Replique project() : fraction = produit de differences de CDF sur une largeur `pitch`,
    coupure de voisinage max(3*pitch, 4*sigma_T), temps = gaussienne echantillonnee en k*dt
    (valeur ponctuelle * dt), tronquee a +-4 sigma_t."""
    dkg, v_e, dt, pitch, DT, DL, n_bins = cfg
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    drift = dkg - z
    valid = drift > 0
    dr = np.where(valid, drift, 1.0)
    sT = DT * np.sqrt(dr) * sscale
    sL = DL * np.sqrt(dr) * sscale
    ti = dr / v_e
    sti = sL / v_e
    half = pitch / 2.0
    px, py = padxy[:, 0:1], padxy[:, 1:2]
    F = (ndtr((px + half - x) / sT) - ndtr((px - half - x) / sT)) * \
        (ndtr((py + half - y) / sT) - ndtr((py - half - y) / sT))
    rc = np.maximum(3.0 * pitch, 4.0 * sT)
    F = np.where(((px - x) ** 2 + (py - y) ** 2 <= rc ** 2) & valid, F, 0.0)
    k = rows_k[:, None].astype(float)
    G = np.exp(-0.5 * ((k * dt - ti) / sti) ** 2) / (SQ2PI * sti) * dt
    ts = np.maximum(0, np.floor((ti - 4 * sti) / dt))
    te = np.minimum(n_bins - 1, np.ceil((ti + 4 * sti) / dt))
    G = np.where((k >= ts) & (k <= te), G, 0.0)
    return F[rows_pad, :] * G


def second_diff(active, nm):
    """D2 sur les cellules actives, sans enjamber le vertex (entre cellules nm-1 et nm)."""
    rows = []
    n = len(active)
    for c in range(1, n - 1):
        if not (active[c - 1] and active[c] and active[c + 1]):
            continue
        if (c - 1) < nm <= (c + 1):        # le stencil enjambe le vertex
            continue
        r = np.zeros(n)
        r[c - 1], r[c], r[c + 1] = 1.0, -2.0, 1.0
        rows.append(r)
    return np.array(rows) if rows else np.zeros((0, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("--pads", default="my_pad.txt")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--plot", type=int, default=20, help="pages evenement dans le PDF")
    ap.add_argument("--fwhm", type=float, default=2000.0)
    ap.add_argument("--nref", type=float, default=7.0)
    ap.add_argument("--kq", type=float, default=4.0)
    ap.add_argument("--kb", type=float, default=4.0)
    ap.add_argument("--gap", type=int, default=2)
    ap.add_argument("--sz", type=float, default=0.05)
    ap.add_argument("--wp", type=float, default=1.0, help="exposant de poids pour la droite 3D")
    ap.add_argument("--kout", type=float, default=4.0,
                    help="rejet des points a plus de kout (unites sigma) de la droite ; 0 = desactive")
    ap.add_argument("--h", type=float, default=0.1, help="pas de la grille en l (cm)")
    ap.add_argument("--sub", type=int, default=4, help="sous-points par cellule")
    ap.add_argument("--margin", type=float, default=0.5, help="marge de grille (cm)")
    ap.add_argument("--lmax", type=float, default=12.0, help="demi-longueur max (cm)")
    ap.add_argument("--wmargin", type=int, default=3, help="marge de fenetre en bins")
    ap.add_argument("--lam", type=float, default=None, help="lambda fixe (sinon discrepance)")
    ap.add_argument("--lam-min", dest="lam_min", type=float, default=3.0,
                    help="plancher de lambda pour la discrepance (evite les solutions oscillantes "
                         "quand le modele est imparfait)")
    ap.add_argument("--dve", type=float, default=0.0)
    ap.add_argument("--dsig", type=float, default=0.0)
    ap.add_argument("--nocheck", action="store_true")
    ap.add_argument("--refine", action="store_true",
                    help="affine (theta, phi, vertex) en minimisant le chi2 de l'inversion (lent)")
    ap.add_argument("--lam-refine", dest="lam_refine", type=float, default=3.0)
    ap.add_argument("--maxfev", type=int, default=150)
    ap.add_argument("--truthdir", action="store_true",
                    help="diagnostic : utilise la vraie droite (theta, phi, vertex) au lieu de l'estimateur")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="invert_dedx")
    a = ap.parse_args()

    sig_q = a.fwhm / 2.3548
    sig_b = sig_q / np.sqrt(a.nref)
    NSC = 1e4                                   # echelle des inconnues (electrons)
    LAMS = np.logspace(-3, 3, 13)
    LAMS = LAMS[LAMS >= a.lam_min * 0.999]
    rng = np.random.default_rng(a.seed)

    pads = np.loadtxt(a.pads, comments="#")
    with h5py.File(a.h5, "r") as f:
        md = f["/metadata"].attrs
        n_pads = int(md["n_pads"])
        dkg, v_e, dt = float(md["dkg"]), float(md["v_e"]), float(md["delta_t"])
        W = float(md["W"])
        DT, DL = float(md["D_T"]), float(md["D_L"])
        pitch = float(md["pad_size"]) + float(md["inter_pad"])
        i0, i1 = a.start, min(a.start + a.n, f["/H/theta"].shape[0])
        th, ph = f["/H/theta"][i0:i1], f["/H/phi"][i0:i1]
        vxt, vzt = f["/H/vx"][i0:i1], f["/H/vz"][i0:i1]
        D = {(fr, k): f[f"/{fr}/{k}"][i0:i1]
             for fr in ("H", "L")
             for k in ("pid", "pq", "pt", "pqt", "pnt", "x", "y", "z", "dE")}
    N = i1 - i0
    n_bins = int(np.ceil((2.0 * dkg / v_e + 200.0) / dt)) + 1
    xy = np.zeros((n_pads, 2))
    is_cart = np.zeros(n_pads, bool)
    for r in pads:
        j = int(r[0])
        xy[j] = r[1], r[2]
        is_cart[j] = (r[3] == 0.0 and r[4] == 0.0)
    v_used = v_e * (1.0 + a.dve)
    scale = np.array([pitch / np.sqrt(12)] * 2 + [a.sz])
    cfg_true = (dkg, v_e, dt, pitch, DT, DL, n_bins)
    cfg_mod = (dkg, v_used, dt, pitch, DT, DL, n_bins)
    h, s = a.h, a.sub

    print(f"sig_q={sig_q:.0f} e-, sig_b={sig_b:.0f} e-/bin, h={h*10:.1f} mm, sub={s}, "
          f"dve={a.dve:+.3f}, dsig={a.dsig:+.3f}")

    dense = np.zeros((n_pads, n_bins))
    res = []
    pages = PdfPages(a.out + ".pdf")
    npage = 0

    for ev in range(N):
        # ---------------- mesure simulee (H+L) ----------------
        q = np.zeros(n_pads)
        touched = []
        for fr in ("H", "L"):
            pid = np.asarray(D[fr, "pid"][ev]).astype(int)
            q += np.bincount(pid, weights=np.asarray(D[fr, "pq"][ev]), minlength=n_pads)
            fill(dense, pid, D[fr, "pnt"][ev], D[fr, "pt"][ev], D[fr, "pqt"][ev], dt)
            touched.append(pid)
        touched = np.unique(np.concatenate(touched))

        # ---------------- controle du modele direct (1er evenement) ----------------
        if ev == 0 and not a.nocheck:
            Pt = np.vstack([np.column_stack([D[fr, "x"][ev], D[fr, "y"][ev], D[fr, "z"][ev]])
                            for fr in ("H", "L")])
            nt = np.concatenate([np.asarray(D[fr, "dE"][ev]) / W for fr in ("H", "L")])
            sub = dense[touched]
            rp, rk = np.nonzero(sub)
            Rt = response(Pt, xy[touched], rp, rk, cfg_true)
            mod, dat = Rt @ nt, sub[rp, rk]
            print(f"[controle modele direct, evt {i0}] {len(dat)} echantillons : "
                  f"max|mod-dat|/max(dat) = {np.max(np.abs(mod-dat))/dat.max():.2e} ; "
                  f"sum(mod)/sum(dat) = {mod.sum()/dat.sum():.6f}")

        # ---------------- bruit, selection, points ----------------
        qn = q + rng.normal(0.0, sig_q, n_pads)
        sel = np.flatnonzero((qn > a.kq * sig_q) & is_cart)
        wf = dense[sel] + rng.normal(0.0, sig_b, (sel.size, n_bins)) if sel.size else None
        tk = np.arange(n_bins) * dt
        pts, ws, kept = [], [], []
        for j, pad in enumerate(sel):
            cl = best_cluster(wf[j], a.kb * sig_b, a.gap)
            if cl is None:
                continue
            seg = wf[j, cl[0]:cl[1] + 1]
            if seg.sum() <= 0:
                continue
            tb = (seg * tk[cl[0]:cl[1] + 1]).sum() / seg.sum()
            pts.append((xy[pad, 0], xy[pad, 1], dkg - v_used * tb))
            ws.append(qn[pad])
            kept.append((j, pad, cl))
        dense[touched] = 0.0
        if len(pts) < 4:
            continue
        P3, Wt = np.array(pts), np.array(ws)
        d, v, keepm = fit3d(P3, Wt ** a.wp, scale, a.kout)
        if d is None:
            continue
        n_rej = int((~keepm).sum())
        kept = [k_ for k_, m_ in zip(kept, keepm) if m_]      # les pads aberrants sortent aussi de l'inversion
        P3 = P3[keepm]
        if a.truthdir:
            d = np.array([np.sin(th[ev]) * np.cos(ph[ev]), np.cos(th[ev]),
                          np.sin(th[ev]) * np.sin(ph[ev])])
            if d[1] < 0:
                d = -d
            v = np.array([vxt[ev], 0.0, vzt[ev]])

        # ---------------- lignes de donnees ----------------
        rp, rk, bb = [], [], []
        for jj, (j, pad, cl) in enumerate(kept):
            lo, hi = max(0, cl[0] - a.wmargin), min(n_bins - 1, cl[1] + a.wmargin)
            ks = np.arange(lo, hi + 1)
            rp.append(np.full(ks.size, jj)); rk.append(ks); bb.append(wf[j, lo:hi + 1])
        rp, rk, bb = np.concatenate(rp), np.concatenate(rk), np.concatenate(bb)
        kxy = xy[[pad for _, pad, _ in kept]]

        # ---------------- grille signee ----------------
        lp = (P3 - v) @ d
        NP = int(np.clip(np.ceil((lp.max() + a.margin) / h), 3, a.lmax / h))
        NM = int(np.clip(np.ceil((-lp.min() + a.margin) / h), 3, a.lmax / h))
        NC = NM + NP
        cell = np.arange(NC) - NM
        lsub = ((cell[:, None] + (np.arange(s)[None, :] + 0.5) / s) * h).ravel()
        def geom(dd, vv):
            Pv = vv[None, :] + lsub[:, None] * dd[None, :]
            Rs = response(Pv, kxy, rp, rk, cfg_mod, sscale=1.0 + a.dsig)      # (rows, NC*s)
            Rc_ = Rs.reshape(len(rp), NC, s).sum(axis=2) / s * NSC            # par cellule
            nrm = np.linalg.norm(Rc_, axis=0)
            return Rc_, nrm > 1e-9 * max(nrm.max(), 1e-30)

        def chi2_at(dd, vv, active_, lam):
            Rc_, _ = geom(dd, vv)
            ia_ = np.flatnonzero(active_)
            D2_ = second_diff(active_, NM)[:, ia_]
            Rw_ = Rc_[:, ia_] / sig_b
            A = np.vstack([Rw_, np.sqrt(lam) * D2_]) if D2_.shape[0] else Rw_
            rhs = np.concatenate([bb / sig_b, np.zeros(D2_.shape[0])])
            y_, _ = nnls(A, rhs, maxiter=50 * A.shape[1])
            return np.sum((Rw_ @ y_ - bb / sig_b) ** 2)

        d_est = d.copy()
        if a.refine:
            # raffinement conjoint (2 angles + vertex x,z) : minimise le chi2 de l'inversion
            _, active0 = geom(d, v)
            e1 = np.cross(d, [0.0, 0.0, 1.0]); e1 /= np.linalg.norm(e1)
            e2 = np.cross(d, e1)
            unit = np.array([1e-3, 1e-3, 0.01, 0.01])            # rad, rad, cm

            def obj(x):
                dd = d + x[0] * unit[0] * e1 + x[1] * unit[1] * e2
                dd = dd / np.linalg.norm(dd)
                vv = v + np.array([x[2] * unit[2], 0.0, x[3] * unit[3]])
                return chi2_at(dd, vv, active0, a.lam_refine)

            sim = np.vstack([np.zeros(4), np.eye(4)])
            r_ = minimize(obj, np.zeros(4), method="Nelder-Mead",
                          options=dict(initial_simplex=sim, xatol=0.05, fatol=0.5, maxfev=a.maxfev))
            x = r_.x
            d = d + x[0] * unit[0] * e1 + x[1] * unit[1] * e2
            d = d / np.linalg.norm(d)
            v = v + np.array([x[2] * unit[2], 0.0, x[3] * unit[3]])
        Rc, active = geom(d, v)
        D2 = second_diff(active, NM)
        ia = np.flatnonzero(active)
        Rw = Rc[:, ia] / sig_b
        bw = bb / sig_b
        D2a = D2[:, ia]

        def solve(lam):
            A = np.vstack([Rw, np.sqrt(lam) * D2a]) if D2a.shape[0] else Rw
            rhs = np.concatenate([bw, np.zeros(D2a.shape[0])])
            y, _ = nnls(A, rhs, maxiter=50 * A.shape[1])
            return y, np.sum((Rw @ y - bw) ** 2)

        if a.lam is not None:
            y, chi2 = solve(a.lam)
            lam_used = a.lam
        else:
            sols = [(lam,) + solve(lam) for lam in LAMS]
            chi2_min = min(t[2] for t in sols)
            target = max(len(bw), 1.1 * chi2_min)      # discrepance, tolerante si le modele est imparfait
            lam_used, y, chi2 = max([t for t in sols if t[2] <= target], key=lambda t: t[0])
        n_rec = np.zeros(NC)
        n_rec[ia] = y * NSC
        E_rec = n_rec * W                                           # MeV par cellule

        # ---------------- verite sur la meme grille ----------------
        u_H = np.array([np.sin(th[ev]) * np.cos(ph[ev]), np.cos(th[ev]),
                        np.sin(th[ev]) * np.sin(ph[ev])])
        sgnH = 1.0 if u_H @ d > 0 else -1.0
        E_true = np.zeros(NC)
        E_out = 0.0
        for fr, sg in (("H", sgnH), ("L", -sgnH)):
            r = np.column_stack([D[fr, "x"][ev], D[fr, "y"][ev], D[fr, "z"][ev]])
            dE = np.asarray(D[fr, "dE"][ev])
            sp = np.linalg.norm(r - r[0], axis=1)                 # abscisse curviligne des voxels
            if len(sp) < 2:
                continue
            stp = np.median(np.diff(sp))                          # pas reel des voxels (pas forcement 0.5 mm)
            # chaque voxel est etale uniformement sur son pas, centre sur sa position (sp[0]=0 : borne a 0).
            # Sans cela, un pas de voxel non commensurable avec la grille cree des pics regulierement espaces.
            edges = np.concatenate([[0.0], sp[1:] - stp / 2.0, [sp[-1] + stp / 2.0]])
            cum = np.concatenate([[0.0], np.cumsum(dE)])
            nj = int(np.ceil(edges[-1] / h)) + 1
            Ej = np.diff(np.interp(np.arange(nj + 1) * h, edges, cum))
            jc = np.arange(nj)
            c = NM + jc if sg > 0 else NM - 1 - jc
            inside = (c >= 0) & (c < NC)
            np.add.at(E_true, c[inside], Ej[inside])
            E_out += dE.sum() - Ej[inside].sum()
        Et = E_true.sum() + E_out
        mask = E_true > 0
        rel = np.sqrt(np.sum((E_rec - E_true) ** 2) / np.sum(E_true ** 2))
        ratio = abs(u_H[2] / u_H[1])
        ang = np.degrees(np.arccos(np.clip(abs(d @ u_H), 0, 1)))
        ang0 = np.degrees(np.arccos(np.clip(abs(d_est @ u_H), 0, 1)))
        res.append((ev + i0, ratio, rel, E_rec.sum() / Et, chi2 / len(bw), lam_used, ang,
                    E_out / Et, ang0, n_rej, len(kept)))

        # ---------------- page PDF ----------------
        if npage < a.plot:
            npage += 1
            fig, ax = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [2, 1]})
            xc = (cell + 0.5) * h * 10
            fac = 1.0 / (h * 10)
            ax[0].step(xc, E_true * fac, where="mid", color="k", lw=1.6, label="verite")
            ax[0].step(xc, E_rec * fac, where="mid", color="C3", lw=1.4, label="reconstruit")
            ax[0].axvline(0, color="gray", lw=0.6)
            ax[0].set_xlabel("l (mm)  [droite ajustee]"); ax[0].set_ylabel("dE/dx (MeV/mm)")
            left, right = ("L", "H") if sgnH > 0 else ("H", "L")
            ax[0].text(0.02, 0.95, f"<- {left}", transform=ax[0].transAxes, va="top")
            ax[0].text(0.98, 0.95, f"{right} ->", transform=ax[0].transAxes, va="top", ha="right")
            ax[0].legend(loc="upper center")
            ax[0].set_title(f"evt {ev+i0}  theta={np.degrees(th[ev]):.0f}  phi={np.degrees(ph[ev]):.0f}"
                            f"  |uz/uy|={ratio:.2f}  err. angle={ang:.3f} deg (estimateur {ang0:.3f})\n"
                            f"err. rel. profil={100*rel:.1f}%  E_rec/E_vraie={E_rec.sum()/Et:.3f}  "
                            f"chi2/n={chi2/len(bw):.2f}  lam={lam_used:.2g}")
            kk = np.unique(rk)
            dat_t = np.array([bb[rk == k_].sum() for k_ in kk])
            mod_t = np.array([(Rc[:, ia] @ y)[rk == k_].sum() for k_ in kk])
            ax[1].plot(kk * dt, dat_t, ".", color="k", ms=3, label="donnees (bruitees)")
            ax[1].plot(kk * dt, mod_t, "-", color="C3", lw=1, label="modele")
            ax[1].set_xlabel("t (ns)"); ax[1].set_ylabel("somme des pads (e-/bin)")
            ax[1].legend()
            fig.tight_layout()
            pages.savefig(fig)
            plt.close(fig)

    R = np.array(res)
    print(f"\n{len(R)} evenements inverses sur {N}")
    if len(R):
        print(f"erreur relative du profil (L2) : mediane={np.median(R[:,2])*100:.1f}%  "
              f"68%={np.percentile(R[:,2],68)*100:.1f}%")
        print(f"E_rec/E_vraie : mediane={np.median(R[:,3]):.3f}  [5%,95%]="
              f"[{np.percentile(R[:,3],5):.3f},{np.percentile(R[:,3],95):.3f}]")
        print(f"chi2/n_lignes : mediane={np.median(R[:,4]):.2f} ; lambda median={np.median(R[:,5]):.2g}")
        print(f"erreur angulaire de la droite : mediane={np.median(R[:,6]):.3f} deg "
              f"(estimateur seul : {np.median(R[:,8]):.3f} deg)")
        print(f"energie verite hors grille : mediane={np.median(R[:,7])*100:.2f}%")
        bad = np.flatnonzero((R[:, 2] > 0.2) | (R[:, 4] > 10))
        print(f"\nevenements problematiques (err. profil > 20% ou chi2/n > 10) : {len(bad)}")
        for b in bad[np.argsort(-R[bad, 2])][:12]:
            print(f"  evt {int(R[b,0]):6d} : err={R[b,2]*100:5.1f}%  E_rec/E_vraie={R[b,3]:.2f}  "
                  f"chi2/n={R[b,4]:8.1f}  |uz/uy|={R[b,1]:.2f}  ang={R[b,6]:.3f} deg  "
                  f"pads rejetes={int(R[b,9])}/{int(R[b,9]+R[b,10])}")
        print("\npar tranche de |uz|/|uy| :")
        for lo, hi in [(0, 0.2), (0.2, 0.5), (0.5, 1), (1, np.inf)]:
            m = (R[:, 1] >= lo) & (R[:, 1] < hi)
            if m.sum() >= 3:
                print(f"  {lo:g}-{hi:g} ({m.sum():3d} evts) : err. profil med={np.median(R[m,2])*100:5.1f}%  "
                      f"E_rec/E_vraie med={np.median(R[m,3]):.3f}")
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
        ax[0].scatter(R[:, 1], R[:, 2] * 100, s=6); ax[0].set_xscale("log")
        ax[0].set_xlabel("|uz|/|uy|"); ax[0].set_ylabel("erreur relative du profil (%)")
        ax[1].hist(R[:, 3], bins=50); ax[1].set_xlabel("E_rec / E_vraie")
        ax[2].hist(R[:, 4], bins=50); ax[2].set_xlabel("chi2 / n_lignes")
        fig.tight_layout(); pages.savefig(fig); plt.close(fig)
        np.savez(a.out + ".npz", res=R)
    pages.close()
    print(f"\nPDF : {a.out}.pdf")


if __name__ == "__main__":
    main()
