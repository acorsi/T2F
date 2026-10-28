#!/usr/bin/env python3
"""
Estimateur 3D de trajectoire (T2F type2) : theta, phi, vertex.

Principe
  1. H+L sommes par pad (un seul canal par pad, comme sur les donnees reelles).
  2. Bruit : (a) charge totale du pad  N(0, sig_q)            -> selection des pads
             (b) Q(t) : bruit blanc par bin, sig_b = sig_q/sqrt(nref)
                 (somme sur nref bins = sig_q, ex. nref=20)
  3. Pad retenu si charge bruitee > kq*sig_q. Dans son Q(t) bruite, on garde le plus
     gros amas de bins > kb*sig_b (trous <= gap bins), temps moyen -> z = dkg - v_e*t.
     t0 non necessaire : il est absorbe par z0 (intersection avec y=0).
  4. Points (x_pad, y_pad, z) ; droite 3D par ACP ponderee (poids q^p), axes remis a
     l'echelle (sigma_xy = pitch/sqrt(12), sigma_z = sz). Vertex = intersection avec y=0.
  5. Comparaison a la verite, droite non orientee (H replie sur uy>0).

Usage
  python fit3d_t2f.py drift_G4_40mu_merged.h5 --pads my_pad.txt -n 5000
"""
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WPOWS = (0.0, 0.5, 1.0)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def fill(dense, pid, pnt, pt, pqt, dt):
    """Remplit dense[pad, bin] (indices uniques pour un fragment donne)."""
    pnt = np.asarray(pnt).astype(int)
    if pnt.size == 0:
        return
    rows = np.repeat(np.asarray(pid).astype(int), pnt)
    k = np.rint(np.asarray(pt) / dt).astype(int)
    dense[rows, k] += np.asarray(pqt)


def best_cluster(row, thr, gap):
    """Plus gros amas de bins > thr, trous <= gap. Retourne (i0, i1) ou None."""
    idx = np.flatnonzero(row > thr)
    if idx.size == 0:
        return None
    cuts = np.flatnonzero(np.diff(idx) > gap + 1)
    groups = np.split(idx, cuts + 1)
    best = max(groups, key=lambda g: row[g].sum())
    return int(best[0]), int(best[-1])


def fit3d(P, w, scale):
    """ACP ponderee anisotrope. Retourne (direction unitaire, uy>0 ; vertex a y=0)."""
    c = (w[:, None] * P).sum(0) / w.sum()
    X = (P - c) / scale * np.sqrt(w)[:, None]
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    d = Vt[0] * scale
    d /= np.linalg.norm(d)
    if d[1] < 0:
        d = -d
    if abs(d[1]) < 1e-6:
        return None, None
    return d, c - (c[1] / d[1]) * d


def q68(x):
    return np.nanpercentile(np.abs(x), 68)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("--pads", default="../build/my_pad.txt")
    ap.add_argument("-n", type=int, default=5000)
    ap.add_argument("--fwhm", type=float, default=2000.0, help="bruit charge pad, FWHM e-")
    ap.add_argument("--nref", type=float, default=20.0, help="bins d'integration equivalents")
    ap.add_argument("--kq", type=float, default=4.0, help="seuil pad, en sig_q")
    ap.add_argument("--kb", type=float, default=4.0, help="seuil bin, en sig_b")
    ap.add_argument("--gap", type=int, default=2, help="trous admis dans un amas (bins)")
    ap.add_argument("--sz", type=float, default=0.05, help="sigma_z de reference (cm)")
    ap.add_argument("--wplot", type=float, default=0.5, help="puissance de poids pour les figures")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="fit3d_t2f")
    a = ap.parse_args()

    sig_q = a.fwhm / 2.3548
    sig_b = sig_q / np.sqrt(a.nref)
    rng = np.random.default_rng(a.seed)

    pads = np.loadtxt(a.pads, comments="#")
    with h5py.File(a.h5, "r") as f:
        md = f["/metadata"].attrs
        n_pads = int(md["n_pads"])
        dkg, v_e, dt = float(md["dkg"]), float(md["v_e"]), float(md["delta_t"])
        pitch = float(md["pad_size"]) + float(md["inter_pad"])
        N = min(a.n, f["/H/theta"].shape[0])
        th, ph = f["/H/theta"][:N], f["/H/phi"][:N]
        vx_t, vz_t = f["/H/vx"][:N], f["/H/vz"][:N]
        D = {(fr, k): f[f"/{fr}/{k}"][:N]
             for fr in ("H", "L") for k in ("pid", "pq", "pt", "pqt", "pnt")}

    n_bins = int(np.ceil((2.0 * dkg / v_e + 200.0) / dt)) + 1
    xy = np.zeros((n_pads, 2))
    is_cart = np.zeros(n_pads, bool)
    for r in pads:
        i = int(r[0])
        xy[i] = r[1], r[2]
        is_cart[i] = (r[3] == 0.0 and r[4] == 0.0)
    scale = np.array([pitch / np.sqrt(12), pitch / np.sqrt(12), a.sz])

    print(f"sig_q={sig_q:.0f} e-, sig_b={sig_b:.0f} e-/bin (nref={a.nref:g}), "
          f"n_bins={n_bins}, pitch={pitch*10:.1f} mm, sigma_xy={scale[0]*10:.2f} mm, "
          f"sigma_z={a.sz*10:.2f} mm")

    dense = np.zeros((n_pads, n_bins))
    ok = np.zeros(N, bool)
    npts = np.zeros(N, int)
    widths = []
    R = {p: {k: np.full(N, np.nan) for k in ("delta", "dth", "dphi", "dvx", "dvz")}
         for p in WPOWS}
    ratio = np.zeros(N)

    for i in range(N):
        u = np.array([np.sin(th[i]) * np.cos(ph[i]), np.cos(th[i]),
                      np.sin(th[i]) * np.sin(ph[i])])
        if u[1] < 0:
            u = -u
        ratio[i] = abs(u[2]) / abs(u[1])

        q = np.zeros(n_pads)
        touched = []
        for fr in ("H", "L"):
            pid = np.asarray(D[fr, "pid"][i]).astype(int)
            q += np.bincount(pid, weights=np.asarray(D[fr, "pq"][i]), minlength=n_pads)
            fill(dense, pid, D[fr, "pnt"][i], D[fr, "pt"][i], D[fr, "pqt"][i], dt)
            touched.append(pid)
        touched = np.unique(np.concatenate(touched))

        qn = q + rng.normal(0.0, sig_q, n_pads)
        sel = np.flatnonzero((qn > a.kq * sig_q) & is_cart)
        pts, ws = [], []
        if sel.size:
            wf = dense[sel] + rng.normal(0.0, sig_b, (sel.size, n_bins))
            tk = np.arange(n_bins) * dt
            for j, pad in enumerate(sel):
                cl = best_cluster(wf[j], a.kb * sig_b, a.gap)
                if cl is None:
                    continue
                seg = wf[j, cl[0]:cl[1] + 1]
                s = seg.sum()
                if s <= 0:
                    continue
                tbar = (seg * tk[cl[0]:cl[1] + 1]).sum() / s
                pts.append((xy[pad, 0], xy[pad, 1], dkg - v_e * tbar))
                ws.append(qn[pad])
                widths.append(cl[1] - cl[0] + 1)
        dense[touched] = 0.0

        npts[i] = len(pts)
        if len(pts) < 4:
            continue
        P, W = np.array(pts), np.array(ws)
        for p in WPOWS:
            d, v = fit3d(P, W ** p, scale)
            if d is None:
                continue
            r = R[p]
            r["delta"][i] = np.degrees(np.arccos(np.clip(d @ u, -1, 1)))
            r["dth"][i] = np.degrees(np.arccos(d[1]) - np.arccos(u[1]))
            r["dphi"][i] = np.degrees(wrap(np.arctan2(d[2], d[0]) - np.arctan2(u[2], u[0])))
            r["dvx"][i] = (v[0] - vx_t[i]) * 10.0
            r["dvz"][i] = (v[2] - vz_t[i]) * 10.0
        ok[i] = True

    print(f"\n{N} evenements, {ok.sum()} ajustes ; points par ajustement : "
          f"med={np.median(npts[ok]):.0f} [5%,95%]="
          f"[{np.percentile(npts[ok],5):.0f},{np.percentile(npts[ok],95):.0f}]")
    if widths:
        print(f"Largeur des amas Q(t) : med={np.median(widths):.0f} bins "
              f"(comparer a nref={a.nref:g}) ; 5-95% = "
              f"[{np.percentile(widths,5):.0f},{np.percentile(widths,95):.0f}]")

    bins = [(0, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, np.inf)]
    print("\nErreurs : mediane / percentile 68 de |erreur|")
    for p in WPOWS:
        r = R[p]
        print(f"\n=== poids q^{p:g} ===")
        print(f"{'|uz|/|uy|':>12} {'frac':>6} {'delta(deg)':>14} {'dtheta(deg)':>14} "
              f"{'dphi(deg)':>14} {'dvx(mm)':>14} {'dvz(mm)':>14}")
        for lo, hi in [(0, np.inf)] + bins:
            m = ok & (ratio >= lo) & (ratio < hi)
            if m.sum() < 5:
                continue
            lab = "tous" if hi == np.inf and lo == 0 else f"{lo:g}-{hi:g}"
            cells = [f"{np.nanmedian(np.abs(r[k][m])):6.2f}/{q68(r[k][m]):6.2f}"
                     for k in ("delta", "dth", "dphi", "dvx", "dvz")]
            print(f"{lab:>12} {100*m.sum()/ok.sum():5.0f}% " + " ".join(f"{c:>14}" for c in cells))
        m = ok
        print(f"biais moyen : dvx={np.nanmean(r['dvx'][m]):+.2f} mm, dvz={np.nanmean(r['dvz'][m]):+.2f} mm, "
              f"dtheta={np.nanmean(r['dth'][m]):+.2f} deg")

    r = R[a.wplot]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0, 0].scatter(ratio[ok], r["delta"][ok], s=2, alpha=0.3)
    ax[0, 0].set_xscale("log"); ax[0, 0].set_yscale("log")
    ax[0, 0].set_xlabel("|uz|/|uy|"); ax[0, 0].set_ylabel("erreur angulaire (deg)")
    ax[0, 1].scatter(ratio[ok], r["dvz"][ok], s=2, alpha=0.3)
    ax[0, 1].set_xscale("log"); ax[0, 1].set_xlabel("|uz|/|uy|"); ax[0, 1].set_ylabel("dvz (mm)")
    ax[0, 2].scatter(ratio[ok], r["dvx"][ok], s=2, alpha=0.3)
    ax[0, 2].set_xscale("log"); ax[0, 2].set_xlabel("|uz|/|uy|"); ax[0, 2].set_ylabel("dvx (mm)")
    ax[1, 0].hist(r["dth"][ok], bins=100, range=(-5, 5)); ax[1, 0].set_title("dtheta (deg)")
    ax[1, 1].hist(r["dphi"][ok], bins=100, range=(-20, 20)); ax[1, 1].set_title("dphi (deg)")
    ax[1, 2].hist(npts[ok], bins=50); ax[1, 2].set_title("points par ajustement")
    fig.suptitle(f"poids q^{a.wplot:g}")
    fig.tight_layout()
    fig.savefig(a.out + ".pdf")
    np.savez(a.out + ".npz", ratio=ratio, ok=ok, npts=npts,
             **{f"{k}_p{p:g}": R[p][k] for p in WPOWS for k in R[p]})
    print(f"\nFigures : {a.out}.pdf   Donnees : {a.out}.npz")


if __name__ == "__main__":
    main()
