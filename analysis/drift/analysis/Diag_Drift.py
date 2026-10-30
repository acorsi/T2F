#!/usr/bin/env python3
"""
Diagnostic T2F (type2) avant reconstruction inverse.

Usage :
    python diag_t2f.py ../drift_output/drift_G4_40mu_merged.h5 --pads my_pad.txt -n 5000

Convention : theta mesure depuis Y, uy=cos(theta), uz=sin(theta)*sin(phi).
Bruit : gaussien sur la CHARGE TOTALE du pad (FWHM en electrons), H+L sommes par pad.
Sorties : resume texte + diag_t2f.pdf + diag_t2f.npz
"""
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THRESH = [3.0, 4.0, 5.0]   # seuils en sigma sur la charge du pad


def classify(pads, r_trans, n_pads):
    """0 = quinconce, 1 = pad de transition, 2 = anneau peripherique."""
    cls = np.zeros(n_pads, int)
    for row in pads:
        pid, r_out = int(row[0]), row[4]
        if r_out <= 0.0:
            c = 0
        elif abs(r_out - r_trans) < 1e-3:
            c = 1
        else:
            c = 2
        cls[pid] = c
    return cls


def pad_charge(pid_H, pq_H, pid_L, pq_L, n_pads):
    q = np.bincount(np.asarray(pid_H, int), weights=pq_H, minlength=n_pads)
    q += np.bincount(np.asarray(pid_L, int), weights=pq_L, minlength=n_pads)
    return q


def span(pt):
    return float(pt.max() - pt.min()) if len(pt) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("--pads", default="my_pad.txt")
    ap.add_argument("-n", type=int, default=5000)
    ap.add_argument("--fwhm", type=float, default=2000.0, help="bruit sur charge pad, FWHM en e-")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="diag_t2f")
    a = ap.parse_args()

    sigma = a.fwhm / 2.3548
    rng = np.random.default_rng(a.seed)
    pads = np.loadtxt(a.pads, comments="#")

    with h5py.File(a.h5, "r") as f:
        md = f["/metadata"].attrs
        n_pads = int(md["n_pads"])
        W = float(md["W"])
        r_trans = float(md["r_transition"])
        N = min(a.n, f["/H/theta"].shape[0])
        theta = f["/H/theta"][:N]
        phi = f["/H/phi"][:N]
        d = {}
        for fr in ("H", "L"):
            for k in ("pid", "pq", "pt", "dE"):
                d[fr, k] = f[f"/{fr}/{k}"][:N]

    cls = classify(pads, r_trans, n_pads)
    uy = np.cos(theta)
    uz = np.sin(theta) * np.sin(phi)
    ratio = np.abs(uz) / np.abs(uy)          # |uz|/|uy|

    res = {k: [] for k in ["q_ratio", "f_cart", "f_trans", "f_ring", "n_touched",
                           "dt_H", "dt_L"]}
    for t in THRESH:
        res[f"n_kept_{t}"] = []
        res[f"n_false_{t}"] = []
        res[f"f_kept_{t}"] = []

    for i in range(N):
        q = pad_charge(d["H", "pid"][i], d["H", "pq"][i],
                       d["L", "pid"][i], d["L", "pq"][i], n_pads)
        q_tot = q.sum()
        q_exp = (d["H", "dE"][i].sum() + d["L", "dE"][i].sum()) / W
        res["q_ratio"].append(q_tot / q_exp if q_exp > 0 else np.nan)
        fc = np.bincount(cls, weights=q, minlength=3) / q_tot
        res["f_cart"].append(fc[0]); res["f_trans"].append(fc[1]); res["f_ring"].append(fc[2])
        res["n_touched"].append(int((q > 0).sum()))
        res["dt_H"].append(span(d["H", "pt"][i]))
        res["dt_L"].append(span(d["L", "pt"][i]))

        qn = q + rng.normal(0.0, sigma, n_pads)
        for t in THRESH:
            kept = qn > t * sigma
            res[f"n_kept_{t}"].append(int(kept.sum()))
            res[f"n_false_{t}"].append(int((kept & (q == 0)).sum()))
            res[f"f_kept_{t}"].append(q[kept].sum() / q_tot)

    R = {k: np.array(v) for k, v in res.items()}
    np.savez(a.out + ".npz", theta=theta, phi=phi, uy=uy, uz=uz, ratio=ratio, **R)

    # ------------------------------ resume ------------------------------
    def pct(x):
        return "med=%.3g  [5%%,95%%]=[%.3g, %.3g]" % tuple(np.nanpercentile(x, [50, 5, 95]))

    print(f"\n{N} evenements, sigma_bruit = {sigma:.0f} e- rms, {n_pads} pads")
    print("Conservation charge (sum pq / sum dE/W) :", pct(R["q_ratio"]))
    print("Fraction charge quinconce   :", pct(R["f_cart"]))
    print("Fraction charge transition  :", pct(R["f_trans"]))
    print("Fraction charge anneaux     :", pct(R["f_ring"]))
    print("Pads touches                :", pct(R["n_touched"]))
    for t in THRESH:
        print(f"Seuil {t:.0f} sigma ({t*sigma:.0f} e-) : pads gardes {pct(R[f'n_kept_{t}'])} | "
              f"faux pads moy={R[f'n_false_{t}'].mean():.2f} | "
              f"charge gardee {pct(R[f'f_kept_{t}'])}")
    for r0 in (0.1, 0.2, 0.5, 1.0):
        print(f"Evenements avec |uz|/|uy| > {r0} : {100*np.mean(ratio > r0):.1f} %")
    print("Etalement temporel H (ns)   :", pct(R["dt_H"]))

    # ------------------------------ figures ------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0, 0].hist(R["q_ratio"], bins=100); ax[0, 0].set_title("sum pq / sum(dE)/W")
    ax[0, 1].scatter(ratio, R["f_ring"] + R["f_trans"], s=2, alpha=0.3)
    ax[0, 1].set_xscale("log"); ax[0, 1].set_xlabel("|uz|/|uy|")
    ax[0, 1].set_title("fraction charge transition+anneaux")
    ax[0, 2].hist(R["n_touched"], bins=60); ax[0, 2].set_title("pads touches (sans bruit)")
    ax[1, 0].scatter(np.abs(uz), R["dt_H"], s=2, alpha=0.3)
    ax[1, 0].set_xlabel("|uz|"); ax[1, 0].set_ylabel("etalement temps H (ns)")
    for t in THRESH:
        ax[1, 1].hist(R[f"f_kept_{t}"], bins=100, histtype="step", label=f"{t:.0f} sigma")
    ax[1, 1].set_title("fraction de charge au-dessus du seuil"); ax[1, 1].legend()
    ax[1, 2].hist(ratio, bins=np.logspace(-2, 1.5, 60)); ax[1, 2].set_xscale("log")
    ax[1, 2].set_title("|uz|/|uy|")
    fig.tight_layout()
    fig.savefig(a.out + ".pdf")
    print(f"\nFigures : {a.out}.pdf   Donnees : {a.out}.npz")


if __name__ == "__main__":
    main()
