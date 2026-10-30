#!/usr/bin/env python3
"""
Ajustement du PHD residuel : dPHD(Se_tot) = alpha * (1 - exp(-lambda * Se_tot)).

dPHD = PHD_Peck - NI,  NI = a * Sn_int^b  (a, b issus de ni_fit.py, avec les tables T2F)
Sn_int = A*g_Z(E0/A), Se_tot = A*h_Z(E0/A)  (memes tables dedx_<Z>.lg que ni_fit.py)

Fichier d'entree (peck.dat), une ligne par fragment :
    # label  Z  A  E0(MeV)  PHD(MeV)
    106Zr    40 106 100.0    4.25
E0 = energie du fragment utilisee dans [Peck19] pour cette mesure.

Usage : python3 resid_fit.py peck.dat [--dir .] [--a 0.7475] [--b 0.8131]
Necessite ni_fit.py dans le meme dossier.
"""
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import ni_fit as nf

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("--dir", default=".")
ap.add_argument("--a", type=float, default=0.7475); ap.add_argument("--b", type=float, default=0.8131)
ap.add_argument("--out", default="resid_fit.png")
args = ap.parse_args()

rows = []
for line in open(args.data):
    s = line.split("#")[0].split()
    if len(s) >= 5:
        rows.append((s[0], int(s[1]), int(s[2]), float(s[3]), float(s[4])))
lab, Sn, Se, Phd = [], [], [], []
for name, Z, A, E0, phd in rows:
    f = os.path.join(args.dir, f"dedx_{Z}.lg")
    if not os.path.exists(f):
        print(f"[!] {name}: {f} introuvable, ignore", file=sys.stderr); continue
    e, se, sn = nf.read_dedx(f)
    gn, ge = nf.make_g(e, se, sn), nf.make_g(e, sn, se)
    if E0 / A > e.max():
        print(f"[!] {name}: E0/A={E0/A:.3f} > table ({e.max():.3f})", file=sys.stderr)
    lab.append(name); Sn.append(A * gn(E0 / A)); Se.append(A * ge(E0 / A)); Phd.append(phd)
lab, Sn, Se, Phd = map(np.array, (lab, Sn, Se, Phd))
NI = args.a * Sn ** args.b
R = Phd - NI

model_exp = lambda s, al, lam: al * (1 - np.exp(-lam * s))
def fit_exp(s, r):
    p, _ = curve_fit(model_exp, s, r, p0=[r.max() * 1.05, 0.03], maxfev=20000); return p
def fit_const(s, r): return np.array([r.mean()])

def loo(fit, pred, s, r):
    out = np.empty_like(r)
    for i in range(len(r)):
        m = np.arange(len(r)) != i
        out[i] = r[i] - pred(fit(s[m], r[m]), s[i])
    return out

p = fit_exp(Se, R); c = fit_const(Se, R)
res_e = R - model_exp(Se, *p); res_c = R - c[0]
loo_e = loo(fit_exp, lambda q, s: model_exp(s, *q), Se, R)
loo_c = loo(fit_const, lambda q, s: q[0], Se, R)

print(f"{'frag':8s} {'Sn_int':>7s} {'Se_tot':>7s} {'PHD':>6s} {'NI':>6s} {'dPHD':>6s} {'res(keV)':>9s} {'LOO(keV)':>9s}")
for i in range(len(lab)):
    print(f"{lab[i]:8s} {Sn[i]:7.3f} {Se[i]:7.1f} {Phd[i]:6.3f} {NI[i]:6.3f} {R[i]:6.3f} {1e3*res_e[i]:9.0f} {1e3*loo_e[i]:9.0f}")
rms = lambda x: 1e3 * np.sqrt(np.mean(x ** 2))
print(f"\nexponentielle : alpha = {p[0]:.3f} MeV, lambda = {p[1]:.4f} MeV^-1")
print(f"   rms ajustement = {rms(res_e):.0f} keV   rms leave-one-out = {rms(loo_e):.0f} keV")
print(f"constante     : {c[0]:.3f} MeV")
print(f"   rms ajustement = {rms(res_c):.0f} keV   rms leave-one-out = {rms(loo_c):.0f} keV")
print(f"PHD total (NI + dPHD) : rms = {rms(res_e):.0f} keV, max = {1e3*np.abs(res_e).max():.0f} keV")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
xs = np.linspace(0, Se.max() * 1.4, 200)
ax[0].plot(Se, R, "s", mfc="none", label="PHD$_{Peck}$ − NI")
ax[0].plot(xs, model_exp(xs, *p), "-", label=f"{p[0]:.2f}[1−exp(−{p[1]:.3f} S$_e$)]")
ax[0].axhline(c[0], color="gray", ls="--", label=f"constante {c[0]:.2f}")
for x, y, l in zip(Se, R, lab): ax[0].annotate(l, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax[0].set_xlabel("S$_e^{Tot}$ (MeV)"); ax[0].set_ylabel("PHD résiduel (MeV)"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(Se, 1e3 * res_e, "s", mfc="none", label="ajustement")
ax[1].plot(Se, 1e3 * loo_e, "o", mfc="none", label="leave-one-out")
ax[1].axhline(0, color="k", lw=.8); ax[1].set_xlabel("S$_e^{Tot}$ (MeV)"); ax[1].set_ylabel("résidu (keV)")
ax[1].legend(); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(args.out, dpi=130); print("Figure :", args.out)
