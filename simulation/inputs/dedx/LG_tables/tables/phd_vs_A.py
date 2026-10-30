#!/usr/bin/env python3
"""
PHD en fonction de A a partir de <KE>(A) digitalise, avec Z = A / 2.57.

Entrees
  ke_A.dat       : 2 colonnes "A  <KE>(MeV)"  (digitaliseur, '#' = commentaire ; --swap si les colonnes sont inversees)
  peck_A.dat     : "A  PHD(MeV)" sur A entiers (sortie de peck_interp.py)
  gh_tables/     : gh_<Z>.dat (sortie de make_gh_tables.py : KE/A, g_Z, h_Z)

Pour chaque A entier :  Z = A/2.57 (non entier : g est interpole lineairement entre les Z voisins)
  Sn_int = A*g_Z(KE/A),  Se_tot = KE - Sn_int  (car g + h = e),  NI = a*Sn_int^b
  residu = PHD_Peck - NI,  ajuste par  alpha*(1-exp(-lambda*Se_tot)).
Sortie : phd_vs_A.dat, phd_vs_A.png, parametres alpha/lambda, rms, leave-one-out.

Usage : python3 phd_vs_A.py ke_A.dat [--peck peck_A.dat] [--gh gh_tables] [--a 0.7475] [--b 0.8131]
                                     [--ratio 2.57] [--method linear|pchip] [--step 4] [--swap]
"""
import argparse, glob, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

ap = argparse.ArgumentParser()
ap.add_argument("ke"); ap.add_argument("--peck", default="peck_A.dat"); ap.add_argument("--gh", default="gh_tables")
ap.add_argument("--a", type=float, default=0.7475); ap.add_argument("--b", type=float, default=0.8131)
ap.add_argument("--ratio", type=float, default=2.57, help="A/Z")
ap.add_argument("--method", default="linear", choices=["linear", "pchip"])
ap.add_argument("--step", type=int, default=4, help="un A sur 'step' pour l'ajustement et le leave-one-out (points moins correles)")
ap.add_argument("--swap", action="store_true"); ap.add_argument("--out", default="phd_vs_A")
args = ap.parse_args()

# ---- <KE>(A) digitalise -> A entiers
def read2(path):
    """Lit 2 colonnes (espaces, tabulations ou virgules) ; ignore les commentaires '#' et un en-tete texte."""
    rows = []
    for line in open(path):
        t = line.split("#")[0].replace(",", " ").replace(";", " ").split()
        if len(t) >= 2:
            try: rows.append((float(t[0]), float(t[1])))
            except ValueError: pass          # ligne d'en-tete
    return np.array(rows).T
x, y = read2(args.ke)
if args.swap: x, y = y, x
o = np.argsort(x); x, y = x[o], y[o]
if np.any(np.diff(x) <= 0): sys.exit("A non strictement croissant apres tri : doublons dans le fichier ?")
if x.min() < 30 or x.max() > 250 or y.min() < 20 or y.max() > 250:
    print(f"[!] plages inattendues : A in [{x.min():.0f},{x.max():.0f}], KE in [{y.min():.0f},{y.max():.0f}] -> --swap ?", file=sys.stderr)

# ---- PHD de Peck
pa, pphd = read2(args.peck)

# ---- tables g_Z
gh = {}
for f in glob.glob(os.path.join(args.gh, "gh_*.dat")):
    Z = int(re.search(r"gh_(\d+)", f).group(1)); d = np.loadtxt(f); gh[Z] = (d[:, 0], d[:, 1])
if not gh: sys.exit(f"aucune table gh_*.dat dans {args.gh}")
Zs = np.array(sorted(gh))
def g_of(Z, e):
    """g(Z,e) : interpolation lineaire en Z entre les deux tables disponibles qui encadrent Z ;
    au-dela de e_max, g est plafonne a g(e_max) (le nucleaire est depose a basse energie)."""
    Z = np.asarray(Z, float); e = np.asarray(e, float)
    k = np.clip(np.searchsorted(Zs, Z, side="right") - 1, 0, len(Zs) - 2)
    out = np.empty_like(e)
    for i in range(len(Z)):
        Za, Zb = Zs[k[i]], Zs[k[i] + 1]; w = (Z[i] - Za) / (Zb - Za)
        out[i] = (1 - w) * np.interp(e[i], *gh[Za]) + w * np.interp(e[i], *gh[Zb])
    return out

lo = max(int(np.ceil(x.min())), int(pa.min()))
hi = min(int(np.floor(x.max())), int(pa.max()))
A = np.array([a_ for a_ in range(lo, hi + 1) if Zs.min() <= a_ / args.ratio <= Zs.max()])
if len(A) == 0: sys.exit("aucun A commun aux fichiers et aux tables (Z=A/ratio hors de la plage des tables)")
if len(A) < hi - lo + 1:
    print(f"[i] A restreint a {A.min()}..{A.max()} (tables gh disponibles pour Z={Zs.min()}..{Zs.max()})", file=sys.stderr)

if args.method == "linear": KE = np.interp(A, x, y)
else:
    from scipy.interpolate import PchipInterpolator; KE = PchipInterpolator(x, y)(A)
Z = A / args.ratio
PHD = np.interp(A, pa, pphd)

Sn = A * g_of(Z, KE / A)
Se = KE - Sn
NI = args.a * Sn ** args.b
R = PHD - NI

# ---- ajustement du residuel
model = lambda s, al, lam: al * (1 - np.exp(-lam * s))
fit_e = lambda s, r: curve_fit(model, s, r, p0=[r.max() * 1.05, 0.03], maxfev=20000)[0]
fit_c = lambda s, r: np.array([r.mean()])
idx = np.arange(0, len(A), args.step)
p = fit_e(Se[idx], R[idx]); c = fit_c(Se[idx], R[idx])
def loo(fit, pred):
    out = []
    for k in range(len(idx)):
        m = np.delete(idx, k)
        out.append(R[idx[k]] - pred(fit(Se[m], R[m]), Se[idx[k]]))
    return np.array(out)
rms = lambda v: 1e3 * np.sqrt(np.mean(np.asarray(v) ** 2))
loo_e = loo(fit_e, lambda q, s: model(s, *q)); loo_c = loo(fit_c, lambda q, s: q[0])
PHD_mod = NI + model(Se, *p)
res = PHD - PHD_mod

print(f"A = {A.min()}..{A.max()}  ({len(A)} masses, {len(idx)} points pour l'ajustement, 1 sur {args.step})")
print(f"Z = A/{args.ratio} ; KE in [{KE.min():.1f}, {KE.max():.1f}] MeV ; Sn_int in [{Sn.min():.2f}, {Sn.max():.2f}] MeV ; Se_tot in [{Se.min():.0f}, {Se.max():.0f}] MeV")
print(f"residu PHD_Peck - NI : de {R.min():.3f} a {R.max():.3f} MeV")
print(f"exponentielle : alpha = {p[0]:.3f} MeV, lambda = {p[1]:.4f} MeV^-1")
print(f"   rms (tous les A) = {rms(res):.0f} keV, max = {1e3*np.abs(res).max():.0f} keV ; leave-one-out (sous-echantillon) = {rms(loo_e):.0f} keV")
print(f"constante {c[0]:.3f} MeV : rms (tous les A) = {rms(R - c[0]):.0f} keV ; leave-one-out = {rms(loo_c):.0f} keV")

np.savetxt(args.out + ".dat", np.column_stack([A, Z, KE, Sn, Se, NI, PHD, R, PHD_mod]), fmt="%d %.3f %.3f %.4f %.3f %.4f %.4f %.4f %.4f",
           header="A  Z=A/%.2f  KE  Sn_int  Se_tot  NI  PHD_Peck  resid=PHD-NI  PHD_model(alpha=%.3f,lambda=%.4f)" % (args.ratio, p[0], p[1]))

fig, ax = plt.subplots(2, 2, figsize=(11, 8)); ax = ax.ravel()
ax[0].plot(x, y, "s", mfc="none", label="digitalisé"); ax[0].plot(A, KE, ".", ms=3, label="A entiers")
ax[0].set_xlabel("A"); ax[0].set_ylabel("<KE> (MeV)"); ax[0].legend()
xs = np.linspace(0, Se.max() * 1.1, 200)
ax[1].plot(Se, R, ".", ms=4, label="PHD$_{Peck}$ − NI"); ax[1].plot(Se[idx], R[idx], "ko", mfc="none", label="points ajustés")
ax[1].plot(xs, model(xs, *p), "r-", label=f"{p[0]:.2f}[1−exp(−{p[1]:.3f}S$_e$)]"); ax[1].axhline(c[0], color="gray", ls="--", label=f"constante {c[0]:.2f}")
ax[1].set_xlabel("S$_e^{Tot}$ (MeV)"); ax[1].set_ylabel("PHD résiduel (MeV)"); ax[1].legend()
ax[2].plot(A, PHD, "k-", label="PHD Peck"); ax[2].plot(A, NI, "b--", label="NI"); ax[2].plot(A, PHD_mod, "r-", label="NI + résiduel")
ax[2].set_xlabel("A"); ax[2].set_ylabel("PHD (MeV)"); ax[2].legend()
ax[3].plot(A, 1e3 * res, "r-"); ax[3].axhline(0, color="k", lw=.8); ax[3].set_xlabel("A"); ax[3].set_ylabel("PHD$_{Peck}$ − modèle (keV)")
for a_ in ax: a_.grid(alpha=.3)
fig.tight_layout(); fig.savefig(args.out + ".png", dpi=130); print("->", args.out + ".dat,", args.out + ".png")
