#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Niveau 0 : compare le potentiel de ponderation du code a une reference EXACTE
# calculee independamment (series de Fourier), et trace simu versus exact.
#
# Modele : tranche 0 < h < g, grille a phi = 0 en h = g, anode en h = 0 avec phi = 1 sur le pad
# (tuile carree de demi-cote a) et 0 ailleurs.  Solution exacte :
#
#   phi(u,v,h) = (4/pi^2) Int_0^inf Int_0^inf  [sin(kx a) cos(kx u)/kx] [sin(ky a) cos(ky v)/ky]
#                                              * sinh(k (g-h)) / sinh(k g)   dkx dky ,   k = |(kx,ky)|
#
# integree par la regle du point milieu (exponentiellement precise ici : integrande analytique,
# a decroissance rapide ; l'erreur vient des images periodiques a 2 pi/dk = 12 cm).
# Cette reference n'utilise ni la methode des images ni les angles solides.
# ---------------------------------------------------------------------------
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOL = 1.0e-3          # tolerance ABSOLUE sur phi (phi est une fraction de charge, de 0 a 1)
DK = 0.5              # pas en k (cm^-1) ; verifie en le divisant par 2 plus bas
OUT = "out/"


def read_csv(name):
    with open(OUT + name) as fh:
        return list(csv.DictReader(fh))


class FourierRef:
    """Pour une geometrie (a, g) et une hauteur h : matrice F(k) calculee une fois, puis phi(u,v) bon marche."""

    def __init__(self, a, g, h, dk=DK):
        self.a, self.dk = a, dk
        kmax = 28.0 / h                       # exp(-k h) < 1e-12
        n = int(kmax / dk) + 1
        self.k = (np.arange(n) + 0.5) * dk
        K = np.hypot(self.k[:, None], self.k[None, :])
        # sinh(k(g-h))/sinh(kg), ecrit sans debordement
        self.F = np.exp(-K * h) * (1.0 - np.exp(-2.0 * K * (g - h))) / (1.0 - np.exp(-2.0 * K * g))

    def phi(self, u, v):
        A = np.sin(self.k * self.a) * np.cos(self.k * u) / self.k
        B = np.sin(self.k * self.a) * np.cos(self.k * v) / self.k
        return 4.0 / np.pi ** 2 * self.dk ** 2 * (A @ self.F @ B)


_cache = {}


def ref_phi(a, g, h, u, v):
    key = (round(a, 8), round(g, 8), round(h, 10))
    if key not in _cache:
        _cache[key] = FourierRef(a, g, h)
    return _cache[key].phi(u, v)


results = []     # (nom, erreur max, tolerance, ok)


def report(name, err, tol=TOL):
    ok = err < tol
    results.append((name, err, tol, ok))


# ---------------------------------------------------------------------------
# 0. validite de la reference : dk et dk/2 doivent donner la meme chose
# ---------------------------------------------------------------------------
a0, g0 = 0.2575, 0.3
r1 = FourierRef(a0, g0, 0.2 * g0, DK).phi(0.3, 0.1)
r2 = FourierRef(a0, g0, 0.2 * g0, DK / 2).phi(0.3, 0.1)
print("validite de la reference de Fourier : |dk - dk/2| = %.2e (doit etre << 1e-3)" % abs(r1 - r2))
ref_ok = abs(r1 - r2) < 1e-8

# ---------------------------------------------------------------------------
# 1. phi : code contre reference
# ---------------------------------------------------------------------------
rows = read_csv("l0_phi.csv")
data = {}
for r in rows:
    r["a"], r["g"], r["u"], r["v"], r["h"], r["phi"] = [float(r[k]) for k in ("a", "g", "u", "v", "h", "phi")]
    r["n_img"] = int(r["n_img"])
    r["ref"] = float(ref_phi(r["a"], r["g"], r["h"], r["u"], r["v"]))
    r["err"] = r["phi"] - r["ref"]

pt = [r for r in rows if r["tag"] == "pt"]
sc = [r for r in rows if r["tag"] == "scan"]
ni = [r for r in rows if r["tag"] == "nimg"]

for geo in "ABC":
    e = max(abs(r["err"]) for r in pt if r["geo"] == geo)
    report("phi_point vs Fourier, points isoles, geometrie %s (n_img=20)" % geo, e)
    e = max(abs(r["err"]) for r in sc if r["geo"] == geo)
    report("phi_point vs Fourier, balayage en u, geometrie %s" % geo, e)

# ---------------------------------------------------------------------------
# 2. regle de somme
# ---------------------------------------------------------------------------
srows = read_csv("l0_sum.csv")
for r in srows:
    r["hfrac"], r["sum"] = float(r["hfrac"]), float(r["sum"])
    r["err"] = r["sum"] - (1.0 - r["hfrac"])
for geo in "ABC":
    e = max(abs(r["err"]) for r in srows if r["geo"] == geo)
    report("regle de somme sur le pavage = 1 - h/g, geometrie %s" % geo, e)

# ---------------------------------------------------------------------------
# 3. cas triviaux
# ---------------------------------------------------------------------------
mrows = read_csv("l0_misc.csv")
for r in mrows:
    report("cas simple : " + r["name"], abs(float(r["sim"]) - float(r["exact"])))

# ---------------------------------------------------------------------------
# 4. convergence en nombre d'images
# ---------------------------------------------------------------------------
nimg_err = {}
for r in ni:
    nimg_err.setdefault((r["geo"], r["n_img"]), []).append(abs(r["err"]))

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}
labels = {"A": "A : pad 5 mm, gap 3 mm", "B": "B : pad 3 mm, gap 3 mm", "C": "C : pad 5 mm, gap 5 mm"}

fig, ax = plt.subplots(2, 3, figsize=(16, 8), sharex="col", gridspec_kw={"height_ratios": [2, 1]})

# (a) phi au centre du pad en fonction de h/g
for geo in "ABC":
    s = sorted([r for r in pt if r["geo"] == geo and r["u"] == 0 and r["v"] == 0], key=lambda r: r["h"])
    x = [r["h"] / r["g"] for r in s]
    ax[0, 0].plot(x, [r["ref"] for r in s], "-", color=colors[geo], label="exact " + labels[geo])
    ax[0, 0].plot(x, [r["phi"] for r in s], "o", color=colors[geo], mfc="none", ms=7, label="simu " + geo)
    ax[1, 0].plot(x, [r["err"] for r in s], "o-", color=colors[geo], ms=4)
ax[0, 0].set_ylabel("phi au centre du pad")
ax[0, 0].set_title("phi(u=0, v=0) en fonction de la hauteur")
ax[0, 0].legend(fontsize=7)
ax[1, 0].set_xlabel("h / g")
ax[1, 0].set_ylabel("simu - exact")

# (b) balayage en u (geometrie A), trois hauteurs
for hf, ls in zip((0.2, 0.5, 0.8), ("tab:red", "tab:purple", "tab:brown")):
    s = [r for r in sc if r["geo"] == "A" and abs(r["h"] / r["g"] - hf) < 1e-9]
    s.sort(key=lambda r: r["u"])
    x = [r["u"] / r["a"] for r in s]
    ax[0, 1].plot(x, [r["ref"] for r in s], "-", color=ls, label="exact h=%.1f g" % hf)
    ax[0, 1].plot(x, [r["phi"] for r in s], ".", color=ls, ms=6)
    ax[1, 1].plot(x, [r["err"] for r in s], ".-", color=ls, ms=4)
ax[0, 1].set_title("phi(u, v=0) a h fixe, geometrie A  (points = simu)")
ax[0, 1].legend(fontsize=7)
ax[1, 1].set_xlabel("u / a")

# (c) regle de somme
for geo in "ABC":
    s = [r for r in srows if r["geo"] == geo]
    ax[0, 2].plot([r["hfrac"] for r in s], [r["sum"] for r in s], ".", color=colors[geo], ms=4, label="simu " + geo)
    ax[1, 2].plot([r["hfrac"] for r in s], [r["err"] for r in s], ".", color=colors[geo], ms=4)
hh = np.linspace(0, 1, 50)
ax[0, 2].plot(hh, 1 - hh, "k-", label="exact 1 - h/g")
ax[0, 2].set_title("somme de phi sur tout le pavage")
ax[0, 2].legend(fontsize=7)
ax[1, 2].set_xlabel("h / g")

for a_ in ax[1, :]:
    a_.axhline(TOL, color="gray", ls="--", lw=0.8)
    a_.axhline(-TOL, color="gray", ls="--", lw=0.8)
    a_.axhline(0, color="k", lw=0.5)
    a_.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
fig.suptitle("Niveau 0 : potentiel de ponderation, simu versus exact   (traits gris : tolerance +-1e-3)")
fig.tight_layout()
fig.savefig(OUT + "l0_phi.png", dpi=110)

# convergence en n_img
fig2, ax2 = plt.subplots(figsize=(6, 4))
for geo in "ABC":
    ns = sorted(set(n for (g_, n) in nimg_err if g_ == geo))
    ax2.semilogy(ns, [max(1e-17, max(nimg_err[(geo, n)])) for n in ns], "o-", color=colors[geo], label=labels[geo])
ax2.axhline(TOL, color="gray", ls="--")
ax2.set_xscale("log")
ax2.set_xlabel("n_img")
ax2.set_ylabel("max |simu - exact|")
ax2.set_title("Convergence de phi_point en nombre d'images")
ax2.legend(fontsize=7)
fig2.tight_layout()
fig2.savefig(OUT + "l0_nimg.png", dpi=110)

# ---------------------------------------------------------------------------
# Bilan
# ---------------------------------------------------------------------------
print("\n%-78s %10s  %s" % ("test", "ecart max", ""))
nfail = 0
for name, err, tol, ok in results:
    print("%-78s %10.2e  %s" % (name, err, "OK" if ok else "ECHEC (tolerance %.0e)" % tol))
    nfail += (not ok)
print("\nconvergence en n_img (ecart max sur les points de test) :")
for geo in "ABC":
    ns = sorted(set(n for (g_, n) in nimg_err if g_ == geo))
    print("  %s : " % geo + "  ".join("n=%d: %.1e" % (n, max(nimg_err[(geo, n)])) for n in ns))
print("\nreference de Fourier valide : %s" % ("oui" if ref_ok else "NON (ne pas se fier au bilan)"))
print("figures : %sl0_phi.png  %sl0_nimg.png" % (OUT, OUT))
print("BILAN NIVEAU 0 : %s" % ("tout est OK" if nfail == 0 and ref_ok else "%d echec(s)" % nfail))
sys.exit(0 if nfail == 0 and ref_ok else 2)
