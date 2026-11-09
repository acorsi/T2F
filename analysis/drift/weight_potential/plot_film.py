#!/usr/bin/env python3
"""Film du signal induit en fonction du temps, produit a partir de dump_film.

Usage :  python3 plot_film.py [film.csv] [film.gif]

Une colonne par valeur de d. Le temps tau est compte depuis l'entree du centre du nuage
dans le gap (tau = 0). Le film avance par pas de 10 ns (un bin du detecteur).
  ligne 1 : charge cumulee Psi(tau) de chaque pad (blanc = 0), echelle en racine ;
  ligne 2 : signal qt du bin [tau, tau+10 ns] (blanc = 0, rouge > 0, bleu < 0),
            echelle en puissance 0,35 pour laisser voir les lobes negatifs ;
  ligne 3 : courbes Psi(tau) (echelle lineaire) des trois memes pads ;
  ligne 4 : courbes qt(tau) de trois pads (contours de meme couleur sur les cartes) :
            le pad le plus proche du voxel, son plus proche voisin, et un pad a ~2 pitchs.
            Echelle symlog (lineaire sous 1e-4). Trait vertical = instant courant.
La croix noire marque le centre du nuage.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import PowerNorm, FuncNorm
from matplotlib.animation import FuncAnimation, PillowWriter

fichier = sys.argv[1] if len(sys.argv) > 1 else "film.csv"
sortie = sys.argv[2] if len(sys.argv) > 2 else "film.gif"
PAD = 0.3          # cote du pad (cm)
PITCH = 0.315
BIN_NS = 10.0
COULEURS = ["tab:green", "tab:orange", "tab:purple"]
NOMS = ["pad le plus proche", "plus proche voisin", "pad a ~2 pitchs"]

with open(fichier) as f:
    x0, y0 = [float(s) for s in f.readline().split(":")[1].split()]
    tau0, dtau, n_tau = f.readline().split(":")[1].split()
    tau0, dtau, n_tau = float(tau0), float(dtau), int(n_tau)
data = np.loadtxt(fichier, delimiter=",", skiprows=2)
ds = sorted(set(data[:, 0]))

pas = int(round(BIN_NS / dtau))              # noeuds par bin
instants = np.arange(0, n_tau - pas, pas)    # indice de depart de chaque bin
taus = tau0 + instants * dtau

# --- pads mis en evidence (les memes pour toutes les colonnes) ---
sel0 = data[data[:, 0] == ds[0]]
dist = np.hypot(sel0[:, 1] - x0, sel0[:, 2] - y0)
i_a = int(np.argmin(dist))
ordre = np.argsort(dist)
i_b = int(ordre[1])
i_c = int(np.argmin(np.abs(dist - 2.0 * PITCH) + (np.arange(len(dist)) == i_a) * 10.0))
choisis = [i_a, i_b, i_c]

fig, axes = plt.subplots(4, len(ds), figsize=(3.0 * len(ds), 12.4), squeeze=False)
colls1, colls2, curseurs = [], [], []
donnees = []
lim = 3.6 * 3.15
for c, d in enumerate(ds):
    sel = data[data[:, 0] == d]
    psi = sel[:, 3:]
    qt = psi[:, instants + pas] - psi[:, instants]      # (n_pads, n_bins)
    donnees.append((psi, qt))

    for r in (0, 1):
        ax = axes[r][c]
        patches = [Rectangle((x * 10 - PAD * 5, y * 10 - PAD * 5), PAD * 10, PAD * 10)
                   for x, y in zip(sel[:, 1], sel[:, 2])]
        if r == 0:
            norm = PowerNorm(0.5, vmin=0.0, vmax=max(psi.max(), 1e-6))
            cmap = "Blues"
        else:
            vm = max(np.abs(qt).max(), 1e-6)
            fw = lambda x: np.sign(x) * np.abs(x) ** 0.35
            bw = lambda x: np.sign(x) * np.abs(x) ** (1.0 / 0.35)
            norm = FuncNorm((fw, bw), vmin=-vm, vmax=vm)
            cmap = "RdBu_r"
        pc = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor="0.75", linewidth=0.3)
        pc.set_array(np.zeros(len(patches)))
        ax.add_collection(pc)
        (colls1 if r == 0 else colls2).append(pc)
        for k, i in enumerate(choisis):
            ax.add_patch(Rectangle((sel[i, 1] * 10 - PAD * 5, sel[i, 2] * 10 - PAD * 5),
                                   PAD * 10, PAD * 10, fill=False,
                                   edgecolor=COULEURS[k], linewidth=2.0))
        ax.plot([x0 * 10], [y0 * 10], "k+", markersize=10, markeredgewidth=1.5)
        ax.set_xlim(x0 * 10 - lim, x0 * 10 + lim)
        ax.set_ylim(y0 * 10 - lim, y0 * 10 + lim)
        ax.set_aspect("equal")
        if r == 0:
            ax.set_title(f"d = {d:g} cm\ncharge cumulee Ψ" if True else "", fontsize=10)
        else:
            ax.set_title("signal qt du bin", fontsize=10)
        if c == 0:
            ax.set_ylabel("y (mm)")
        fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)

    # ligne 3 : charge cumulee Psi(tau), echelle lineaire
    ax = axes[2][c]
    for k, i in enumerate(choisis):
        ax.plot(tau0 + np.arange(n_tau) * dtau, psi[i], color=COULEURS[k])
    ax.set_xlabel("τ (ns)")
    if c == 0:
        ax.set_ylabel("charge cumulée Ψ")
    curseurs.append(ax.axvline(taus[0], color="k", linewidth=1.0))

    # ligne 4 : qt(tau), echelle symlog
    ax = axes[3][c]
    for k, i in enumerate(choisis):
        ax.plot(taus + BIN_NS, qt[i], color=COULEURS[k], label=NOMS[k] if c == 0 else None)
    ax.axhline(0.0, color="0.6", linewidth=0.6)
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xlabel("τ (ns)")
    if c == 0:
        ax.set_ylabel("qt par bin de 10 ns")
        ax.legend(fontsize=7, loc="upper left")
    curseurs.append(ax.axvline(taus[0], color="k", linewidth=1.0))

titre = fig.suptitle("", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))


def maj(k):
    for c in range(len(ds)):
        psi, qt = donnees[c]
        colls1[c].set_array(psi[:, instants[k] + pas])      # charge a la fin du bin
        colls2[c].set_array(qt[:, k])
        for cur in curseurs[2 * c:2 * c + 2]:
            cur.set_xdata([taus[k] + BIN_NS, taus[k] + BIN_NS])
    titre.set_text("bin τ = %.0f à %.0f ns  (τ = 0 : le centre du nuage entre dans le gap)"
                   % (taus[k], taus[k] + BIN_NS))
    return []


anim = FuncAnimation(fig, maj, frames=len(instants), blit=False)
anim.save(sortie, writer=PillowWriter(fps=6), dpi=65)
print("Ecrit :", sortie, "(%d images)" % len(instants))
