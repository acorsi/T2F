#!/usr/bin/env python3
"""Trace les cartes de pads produites par dump_patterns.

Usage :  python3 plot_patterns.py [patterns.csv] [patterns.png]

Chaque pad a une courbe Psi(tau) : la charge accumulée sur ce pad (en fraction de la
charge du nuage) en fonction du temps tau depuis l'entrée du nuage dans le gap.
Chaque case résume cette courbe par UN nombre. Une colonne par valeur de d :
  1. Psi final         : valeur de la courbe à la fin (charge collectée) ;
  2. Psi maximal       : valeur la plus haute atteinte par la courbe, à n'importe quel instant ;
  3. instant du max    : tau (ns) auquel ce maximum est atteint (gris si Psi max < 1e-3) ;
  4. qt max            : plus grand signal positif sur un bin de 10 ns (pente de montée) ;
  5. -qt min           : plus grand signal négatif sur un bin de 10 ns (pente de descente,
                         induction pure sur les voisins).
La croix rouge marque le centre du nuage.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import PowerNorm, Normalize

fichier = sys.argv[1] if len(sys.argv) > 1 else "patterns.csv"
sortie = sys.argv[2] if len(sys.argv) > 2 else "patterns.png"
PAD = 0.3      # côté du pad (cm)
SEUIL_INSTANT = 1.0e-3   # en dessous, l'instant du maximum n'a pas de sens

with open(fichier) as f:
    x0, y0 = [float(s) for s in f.readline().split(":")[1].split()]
data = np.loadtxt(fichier, delimiter=",", skiprows=2)
ds = sorted(set(data[:, 0]))

# (colonne, titre, colormap, signe, type d'échelle)
lignes = [
    (3, "Ψ final (charge collectée)", "Blues", 1.0, "racine"),
    (4, "Ψ maximal (sur toute la durée)", "Blues", 1.0, "racine"),
    (5, "instant du maximum de Ψ (ns)", "plasma", 1.0, "lineaire"),
    (6, "qt max (bin de 10 ns, lobe positif)", "Blues", 1.0, "racine"),
    (7, "−qt min (bin de 10 ns, lobe négatif)", "Oranges", -1.0, "racine"),
]

fig, axes = plt.subplots(len(lignes), len(ds), figsize=(3.3 * len(ds), 3.2 * len(lignes)),
                         squeeze=False)
for c, d in enumerate(ds):
    sel = data[data[:, 0] == d]
    for r, (col, titre, nom_cmap, signe, echelle) in enumerate(lignes):
        ax = axes[r][c]
        vals = signe * sel[:, col]
        patches = [Rectangle((x * 10 - PAD * 5, y * 10 - PAD * 5), PAD * 10, PAD * 10)
                   for x, y in zip(sel[:, 1], sel[:, 2])]
        cmap = matplotlib.colormaps[nom_cmap].copy()
        cmap.set_bad("0.85")
        if echelle == "lineaire":
            masque = sel[:, 4] < SEUIL_INSTANT
            vals = np.ma.masked_where(masque, vals)
            norm = Normalize(vmin=vals.min(), vmax=vals.max())
        else:
            vals = np.clip(vals, 0.0, None)
            norm = PowerNorm(0.5, vmin=0.0, vmax=max(vals.max(), 1e-6))
        pc = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor="0.7", linewidth=0.3)
        pc.set_array(vals)
        ax.add_collection(pc)
        ax.plot([x0 * 10], [y0 * 10], "r+", markersize=10, markeredgewidth=1.5)
        lim = 3.6 * 3.15
        ax.set_xlim(x0 * 10 - lim, x0 * 10 + lim)
        ax.set_ylim(y0 * 10 - lim, y0 * 10 + lim)
        ax.set_aspect("equal")
        ax.set_title(f"d = {d:g} cm\n{titre}" if r == 0 else titre, fontsize=9)
        if c == 0:
            ax.set_ylabel("y (mm)")
        if r == len(lignes) - 1:
            ax.set_xlabel("x (mm)")
        cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7)

fig.suptitle("Chaque case résume la courbe Ψ(τ) du pad par un nombre "
             "(échelle de couleur en racine, sauf l'instant)", fontsize=11)
fig.tight_layout()
fig.savefig(sortie, dpi=100)
print("Ecrit :", sortie)
