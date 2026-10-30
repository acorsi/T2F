#!/usr/bin/env python3
"""
Viewer type2 : plan d'anode 2D (H+L) + profils Q(t) H et L.

Usage:
    python view_type2.py output.h5 [--event N]

Navigation:
    -> / d : event suivant
    <- / a : event précédent
    ↑ / ↓  : seuil +/- step
    q       : quitter
"""

import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.path import Path as MplPath
from shapely.geometry import Point, box
from shapely.ops import unary_union
from read_metadata import read_metadata


def _shapely_to_mpl_path(geom):
    """Convertit un Polygon/MultiPolygon shapely (avec trous éventuels)
    en un unique matplotlib.path.Path composé (gère les anneaux multiples)."""
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    verts, codes = [], []
    for poly in polys:
        for ring in [poly.exterior] + list(poly.interiors):
            coords = list(ring.coords)
            n = len(coords)
            if n < 3:
                continue
            verts.extend(coords)
            codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (n - 2) + [MplPath.CLOSEPOLY])
    return MplPath(verts, codes)


class ViewerType2:
    def __init__(self, h5_path, pad_file=None, start_event=0):
        self.f = h5py.File(h5_path, "r")
        self.n_events = self.f["/H/dE"].shape[0]
        self.event = start_event

        m = read_metadata(self.f)
        self.pad_size  = m["pad_size"]
        self.inter_pad = m["inter_pad"]
        self.delta_t   = m["delta_t"]

        self.threshold      = 100.0
        self.threshold_step = 1000.0
        self.cbar = None

        # Charge la géométrie des pads si my_pad.txt fourni/trouvé
        self.pad_geom = None
        self.trans_paths = {}  # pad_id -> matplotlib.path.Path (contour exact)
        if pad_file:
            try:
                data = np.loadtxt(pad_file)
                # colonnes : id x y r_in r_out sx sy
                self.pad_geom = {int(row[0]): row[1:] for row in data}
                self._build_transition_paths()
                self._build_background_patches()
            except OSError:
                print(f"[view_type2] pad_file '{pad_file}' introuvable — "
                      "les anneaux ne seront pas affichés correctement.")

        self.fig = plt.figure(figsize=(18, 7))
        self.ax_pad  = self.fig.add_axes([0.04, 0.08, 0.38, 0.84])
        self.ax_qt   = self.fig.add_axes([0.50, 0.08, 0.44, 0.84])
        self.cbar_ax = self.fig.add_axes([0.44, 0.15, 0.02, 0.7])

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._draw()
        plt.show()

    def _on_key(self, event):
        if event.key in ("right", "d"):
            self.event = min(self.event + 1, self.n_events - 1)
            self._draw()
        elif event.key in ("left", "a"):
            self.event = max(self.event - 1, 0)
            self._draw()
        elif event.key == "up":
            self.threshold += self.threshold_step
            self._draw()
        elif event.key == "down":
            self.threshold = max(0.0, self.threshold - self.threshold_step)
            self._draw()
        elif event.key == "q":
            plt.close(self.fig)

    def _combined_qt(self, i):
        """Construit, par pad_id, la série temporelle Q(t) combinée H+L.

        Les qt sont sommés uniquement pour les bins temporels identiques
        (même k = round(t/delta_t)) ; un bin présent dans un seul fragment
        est conservé tel quel. Utilise k plutôt que t directement pour
        éviter tout souci d'égalité en flottant.
        """
        series = {}  # pad_id -> {k: qt_sum}
        for frag in ("H", "L"):
            pid = np.array(self.f[f"/{frag}/pid"][i], dtype=int)
            pt  = np.array(self.f[f"/{frag}/pt"][i])
            pqt = np.array(self.f[f"/{frag}/pqt"][i])
            pnt = np.array(self.f[f"/{frag}/pnt"][i], dtype=int)
            if len(pnt) == 0:
                continue

            offsets = np.concatenate([[0], np.cumsum(pnt)])
            for k_pad in range(len(pnt)):
                n = pnt[k_pad]
                if n == 0:
                    continue
                pad_id = int(pid[k_pad])
                t_pad  = pt[offsets[k_pad]:offsets[k_pad] + n]
                qt_pad = pqt[offsets[k_pad]:offsets[k_pad] + n]

                bins = series.setdefault(pad_id, {})
                for t_val, q_val in zip(t_pad, qt_pad):
                    key = int(round(float(t_val) / self.delta_t))
                    bins[key] = bins.get(key, 0.0) + float(q_val)

        return series

    def _draw_qt(self, i, pad_totals):
        """Trace, pour chaque pad passant le seuil (charge totale H+L),
        son profil Q(t) combiné."""
        ax = self.ax_qt
        ax.clear()

        series = self._combined_qt(i)
        passing_ids = sorted(pid for pid, q in pad_totals.items()
                             if q >= self.threshold and pid in series)

        if not passing_ids:
            ax.text(0.5, 0.5, "Aucun signal", ha="center", va="center",
                   transform=ax.transAxes)
            ax.set_title("Q(t) combiné (H+L)")
            return

        cmap = plt.colormaps["tab20"]
        for n, pad_id in enumerate(passing_ids):
            bins = series[pad_id]
            ks = sorted(bins.keys())
            t_vals  = [k * self.delta_t for k in ks]
            qt_vals = [bins[k] for k in ks]
            ax.plot(t_vals, qt_vals, color=cmap(n % 20 / 20), lw=1.2, alpha=0.8)

        ax.set_xlabel("t (ns)")
        ax.set_ylabel("Q (e⁻)")
        ax.set_title(f"Q(t) combiné H+L  ({len(passing_ids)} pads, "
                    f"seuil>{self.threshold:.0f} e⁻)")

    def _compute_transition_shapes(self, squares_union, ring_entries, r_quinc):
        """Anneau de transition = disque complet (rayon r_transition) moins
        les pads (l'union des carrés cartésiens) — la définition la plus
        simple et la plus directe : tout ce qui n'est pas un pad, jusqu'à
        r_transition."""
        big = 1000.0  # bien plus grand que r_active, pour un demi-plan "infini"
        shapes = {}
        for pid, (x, y, r_in, r_out, sx, sy) in ring_entries.items():
            if r_in != r_quinc:
                continue  # pas un pad de transition, vrai anneau → Wedge classique
            quadrant_box = box(0 if sx > 0 else -big, 0 if sy > 0 else -big,
                               big if sx > 0 else 0, big if sy > 0 else 0)
            quarter_disk = Point(0, 0).buffer(r_out, resolution=64).intersection(quadrant_box)
            shape = quarter_disk.difference(squares_union) if squares_union else quarter_disk
            if not shape.is_empty:
                shapes[pid] = shape
        return shapes

    def _build_transition_paths(self):
        """Calcule, une fois pour toutes, le contour exact de chaque pad de
        transition : quart de disque (rayon r_transition) moins l'union des
        carrés cartésiens qui l'empiètent réellement — reflète le vrai bord
        en escalier plutôt que la fausse frontière circulaire r_in. Utilisé
        pour l'affichage des charges réelles (self.trans_paths, précision
        physique) ; conserve aussi l'union exacte et les entrées d'anneaux
        pour que _build_background_patches puisse recalculer une variante
        dilatée cohérente avec son propre contour cartésien."""
        # Union de tous les carrés cartésiens (empreinte géométrique réelle,
        # peu importe leur quadrant : la différence booléenne clippe seule
        # à la bonne portion pour chaque quart de disque)
        h = self.pad_size / 2.0
        squares = [box(x - h, y - h, x + h, y + h)
                  for pid, (x, y, r_in, r_out, sx, sy) in
                  ((pid, g) for pid, g in self.pad_geom.items())
                  if r_out == 0.0]
        self._squares_union = unary_union(squares) if squares else None

        # Pad de transition = anneau le plus interne (plus petit r_in) par quadrant
        self._ring_entries = {pid: g for pid, g in self.pad_geom.items() if g[3] > 0.0}
        if not self._ring_entries:
            self._r_quinc = None
            return
        self._r_quinc = min(g[2] for g in self._ring_entries.values())

        shapes = self._compute_transition_shapes(
            self._squares_union, self._ring_entries, self._r_quinc)
        for pid, shape in shapes.items():
            self.trans_paths[pid] = _shapely_to_mpl_path(shape)

    def _build_background_patches(self):
        """Construit une fois le fond neutre de repère : chaque pad
        cartésien individuellement (visible même non touché), les pads de
        transition (disque moins carrés, cf. _compute_transition_shapes) et
        les vrais anneaux périphériques (Wedge)."""
        self.bg_patches = []
        h = self.pad_size / 2.0

        for pid, g in self.pad_geom.items():
            x, y, r_in, r_out, sx, sy = g
            if r_out == 0.0:
                # Pad cartésien : affiché individuellement, même non touché
                self.bg_patches.append(mpatches.Rectangle(
                    (x - h, y - h), self.pad_size, self.pad_size))
            elif pid in self.trans_paths:
                self.bg_patches.append(mpatches.PathPatch(self.trans_paths[pid]))
            else:
                sx, sy = int(sx), int(sy)
                if   sx > 0 and sy > 0: theta1, theta2 = 0, 90
                elif sx < 0 and sy > 0: theta1, theta2 = 90, 180
                elif sx < 0 and sy < 0: theta1, theta2 = 180, 270
                else:                   theta1, theta2 = 270, 360
                width = max(r_out - r_in, 0.01)
                self.bg_patches.append(
                    mpatches.Wedge((0, 0), r_out, theta1, theta2, width=width))

    def _draw(self):
        i = self.event
        self.ax_pad.clear()

        # Grille complète en fond (contour neutre, tous quadrants,
        # touchés ou non) — repère géométrique constant d'un event à l'autre
        if self.pad_geom:
            pc_bg = PatchCollection(self.bg_patches, facecolor="none",
                                   edgecolor="black", linewidths=0.15,
                                   zorder=1)
            self.ax_pad.add_collection(pc_bg)

        # --- Plan d'anode 2D (H + L) ---
        # Agrège d'abord les charges brutes par pad_id (H + L confondus) :
        # un pad touché par les deux fragments doit afficher leur somme,
        # comme le ferait l'électronique réelle sur un canal commun.
        combined = {}  # pad_id -> [q_sum, px, py]
        for frag in ("H", "L"):
            px  = np.array(self.f[f"/{frag}/px"][i])
            py  = np.array(self.f[f"/{frag}/py"][i])
            pq  = np.array(self.f[f"/{frag}/pq"][i])
            pid = np.array(self.f[f"/{frag}/pid"][i], dtype=int)

            for k in range(len(pq)):
                key = int(pid[k])
                if key not in combined:
                    combined[key] = [0.0, px[k], py[k]]
                combined[key][0] += pq[k]

        # Seuil appliqué sur la charge combinée (comportement d'un vrai canal)
        pad_totals = {pid: v[0] for pid, v in combined.items()}
        combined = {pid: v for pid, v in combined.items() if v[0] >= self.threshold}

        if combined:
            q_cat = np.array([v[0] for v in combined.values()])
            q_min = max(q_cat.min(), 1e-3)
            q_max = q_cat.max()
            norm  = mcolors.LogNorm(vmin=q_min, vmax=q_max)
            cmap  = plt.colormaps["viridis"]

            sq_patches, sq_q = [], []
            ring_patches, ring_q = [], []

            for pad_id, (q, px_k, py_k) in combined.items():
                # Détermine si c'est un pad carré ou un anneau/transition
                is_ring = False
                r_in, r_out, sx, sy = 0.0, 0.0, +1, +1
                if self.pad_geom and pad_id in self.pad_geom:
                    g = self.pad_geom[pad_id]
                    r_in, r_out, sx, sy = g[2], g[3], int(g[4]), int(g[5])
                    is_ring = (r_out > 0.0)  # anneau/transition si r_out > 0

                if is_ring:
                    if pad_id in self.trans_paths:
                        # Pad de transition : contour exact (quart de disque
                        # moins les carrés cartésiens qui l'empiètent)
                        patch = mpatches.PathPatch(self.trans_paths[pad_id])
                        ring_patches.append(patch)
                        ring_q.append(q)
                    else:
                        # Vrai anneau périphérique : quart de cercle exact (Wedge)
                        if   sx > 0 and sy > 0: theta1, theta2 = 0, 90
                        elif sx < 0 and sy > 0: theta1, theta2 = 90, 180
                        elif sx < 0 and sy < 0: theta1, theta2 = 180, 270
                        else:                   theta1, theta2 = 270, 360
                        width = max(r_out - r_in, 0.01)
                        w = mpatches.Wedge((0, 0), r_out, theta1, theta2, width=width)
                        ring_patches.append(w)
                        ring_q.append(q)
                else:
                    # Pad carré
                    h = self.pad_size / 2.0
                    sq_patches.append(mpatches.Rectangle(
                        (px_k - h, py_k - h), self.pad_size, self.pad_size))
                    sq_q.append(q)

            if ring_patches:
                pc_ring = PatchCollection(ring_patches, cmap=cmap, norm=norm,
                                         edgecolor="black", linewidths=0.15,
                                         zorder=3)
                pc_ring.set_array(np.array(ring_q))
                self.ax_pad.add_collection(pc_ring)

            if sq_patches:
                pc_sq = PatchCollection(sq_patches, cmap=cmap, norm=norm,
                                       edgecolor="black", linewidths=0.15,
                                       zorder=5)
                pc_sq.set_array(np.array(sq_q))
                self.ax_pad.add_collection(pc_sq)

            # Recrée cbar_ax proprement
            if self.cbar_ax in self.fig.axes:
                self.cbar_ax.remove()
            self.cbar_ax = self.fig.add_axes([0.44, 0.15, 0.02, 0.7])
            self.cbar = self.fig.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=self.cbar_ax, label="Q (n_electrons)")

        # Vertex projeté (vx, vy en type2 vy=0 → on utilise vx uniquement)
        vx = float(self.f["/H/vx"][i])
        vy = float(self.f["/H/vy"][i])
        self.ax_pad.scatter([vx], [vy], color="red", marker="*", s=150,
                           zorder=5, label="Vertex")

        self.ax_pad.set_xlabel("x (cm)")
        self.ax_pad.set_ylabel("y (cm)")
        self.ax_pad.set_xlim(-10, 10)
        self.ax_pad.set_ylim(-10, 10)
        self.ax_pad.set_aspect("equal")
        self.ax_pad.set_title(
            f"Anode — Event {i}/{self.n_events-1}  |  seuil Q>{self.threshold:.0f} e⁻")
        self.ax_pad.legend(fontsize=8)

        self._draw_qt(i, pad_totals)

        self.fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5file")
    parser.add_argument("--event",    type=int,  default=0)
    parser.add_argument("--pad_file", type=str,  default="my_pad.txt",
                       help="my_pad.txt pour afficher la géométrie des anneaux (défaut: my_pad.txt)")
    args = parser.parse_args()

    ViewerType2(args.h5file, pad_file=args.pad_file, start_event=args.event)


if __name__ == "__main__":
    main()
