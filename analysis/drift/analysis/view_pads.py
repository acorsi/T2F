#!/usr/bin/env python3
"""
Viewer type2 : plan d'anode 2D (H+L) + profils Q(t) H et L.

Usage:
    python view_type2.py output.h5 [--event N] [--old ancien.h5] [--pad_file my_pad.txt]

Avec --old : les courbes de l'ancienne version (sans induction, même événement) sont
superposées en tirets à celles de la nouvelle version (trait plein). Les pads sélectionnés
sont toujours ceux de la nouvelle version.

Navigation:
    -> / d : event suivant
    <- / a : event précédent
    ↑ / ↓  : seuil +/- step
    clic    : ajoute / retire le pad cliqué (plan d'anode) des traces
    c       : efface la sélection et revient à la sélection automatique
              (les N_AUTO pads les plus chargés + le pad au lobe négatif le plus profond)
    i       : bascule la sélection automatique : pads au-dessus du seuil <-> tous les pads écrits
    q       : quitter

Ancienne version : qt = charge arrivée dans le bin (positive) ; Q ancienne = somme cumulée
des qt écrits, la charge finale de l'ancien fichier (pq) est marquée par un point à droite.

Panneaux de droite : en haut qt par bin (signé), en bas Q cumulée. Chaque pad
sélectionné a la même couleur sur le plan d'anode (contour épais) et sur les courbes.
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


N_AUTO = 4
COULEURS = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple",
            "tab:brown", "tab:pink", "tab:olive", "tab:cyan", "k"]


class ViewerType2:
    def __init__(self, h5_path, pad_file=None, start_event=0, old_path=None):
        self.f = h5py.File(h5_path, "r")
        self.n_events = self.f["/H/dE"].shape[0]
        self.event = start_event

        m = read_metadata(self.f)
        self.pad_size  = m["pad_size"]
        self.inter_pad = m["inter_pad"]
        self.delta_t   = m["delta_t"]

        self.has_Q = "pQ" in self.f["/H"]
        self.f_old = None
        self.delta_t_old = self.delta_t
        if old_path:
            self.f_old = h5py.File(old_path, "r")
            self.delta_t_old = read_metadata(self.f_old)["delta_t"]
            if self.f_old["/H/dE"].shape[0] != self.n_events:
                print("[view_type2] ATTENTION : nombres d'événements différents (nouveau %d, ancien %d)"
                      % (self.n_events, self.f_old["/H/dE"].shape[0]))
        self.mode = "qt"          # utilisé en interne par _combined_qt
        self.show_all = False     # sélection automatique parmi tous les pads écrits
        self.selected = []        # pad_id choisis au clic (vide = sélection automatique)
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
        self.ax_qt   = self.fig.add_axes([0.52, 0.55, 0.43, 0.37])
        self.ax_Q    = self.fig.add_axes([0.52, 0.08, 0.43, 0.37], sharex=self.ax_qt)
        self.cbar_ax = self.fig.add_axes([0.44, 0.15, 0.02, 0.7])

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
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
        elif event.key == "c":
            self.selected = []
            self._draw()
        elif event.key == "i":
            self.show_all = not self.show_all
            self._draw()
        elif event.key == "q":
            plt.close(self.fig)

    def _pad_at(self, x, y):
        """Identifiant du pad situé au point (x, y) du plan d'anode, ou None."""
        if not self.pad_geom:
            return None
        h = self.pad_size / 2.0
        for pid, (px, py, r_in, r_out, sx, sy) in self.pad_geom.items():
            if r_out == 0.0 and abs(x - px) <= h and abs(y - py) <= h:
                return pid
        r = np.hypot(x, y)
        for pid, (px, py, r_in, r_out, sx, sy) in self.pad_geom.items():
            if r_out > 0.0 and r_in <= r < r_out and (x > 0) == (sx > 0) and (y > 0) == (sy > 0):
                return pid
        return None

    def _on_click(self, event):
        if event.inaxes is not self.ax_pad or event.xdata is None:
            return
        pid = self._pad_at(event.xdata, event.ydata)
        if pid is None:
            return
        if pid in self.selected:
            self.selected.remove(pid)
        elif len(self.selected) < len(COULEURS):
            self.selected.append(pid)
        self._draw()

    def _pad_patch(self, pad_id, px_k=None, py_k=None):
        """Patch matplotlib d'un pad (carré, transition ou anneau)."""
        h = self.pad_size / 2.0
        g = self.pad_geom[pad_id] if (self.pad_geom and pad_id in self.pad_geom) else None
        if g is None:
            return mpatches.Rectangle((px_k - h, py_k - h), self.pad_size, self.pad_size)
        x, y, r_in, r_out, sx, sy = g
        if r_out == 0.0:
            return mpatches.Rectangle((x - h, y - h), self.pad_size, self.pad_size)
        if pad_id in self.trans_paths:
            return mpatches.PathPatch(self.trans_paths[pad_id])
        sx, sy = int(sx), int(sy)
        if   sx > 0 and sy > 0: theta1, theta2 = 0, 90
        elif sx < 0 and sy > 0: theta1, theta2 = 90, 180
        elif sx < 0 and sy < 0: theta1, theta2 = 180, 270
        else:                   theta1, theta2 = 270, 360
        return mpatches.Wedge((0, 0), r_out, theta1, theta2, width=max(r_out - r_in, 0.01))

    def _combined_qt(self, i, f=None, dt=None):
        """Construit, par pad_id, la série temporelle combinée H+L, dict {k: valeur}
        avec k = round(t/delta_t).

        mode "qt" : les qt sont sommés pour les bins identiques (un bin présent dans
        un seul fragment est conservé tel quel).
        mode "Q"  : charge cumulée. Chaque fragment ne fournit Q qu'aux bins gardés ;
        entre deux bins gardés on maintient la dernière valeur (les bins non gardés
        sont sous le seuil, l'erreur est de l'ordre du seuil), puis on somme H et L.
        """
        if f is None:
            f = self.f
        if dt is None:
            dt = self.delta_t
        per_frag = {}   # pad_id -> liste de (ks, valeurs), un élément par fragment
        for frag in ("H", "L"):
            pid = np.array(f[f"/{frag}/pid"][i], dtype=int)
            pt  = np.array(f[f"/{frag}/pt"][i])
            pnt = np.array(f[f"/{frag}/pnt"][i], dtype=int)
            name = "pQ" if self.mode == "Q" else "pqt"      # mode "Qold" : somme cumulée des pqt
            pv  = np.array(f[f"/{frag}/{name}"][i])
            if len(pnt) == 0:
                continue

            offsets = np.concatenate([[0], np.cumsum(pnt)])
            for k_pad in range(len(pnt)):
                n = pnt[k_pad]
                if n == 0:
                    continue
                pad_id = int(pid[k_pad])
                ks = [int(round(float(t) / dt))
                      for t in pt[offsets[k_pad]:offsets[k_pad] + n]]
                vs = [float(v) for v in pv[offsets[k_pad]:offsets[k_pad] + n]]
                if self.mode == "Qold":
                    vs = list(np.cumsum(vs))
                per_frag.setdefault(pad_id, []).append((ks, vs))

        series = {}
        for pad_id, lst in per_frag.items():
            bins = {}
            if self.mode == "qt":
                for ks, vs in lst:
                    for k, v in zip(ks, vs):
                        bins[k] = bins.get(k, 0.0) + v
            else:
                all_k = sorted(set(k for ks, vs in lst for k in ks))
                for k in all_k:
                    tot = 0.0
                    for ks, vs in lst:
                        last = 0.0
                        for kk, vv in zip(ks, vs):
                            if kk <= k:
                                last = vv
                            else:
                                break
                        tot += last
                    bins[k] = tot
            series[pad_id] = bins
        return series

    def _draw_qt(self, i, pad_totals):
        """Trace qt(t) (haut) et Q(t) (bas) des pads sélectionnés, H+L combinés."""
        series_qt = self._combined_qt_mode(i, "qt")
        series_Q = self._combined_qt_mode(i, "Q") if self.has_Q else {}
        old_qt, old_Q, old_q = {}, {}, {}
        if self.f_old is not None:
            old_qt = self._combined_qt_mode(i, "qt", self.f_old, self.delta_t_old)
            old_Q = self._combined_qt_mode(i, "Qold", self.f_old, self.delta_t_old)
            old_q = self._final_charges(self.f_old, i)

        if self.selected:
            ids = [p for p in self.selected if p in series_qt or p in old_qt]
        else:
            if self.show_all:
                cand = list(series_qt.keys())
            else:
                cand = [p for p, q in pad_totals.items() if q >= self.threshold and p in series_qt]
            ids = sorted(cand, key=lambda p: -pad_totals.get(p, 0.0))[:N_AUTO]
            # pad au lobe négatif le plus profond (parmi les candidats)
            if cand:
                prof = min(cand, key=lambda p: min(series_qt[p].values()))
                if min(series_qt[prof].values()) < 0.0 and prof not in ids:
                    ids.append(prof)
        self.shown = ids

        for ax in (self.ax_qt, self.ax_Q):
            ax.clear()
        if not ids:
            self.ax_qt.text(0.5, 0.5, "Aucun pad sélectionné", ha="center", va="center",
                            transform=self.ax_qt.transAxes)
            return

        for n, pad_id in enumerate(ids):
            col = COULEURS[n % len(COULEURS)]
            lab = f"pad {pad_id}  (q={pad_totals.get(pad_id, 0.0):.0f}"
            if self.f_old is not None:
                lab += f" / ancien {old_q.get(pad_id, 0.0):.0f}"
            lab += ")"
            bins = series_qt.get(pad_id, {})
            ks = sorted(bins.keys())
            self.ax_qt.plot([k * self.delta_t for k in ks], [bins[k] for k in ks],
                            color=col, lw=1.4, marker=".", label=lab)
            if pad_id in series_Q:
                bq = series_Q[pad_id]
                kq = sorted(bq.keys())
                self.ax_Q.step([k * self.delta_t for k in kq], [bq[k] for k in kq],
                               where="post", color=col, lw=1.4)
            if pad_id in old_qt:                      # ancienne version : tirets
                bo = old_qt[pad_id]
                ko = sorted(bo.keys())
                self.ax_qt.plot([k * self.delta_t_old for k in ko], [bo[k] for k in ko],
                                color=col, lw=1.2, ls="--", marker="x", ms=4, alpha=0.9)
            if pad_id in old_Q:
                bo = old_Q[pad_id]
                ko = sorted(bo.keys())
                self.ax_Q.step([k * self.delta_t_old for k in ko], [bo[k] for k in ko],
                               where="post", color=col, lw=1.2, ls="--")
        for ax in (self.ax_qt, self.ax_Q):
            ax.axhline(0.0, color="0.6", lw=0.6)
        self.ax_qt.set_ylabel("qt par bin (e⁻)")
        self.ax_qt.set_title("Signal des pads sélectionnés (clic sur le plan pour ajouter / retirer)",
                             fontsize=10)
        leg = self.ax_qt.legend(fontsize=8, loc="upper right")
        if self.f_old is not None:
            from matplotlib.lines import Line2D
            proxy = [Line2D([0], [0], color="0.3", lw=1.4, marker="."),
                     Line2D([0], [0], color="0.3", lw=1.2, ls="--", marker="x", ms=4)]
            self.ax_Q.legend(proxy, ["nouvelle version", "ancienne version"], fontsize=8,
                             loc="upper left")
            self.ax_qt.add_artist(leg)
        self.ax_Q.set_xlabel("t (ns)")
        self.ax_Q.set_ylabel("Q cumulée (e⁻)" if self.has_Q else "(pas de pQ)")

        # contours colorés sur le plan d'anode
        for n, pad_id in enumerate(ids):
            g = self.pad_geom.get(pad_id) if self.pad_geom else None
            patch = self._pad_patch(pad_id, *(g[:2] if g is not None else (0.0, 0.0)))
            patch.set_facecolor("none")
            patch.set_edgecolor(COULEURS[n % len(COULEURS)])
            patch.set_linewidth(2.2)
            patch.set_zorder(8)
            self.ax_pad.add_patch(patch)

    def _combined_qt_mode(self, i, mode, f=None, dt=None):
        old = self.mode
        self.mode = mode
        try:
            return self._combined_qt(i, f, dt)
        finally:
            self.mode = old

    def _final_charges(self, f, i):
        """Charge finale pq par pad_id (H+L) pour l'événement i du fichier f."""
        out = {}
        for frag in ("H", "L"):
            pid = np.array(f[f"/{frag}/pid"][i], dtype=int)
            pq = np.array(f[f"/{frag}/pq"][i])
            for p, q in zip(pid.tolist(), pq.tolist()):
                out[p] = out.get(p, 0.0) + q
        return out

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
    parser.add_argument("--old", type=str, default=None,
                       help="ancien fichier h5 (sans induction) à superposer, même événements")
    args = parser.parse_args()

    ViewerType2(args.h5file, pad_file=args.pad_file, start_event=args.event, old_path=args.old)


if __name__ == "__main__":
    main()
