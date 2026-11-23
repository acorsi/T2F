#!/usr/bin/env python3
"""
Viewer type2 : plan d'anode 2D + courbes de sortie des pads (format /P : liste unique de pads).

Usage:
    python view_pads.py output.h5 [--event N] [--pad_file my_pad.txt]

La geometrie des pads est lue dans le groupe /pads du .h5. --pad_file n'est necessaire que pour un ancien
fichier sans /pads ; si le .h5 a /pads, --pad_file est ignore.

Le fichier lu est celui de ./process avec la liste unique de pads (groupe /P) :
    pid             : numero du pad ; sa position (x, y) vient de /pads (geometrie de tous les pads), pas de /P
    pq              : charge vraie du pad (H+L), en electrons
    pflag           : 1 = H seul, 2 = L seul, 3 = H et L, 0 = pad fantome (bruit seul)
    pk0, pnt, pqn   : echantillons ecrits (e- equivalents) : pqn = sortie du preampli, echantillonnee
                      par le CAN ; le temps de l'echantillon j (compte depuis le debut de la fenetre
                      complete) est  t = (phase + j * adc_ratio) * delta_t  (phase : /P/phase)
Avec bruit (noise_enable = 1) les courbes sont bruitees et la ligne en tirets est le seuil de lecture.

Navigation:
    -> / d : event suivant
    <- / a : event precedent
    haut / bas : seuil d'affichage sur pq +/- step (plan d'anode et selection automatique)
    clic    : ajoute / retire le pad clique (plan d'anode) des courbes
    c       : efface la selection et revient a la selection automatique
              (les N_AUTO pads les plus charges + le pad au lobe negatif le plus profond)
    i       : bascule la selection automatique : pads au-dessus du seuil <-> tous les pads ecrits
    f       : bascule la couleur du plan : charge pq <-> flag (H seul, L seul, H+L, fantome)
    q       : quitter

Titre de l'anode : KE de chaque fragment, puis somme des pads ecrits (x W) comparee a la somme des KE
des fragments actifs. Avec le bruit, les pads non lus manquent a la somme : l'ecart mesure la charge perdue.
"""

import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from shapely.geometry import Point, box
from shapely.ops import unary_union


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

# couleurs du mode "flag" du plan d'anode
FLAG_COULEUR = {1: "tab:blue", 2: "tab:red", 3: "tab:purple", 0: "0.55"}
FLAG_TEXTE   = {1: "H", 2: "L", 3: "H+L", 0: "fantome"}


class ViewerType2:
    def __init__(self, h5_path, pad_file=None, start_event=0):
        self.f = h5py.File(h5_path, "r")
        if "/P" not in self.f or "/event_id" not in self.f:
            raise SystemExit("[view_pads] %s : pas de groupe /P ni de /event_id : ce viewer lit le format "
                             "a liste unique de pads (sortie recente de ./process)." % h5_path)
        self.n_events = self.f["/event_id"].shape[0]
        self.event = min(max(start_event, 0), self.n_events - 1)

        attrs = self.f["/metadata"].attrs
        self.pad_size  = float(attrs["pad_size"])
        self.inter_pad = float(attrs["inter_pad"])
        self.delta_t   = float(attrs["delta_t"])
        self.adc_ratio = int(attrs["adc_ratio"])
        self.noise_on  = int(attrs["noise_enable"]) == 1
        self.read_thr  = float(attrs["read_thr_e"])

        # pour la comparaison a KE
        self.W = float(attrs["W"])                       # MeV par paire
        self.frag_mask = int(attrs["frag_mask"]) if "frag_mask" in attrs else 3
        self.has_KE = ("KE" in self.f["/H"]) and ("KE" in self.f["/L"])

        self.show_all = False     # selection automatique parmi tous les pads ecrits
        self.color_mode = "charge"    # "charge" ou "flag"
        self.selected = []        # pad_id choisis au clic (vide = selection automatique)
        self.threshold      = 100.0
        self.threshold_step = 1000.0
        self.cbar = None
        self.pads = {}            # pad_id -> dict de l'evenement courant

        # Geometrie des pads : groupe /pads du .h5 (fichiers recents), sinon my_pad.txt (--pad_file)
        self.pad_geom = None
        self.trans_paths = {}  # pad_id -> matplotlib.path.Path (contour exact)
        if "/pads" in self.f:
            P = self.f["/pads"]
            cols = [np.array(P[k], dtype=float) for k in ("x", "y", "r_in", "r_out", "sx", "sy")]
            ids = np.array(P["id"], dtype=int)
            # meme contenu que my_pad.txt : id -> (x, y, r_in, r_out, sx, sy)
            self.pad_geom = {int(ids[k]): np.array([c[k] for c in cols]) for k in range(len(ids))}
            if pad_file:
                print("[view_pads] le .h5 contient /pads : --pad_file ignore.")
        elif pad_file:
            try:
                data = np.loadtxt(pad_file)
            except OSError:
                raise SystemExit(f"[view_pads] pad_file '{pad_file}' introuvable.")
            # colonnes : id x y r_in r_out sx sy
            self.pad_geom = {int(row[0]): row[1:] for row in data}
        else:
            raise SystemExit("[view_pads] ce .h5 n'a pas de groupe /pads (ancien format) : "
                             "donner --pad_file my_pad.txt")
        self._build_transition_paths()
        self._build_background_patches()

        self.r_view = 10.0
        if self.pad_geom:
            self._check_pad_file()
            r = 0.0
            for pid, (px, py, r_in, r_out, sx, sy) in self.pad_geom.items():
                if r_out > 0.0:
                    r = max(r, r_out)
                else:
                    r = max(r, np.hypot(px, py) + self.pad_size)
            self.r_view = float(np.ceil(r))

        self.fig = plt.figure(figsize=(18, 7))
        self.ax_pad  = self.fig.add_axes([0.04, 0.08, 0.38, 0.80])
        self.ax_qt   = self.fig.add_axes([0.58, 0.08, 0.39, 0.84])
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
        elif event.key == "f":
            self.color_mode = "flag" if self.color_mode == "charge" else "charge"
            self._draw()
        elif event.key == "q":
            plt.close(self.fig)

    def _check_pad_file(self):
        """Avertit si le fichier de pads ne correspond pas a la sortie .h5."""
        # 1) pas des lignes de pads = pad_size + inter_pad ?
        ys = sorted(set(round(g[1], 6) for g in self.pad_geom.values() if g[3] == 0.0))
        if len(ys) > 1:
            pitch_file = min(b - a for a, b in zip(ys[:-1], ys[1:]))
            pitch_h5 = self.pad_size + self.inter_pad
            if abs(pitch_file - pitch_h5) > 1e-4:
                print("[view_pads] ATTENTION : pas de pads du fichier %.4f cm, du .h5 %.4f cm : "
                      "mauvaise geometrie de pads ?" % (pitch_file, pitch_h5))
        # 2) tous les pad_id ecrits (200 premiers evenements) existent-ils dans le fichier ?
        ids = set()
        for arr in self.f["/P/pid"][:200]:
            ids.update(int(p) for p in arr)
        manquants = [p for p in ids if p not in self.pad_geom]
        if manquants:
            print("[view_pads] ATTENTION : %d pad_id du .h5 absents du fichier de pads "
                  "(ex. %s) : mauvaise geometrie de pads ?" % (len(manquants), manquants[:5]))

    def _load_event(self, i):
        """Lit la liste unique de pads de l'evenement i : dict pad_id -> {x, y, q, flag, t, v}.
        t : temps des echantillons ecrits (ns), v : valeurs (e- equivalents)."""
        P = self.f["/P"]
        pid  = np.array(P["pid"][i], dtype=int)
        pq   = np.array(P["pq"][i], dtype=float)
        flag = np.array(P["pflag"][i], dtype=int)
        pk0  = np.array(P["pk0"][i], dtype=int)
        pnt  = np.array(P["pnt"][i], dtype=int)
        pqn  = np.array(P["pqn"][i], dtype=float)
        phase = int(P["phase"][i])
        offsets = np.concatenate([[0], np.cumsum(pnt)])
        out = {}
        for k in range(len(pid)):
            g = self.pad_geom.get(int(pid[k])) if self.pad_geom else None
            if g is None:
                raise SystemExit("[view_pads] pad %d absent du fichier de pads : mauvaise geometrie de pads ?" % int(pid[k]))
            n = int(pnt[k])
            j = int(pk0[k]) + np.arange(n)
            out[int(pid[k])] = {
                "x": g[0], "y": g[1], "q": pq[k], "flag": int(flag[k]),
                "t": (phase + j * self.adc_ratio) * self.delta_t,
                "v": pqn[offsets[k]:offsets[k] + n],
            }
        self.phase = phase
        self.n_ghost = int(P["n_ghost"][i])
        return out

    def _pad_at(self, x, y):
        """Identifiant du pad situé au point (x, y) du plan d'anode, ou None."""
        h0 = self.pad_size / 2.0
        for pid, (px, py) in getattr(self, "pad_pos", {}).items():
            if abs(x - px) <= h0 and abs(y - py) <= h0:
                return pid
        if not self.pad_geom:
            return None
        h = self.pad_size / 2.0
        for pid, (px, py, r_in, r_out, sx, sy) in self.pad_geom.items():
            if r_out == 0.0 and abs(x - px) <= h and abs(y - py) <= h:
                return pid
        r = np.hypot(x, y)
        # pad de transition : tout point du quadrant, a r < r_out, hors des cellules des pads carres
        rq = getattr(self, "_r_quinc", None)
        for pid, (px, py, r_in, r_out, sx, sy) in self.pad_geom.items():
            if (r_out > 0.0 and rq is not None and r_in == rq and r < r_out
                    and (x > 0) == (sx > 0) and (y > 0) == (sy > 0)
                    and not (self._squares_union is not None and self._squares_union.contains(Point(x, y)))):
                return pid
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

    def _draw_curves(self):
        """Courbes de sortie des pads selectionnes (une courbe par pad : liste unique)."""
        pads = self.pads
        if self.selected:
            ids = [p for p in self.selected if p in pads]
        else:
            if self.show_all:
                cand = list(pads.keys())
            else:
                cand = [p for p, d in pads.items() if d["q"] >= self.threshold or d["flag"] == 0]
            ids = sorted(cand, key=lambda p: -pads[p]["q"])[:N_AUTO]
            # pad au lobe negatif le plus profond (parmi les candidats)
            cand = [p for p in cand if pads[p]["v"].size]
            if cand:
                prof = min(cand, key=lambda p: pads[p]["v"].min())
                if pads[prof]["v"].min() < 0.0 and prof not in ids:
                    ids.append(prof)
        self.shown = ids

        ax = self.ax_qt
        ax.clear()
        if not ids:
            ax.text(0.5, 0.5, "Aucun pad selectionne", ha="center", va="center", transform=ax.transAxes)
            return

        for n, pad_id in enumerate(ids):
            d = pads[pad_id]
            col = COULEURS[n % len(COULEURS)]
            lab = "pad %d [%s]  pq=%.0f  pic=%.0f" % (pad_id, FLAG_TEXTE[d["flag"]], d["q"],
                                                      d["v"].max() if d["v"].size else 0.0)
            ax.plot(d["t"], d["v"], color=col, lw=1.4, marker=".", label=lab)

        ax.axhline(0.0, color="0.6", lw=0.6)
        if self.noise_on:
            ax.axhline(self.read_thr, color="0.3", lw=1.0, ls="--")
            ax.text(0.01, self.read_thr, " seuil de lecture (%.0f e-)" % self.read_thr, color="0.3",
                    fontsize=8, va="bottom", transform=ax.get_yaxis_transform())
        ax.set_xlabel("t (ns)")
        ax.set_ylabel("sortie du preampli (e- equiv.)")
        ax.set_title("Pads selectionnes (clic sur le plan pour ajouter / retirer) - pads ecrits uniquement",
                     fontsize=10)
        ax.legend(fontsize=8, loc="upper right")

        # contours colores sur le plan d'anode
        h = self.pad_size / 2.0
        for n, pad_id in enumerate(ids):
            g = self.pad_geom.get(pad_id) if self.pad_geom else None
            if g is not None and g[3] > 0.0:          # anneau / pad de transition : geometrie du fichier
                patch = self._pad_patch(pad_id)
            else:                                      # pad carre : centre lu dans le .h5
                x, y = pads[pad_id]["x"], pads[pad_id]["y"]
                patch = mpatches.Rectangle((x - h, y - h), self.pad_size, self.pad_size)
            patch.set_facecolor("none")
            patch.set_edgecolor(COULEURS[n % len(COULEURS)])
            patch.set_linewidth(2.2)
            patch.set_zorder(8)
            self.ax_pad.add_patch(patch)

    def _ke_lines(self, i):
        """Lignes de titre : KE de H et L, puis somme des pads ecrits comparee a la somme des KE actifs.
        (la liste unique ne permet plus la comparaison fragment par fragment)"""
        if not self.has_KE:
            return ""

        def rel(s, k):
            return (s - k) / k if k != 0.0 else float("nan")

        lines = []
        ke_act = 0.0
        for bit, frag in ((1, "H"), (2, "L")):
            ke = float(self.f["/%s/KE" % frag][i])
            if not (self.frag_mask & bit):
                lines.append("%s : eteint (KE = %.3f MeV)" % (frag, ke))
                continue
            lines.append("%s : KE = %.3f MeV" % (frag, ke))
            ke_act += ke
        s_tot = sum(d["q"] for d in self.pads.values()) * self.W
        lines.append("Total (pads ecrits) : somme pads x W = %.3f MeV | KE actifs = %.3f MeV | ecart rel = %+.2e"
                     % (s_tot, ke_act, rel(s_tot, ke_act)))
        return "\n".join(lines)

    def _compute_transition_shapes(self, squares_union, ring_entries, r_quinc):
        """Anneau de transition = disque complet (rayon r_transition) moins les
        CELLULES des pads carrés (cellule = pad + inter-pad, de côté pitch) : comme dans la
        simulation (PadPlane::build_cells), la charge d'une cellule reelle va toute entiere a son
        pad, trous inter-pad compris ; seules les cellules virtuelles, au bord de la zone en
        quinconce, vont vers l'anneau de transition."""
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
        cellules (pitch) des pads carrés — reflète le vrai bord en escalier
        plutôt que la fausse frontière circulaire r_in."""
        h = (self.pad_size + self.inter_pad) / 2.0      # demi-cote d'une cellule : pad + inter-pad
        squares = [box(x - h, y - h, x + h, y + h)
                  for pid, (x, y, r_in, r_out, sx, sy) in
                  ((pid, g) for pid, g in self.pad_geom.items())
                  if r_out == 0.0]
        # les cellules en briques se touchent exactement : sans une dilatation de 1 micron, l'arrondi
        # flottant laisse des filets de largeur nulle dans la difference geometrique
        self._squares_union = unary_union(squares).buffer(1.0e-4) if squares else None

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
        transition (disque moins carrés) et les vrais anneaux périphériques (Wedge)."""
        self.bg_patches = []
        h = self.pad_size / 2.0

        for pid, g in self.pad_geom.items():
            x, y, r_in, r_out, sx, sy = g
            if r_out == 0.0:
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
        self.pads = self._load_event(i)

        # Grille complete en fond (contour neutre, tous quadrants, touches ou non)
        if self.pad_geom:
            pc_bg = PatchCollection(self.bg_patches, facecolor="none",
                                    edgecolor="black", linewidths=0.15, zorder=1)
            self.ax_pad.add_collection(pc_bg)

        # --- Plan d'anode 2D : pads ecrits d'au moins le seuil (les fantomes sont toujours montres) ---
        shown = dict((p, d) for p, d in self.pads.items() if d["q"] >= self.threshold or d["flag"] == 0)

        # le colorbar est recree a chaque fois (ou retire en mode flag)
        if self.cbar_ax is not None and self.cbar_ax in self.fig.axes:
            self.cbar_ax.remove()
        self.cbar_ax = None

        if shown:
            charged = [d["q"] for d in shown.values() if d["flag"] != 0]
            norm = None
            cmap = plt.colormaps["viridis"]
            if self.color_mode == "charge" and charged:
                norm = mcolors.LogNorm(vmin=max(min(charged), 1e-3), vmax=max(charged))

            ring_patches, ring_cols = [], []
            sq_patches, sq_cols = [], []
            for pad_id, d in shown.items():
                is_ring = False
                r_in, r_out, sx, sy = 0.0, 0.0, +1, +1
                if self.pad_geom and pad_id in self.pad_geom:
                    g = self.pad_geom[pad_id]
                    r_in, r_out, sx, sy = g[2], g[3], int(g[4]), int(g[5])
                    is_ring = (r_out > 0.0)
                if self.color_mode == "flag" or d["flag"] == 0 or norm is None:
                    col = mcolors.to_rgba(FLAG_COULEUR[d["flag"]])
                else:
                    col = cmap(norm(d["q"]))
                if is_ring:
                    ring_patches.append(self._pad_patch(pad_id))
                    ring_cols.append(col)
                else:
                    h = self.pad_size / 2.0
                    sq_patches.append(mpatches.Rectangle((d["x"] - h, d["y"] - h), self.pad_size, self.pad_size))
                    sq_cols.append(col)

            if ring_patches:
                self.ax_pad.add_collection(PatchCollection(ring_patches, facecolor=ring_cols,
                                           edgecolor="black", linewidths=0.15, zorder=3))
            if sq_patches:
                self.ax_pad.add_collection(PatchCollection(sq_patches, facecolor=sq_cols,
                                           edgecolor="black", linewidths=0.15, zorder=5))

            if self.color_mode == "charge" and norm is not None:
                self.cbar_ax = self.fig.add_axes([0.44, 0.15, 0.02, 0.7])
                self.cbar = self.fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                                              cax=self.cbar_ax, label="pq : charge vraie du pad (e-)")
            elif self.color_mode == "flag":
                proxy = [mpatches.Patch(facecolor=FLAG_COULEUR[k], edgecolor="black", label=FLAG_TEXTE[k])
                         for k in (1, 2, 3, 0)]
                self.ax_pad.legend(handles=proxy, fontsize=8, loc="lower left", title="fragment")

        # Vertex projete (vx, vy)
        if "/vertex" in self.f:
            vx = float(self.f["/vertex/x"][i])
            vy = float(self.f["/vertex/y"][i])
        else:                                   # anciens fichiers
            vx = float(self.f["/H/vx"][i])
            vy = float(self.f["/H/vy"][i])
        self.ax_pad.scatter([vx], [vy], color="red", marker="*", s=150, zorder=9, label="Vertex")

        self.ax_pad.set_xlabel("x (cm)")
        self.ax_pad.set_ylabel("y (cm)")
        self.ax_pad.set_xlim(-self.r_view, self.r_view)
        self.ax_pad.set_ylim(-self.r_view, self.r_view)
        self.ax_pad.set_aspect("equal")
        title = ("Anode - Event %d/%d (id %d) | %d pads ecrits | phase %d | fantomes %d | seuil pq>%.0f e-"
                 % (i, self.n_events - 1, int(self.f["/event_id"][i]), len(self.pads), self.phase,
                    self.n_ghost, self.threshold))
        ke_txt = self._ke_lines(i)
        if ke_txt:
            title += "\n" + ke_txt
        self.ax_pad.set_title(title, fontsize=7.5)
        if self.color_mode == "charge":
            self.ax_pad.legend(fontsize=8)

        self._draw_curves()
        self.fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("h5file")
    parser.add_argument("--event",    type=int,  default=0)
    parser.add_argument("--pad_file", type=str,  default=None,
                        help="my_pad.txt : seulement pour un ancien .h5 sans groupe /pads")
    args = parser.parse_args()

    ViewerType2(args.h5file, pad_file=args.pad_file, start_event=args.event)


if __name__ == "__main__":
    main()
