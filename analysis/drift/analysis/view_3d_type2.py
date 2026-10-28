#!/usr/bin/env python3
"""
Viewer 3D interactif type2 : traces H/L + pads sur anode unique (z=z_anode).

Usage:
    python view_3d_type2.py output.h5 [--event N]

Navigation:
    -> / d : event suivant
    <- / a : event précédent
    ↑ / ↓  : seuil de charge +/- step
    q       : quitter
"""

import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from read_metadata import read_metadata


class Viewer3DType2:
    def __init__(self, h5_path, start_event=0):
        self.f = h5py.File(h5_path, "r")
        self.n_events = self.f["/H/dE"].shape[0]
        self.event = start_event

        m = read_metadata(self.f)
        self.dkg        = m["dkg"]
        self.dga        = m["dga"]
        self.z_anode    = self.dkg + self.dga   # anode unique
        self.pad_size   = m["pad_size"]
        self.inter_pad  = m["inter_pad"]

        self.threshold      = 100.0
        self.threshold_step = 1000.0

        self.fig = plt.figure(figsize=(20, 9))
        self.ax      = self.fig.add_axes([0.02, 0.05, 0.45, 0.9], projection="3d")
        self.ax_qt_h = self.fig.add_axes([0.55, 0.55, 0.38, 0.38])
        self.ax_qt_l = self.fig.add_axes([0.55, 0.08, 0.38, 0.38])
        self.cbar_ax = self.fig.add_axes([0.94, 0.15, 0.02, 0.7])
        self.cbar = None
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

    def _pad_quads(self, px_arr, py_arr, pq_arr, z_plane, cmap, norm):
        """Carrés dans le plan z=z_plane (pads cartésiens)."""
        half = self.pad_size / 2.0
        verts, colors = [], []
        for px, py, pq in zip(px_arr, py_arr, pq_arr):
            verts.append([
                (px - half, py - half, z_plane),
                (px + half, py - half, z_plane),
                (px + half, py + half, z_plane),
                (px - half, py + half, z_plane),
            ])
            colors.append(cmap(norm(pq)))
        if not verts:
            return None
        poly = Poly3DCollection(verts, zsort="min")
        poly.set_facecolor(colors)
        poly.set_edgecolor("black")
        poly.set_linewidth(0.2)
        return poly

    def _ring_poly(self, r_in, r_out, half, z_plane, color, n_pts=60):
        """Demi-anneau dans le plan z=z_plane (half=+1 y>0, half=-1 y<0)."""
        theta = np.linspace(0, np.pi, n_pts)
        if half == -1:
            theta = theta + np.pi  # demi-cercle inférieur

        x_out = r_out * np.cos(theta)
        y_out = r_out * np.sin(theta)
        x_in  = r_in  * np.cos(theta[::-1])
        y_in  = r_in  * np.sin(theta[::-1])

        xs = np.concatenate([x_out, x_in])
        ys = np.concatenate([y_out, y_in])
        zs = np.full_like(xs, z_plane)

        verts = [list(zip(xs, ys, zs))]
        poly = Poly3DCollection(verts, zsort="min")
        poly.set_facecolor(color)
        poly.set_edgecolor("black")
        poly.set_linewidth(0.3)
        poly.set_alpha(0.9)
        return poly

    def _draw_qt(self, ax, frag, i, color):
        ax.clear()
        pq   = np.array(self.f[f"/{frag}/pq"][i])
        px_q = np.array(self.f[f"/{frag}/px"][i])
        py_q = np.array(self.f[f"/{frag}/py"][i])
        pid  = np.array(self.f[f"/{frag}/pid"][i], dtype=int)
        pt   = np.array(self.f[f"/{frag}/pt"][i])
        pqt  = np.array(self.f[f"/{frag}/pqt"][i])
        pnt  = np.array(self.f[f"/{frag}/pnt"][i], dtype=int)

        if len(pnt) == 0 or len(pt) == 0:
            ax.text(0.5, 0.5, "Aucun signal", ha="center", va="center",
                   transform=ax.transAxes)
            ax.set_title(f"Q(t) Fragment {frag}")
            return

        pitch = self.pad_size + self.inter_pad

        def xy_to_rc(px, py):
            row = int(round(float(py) / pitch))
            x_offset = pitch / 2.0 if row % 2 != 0 else 0.0
            col = int(round((float(px) - x_offset) / pitch))
            return (row, col)

        rc_to_id = {xy_to_rc(px_q[k], py_q[k]): int(pid[k]) for k in range(len(pq))}
        passing_ids = {rc_to_id[xy_to_rc(px_q[k], py_q[k])]
                      for k in range(len(pq)) if pq[k] >= self.threshold}

        cmap = plt.colormaps["tab20"]
        offsets = np.concatenate([[0], np.cumsum(pnt)])
        n_plotted = 0
        for k in range(len(pnt)):
            if int(pid[k]) not in passing_ids:
                continue
            n = pnt[k]
            if n == 0:
                continue
            t_pad  = pt[offsets[k]:offsets[k]+n]
            qt_pad = pqt[offsets[k]:offsets[k]+n]
            ax.plot(t_pad, qt_pad, color=cmap(k % 20 / 20), lw=1.2, alpha=0.8)
            n_plotted += 1

        ax.set_xlabel("t (ns)")
        ax.set_ylabel("Q (e⁻)")
        ax.set_title(f"Q(t) Fragment {frag}  ({n_plotted} pads, seuil>{self.threshold:.0f} e⁻)")

    def _draw(self):
        self.ax.clear()
        i = self.event

        # --- Traces 3D ---
        for frag, color in (("H", "crimson"), ("L", "royalblue")):
            x = np.array(self.f[f"/{frag}/x"][i])
            y = np.array(self.f[f"/{frag}/y"][i])
            z = np.array(self.f[f"/{frag}/z"][i])
            self.ax.plot(x, y, z, color=color, lw=2, label=f"Fragment {frag}")

        # Vertex : (vx, 0, vz) en type2
        vx = float(self.f["/H/vx"][i])
        vz = float(self.f["/H/vz"][i])
        self.ax.scatter([vx], [0], [vz], color="black", marker="*",
                       s=80, zorder=5, label="Vertex")

        # --- Pads sur l'anode unique (H + L) ---
        q_all = []
        for frag in ("H", "L"):
            pq = np.array(self.f[f"/{frag}/pq"][i])
            pq = pq[pq >= self.threshold]
            if len(pq) > 0:
                q_all.append(pq)

        if q_all:
            q_cat = np.concatenate(q_all)
            q_min = max(q_cat.min(), 1e-3)
            q_max = q_cat.max()
            norm = mcolors.LogNorm(vmin=q_min, vmax=q_max)
            cmap = plt.colormaps["viridis"]

            for frag in ("H", "L"):
                px = np.array(self.f[f"/{frag}/px"][i])
                py = np.array(self.f[f"/{frag}/py"][i])
                pq = np.array(self.f[f"/{frag}/pq"][i])

                mask = pq >= self.threshold
                px, py, pq = px[mask], py[mask], pq[mask]

                poly = self._pad_quads(px, py, pq, self.z_anode, cmap, norm)
                if poly is not None:
                    self.ax.add_collection3d(poly)

            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            self.cbar_ax.clear()
            self.cbar = self.fig.colorbar(sm, cax=self.cbar_ax,
                                          label="Q (n_electrons)")

        # Fils à plomb
        for frag, color in (("H", "crimson"), ("L", "royalblue")):
            x = np.array(self.f[f"/{frag}/x"][i])
            y = np.array(self.f[f"/{frag}/y"][i])
            z = np.array(self.f[f"/{frag}/z"][i])
            for xi, yi, zi in ((x[0], y[0], z[0]), (x[-1], y[-1], z[-1])):
                self.ax.plot([xi, xi], [yi, yi], [zi, self.z_anode],
                            color=color, lw=1.5, alpha=0.7, ls=":")

        # Plans repères
        for z_ref in (0, self.dkg):
            xx, yy = np.meshgrid([-10, 10], [-10, 10])
            self.ax.plot_surface(xx, yy, np.full_like(xx, z_ref),
                                alpha=0.05, color="gray")

        # Origine + axe Z au vertex
        self.ax.scatter([0], [0], [0], color="black", s=20, zorder=5)
        self.ax.plot([vx, vx], [0, 0], [-self.dkg, self.z_anode],
                    color="black", lw=0.8, ls="--", alpha=0.5)

        self.ax.set_xlabel("x (cm)")
        self.ax.set_ylabel("y (cm)")
        self.ax.set_zlabel("z (cm)")
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_zlim(-self.dkg - 1, self.z_anode + 1)
        self.ax.set_box_aspect([1, 1, 1.2])
        self.ax.set_title(
            f"Event {i}/{self.n_events-1}  |  seuil Q>{self.threshold:.0f} e⁻",
            fontsize=10)
        self.ax.legend(loc="upper left", fontsize=8)

        self._draw_qt(self.ax_qt_h, "H", i, "crimson")
        self._draw_qt(self.ax_qt_l, "L", i, "royalblue")

        self.fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5file")
    parser.add_argument("--event", type=int, default=0)
    args = parser.parse_args()

    Viewer3DType2(args.h5file, start_event=args.event)


if __name__ == "__main__":
    main()
