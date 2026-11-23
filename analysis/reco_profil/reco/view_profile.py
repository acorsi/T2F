#!/usr/bin/env python3
"""
view_profile.py : viewer du profil dE/dx reconstruit (refine_geom.py) contre la verite, evenement par evenement.

A gauche : plan d'anode comme dans view_pads.py (memes touches, memes options), avec en plus l'axe des deux
fragments (trait vert : geometrie affinee, tirets noirs : geometrie vraie du .h5).
A droite : profils des deux fragments (haut : fragment 0, demi-plan y > y_plane ; bas : fragment 1), ajuste (vert,
bande = bruit prevu) et verite (noir). Par defaut (--units h5), dans l'unite du .h5 : dE par pas de delta (2 mm),
la verite etant la somme des dE de /H et /L par groupes de delta / step_size_eff voxels, et l'ajuste l'energie
W x integrale de q(l) sur chaque intervalle [j delta, (j+1) delta] (q lineaire entre noeuds : W delta (c_j + c_j+1) / 2).
Avec --units ecm : densite q(l) en e-/cm aux noeuds, comme refine_geom.py (verite projetee sur les chapeaux).
Les pads selectionnes sont reperes le long de la trace (trait vertical de la couleur du pad, a la position
de la projection de son centre sur l'axe ; pointille si le pad est trop loin de l'axe pour servir a l'ajustement).

L'ajustement est fait a la volee, a la premiere visite de chaque evenement (quelques secondes : geometrie affinee
par chi2 profile, test de fin de profil et rallonge de la grille, comme refine_geom.py), puis garde en memoire.

Touches (celles de view_pads.py) :
    -> / d : event suivant            <- / a : event precedent
    haut / bas : seuil d'affichage sur pq +/- step (plan d'anode et selection automatique)
    clic    : ajoute / retire le pad clique (plan d'anode)
    c       : efface la selection et revient a la selection automatique
    i       : bascule la selection automatique : pads au-dessus du seuil <-> tous les pads ecrits
    f       : bascule la couleur du plan : charge pq <-> flag
    q       : quitter
et une touche en plus :
    w       : bascule profil brut <-> profil fenetre (moyenne glissante --win, comme _win.csv)

Drapeaux (titre rouge) : GRILLE RALLONGEE (zone rallongee ombree) ; ATTENTION : FIN NON CONTENUE (le profil ne
retombe pas a zero meme apres la rallonge maximale : le range n'est pas mesurable). La verite est tracee jusqu'a la
fin vraie de la trace : au-dela de la grille, elle est en gris clair.

Options de l'ajustement : les memes que refine_geom.py (voir --help). --lam : lissage du profil affiche.

Usage :
  python3 view_profile.py sortie.h5 --exe ../build/profile_basis --cfg ../drift.cfg [--event N] [--lam 1e-5]
"""

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import matplotlib.patches as mpatches

# reco_dir.py et reco_split.py (importes par refine_geom) forcent le backend Agg (non interactif) a l'import,
# sauf si "--show" est dans la ligne de commande. Le viewer doit rester interactif : on masque argv pendant
# les imports, puis on le restaure pour argparse.
_argv = list(sys.argv)
sys.argv = [sys.argv[0], "--show"]
from view_pads import ViewerType2, N_AUTO, COULEURS
from refine_geom import (add_common_args, parse_free, Env, build_state, Server, Ctx, evaluate, refine_chunk,
                         extend_chunk, window_matrix, window_truth, geo_from_params, angle_deg)
from fit_profile import truth_nodes, half_length
sys.argv = _argv


class ProfileViewer(ViewerType2):
    def __init__(self, a):
        self.a = a
        self.lam = a.lam
        self.free = parse_free(a)
        self.env = Env(a)
        self.nint = 1
        if a.units == "h5":
            r = a.delta / self.env.step
            self.nint = int(round(r))
            if self.nint < 1 or abs(r - self.nint) > 1.0e-6:
                raise SystemExit("[view_profile] --units h5 : delta = %g cm n'est pas un multiple entier de "
                                 "step_size_eff = %g cm (rapport %.6f) ; changer --delta ou utiliser --units ecm"
                                 % (a.delta, self.env.step, r))
        self.cache = {}
        self.windowed = False
        self.ax_f = None
        self.tmpdir = tempfile.mkdtemp(prefix="view_profile_")
        self.server = Server(a.exe, a.cfg, a.exe_args.split())
        self.ctx = Ctx(a, self.server, self.env.wh, self.env.step, a.delta, a.y_plane,
                       os.path.join(self.tmpdir, "vp"))
        try:
            ViewerType2.__init__(self, a.h5file, pad_file=a.pad_file, start_event=a.event)
        finally:
            self.server.close()
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------ touches
    def _on_key(self, event):
        if event.key == "w":
            self.windowed = not self.windowed
            self._draw()
        else:
            ViewerType2._on_key(self, event)

    # ------------------------------------------------------------------ ajustement d'un evenement
    def _fit(self, i):
        """ajustement de l'evenement i (garde en memoire) : dict de resultats, ou {"error": raison}"""
        if i in self.cache:
            return self.cache[i]
        try:
            out = self._fit_event(i)
        except Exception as e:                      # un evenement qui echoue ne doit pas tuer le viewer
            out = {"error": "exception : %s" % e}
        self.cache[i] = out
        if "error" in out:
            print("[view_profile] evenement %d : %s" % (i, out["error"]))
        else:
            print("[view_profile] evenement %d : chi2/ndl %.3f -> %.3f | angle vs verite %.2f / %.2f deg | "
                  "rallonge %.1f / %.1f cm%s" % (
                      i, out["chi2d0"], out["chi2d1"], out["ang"][0], out["ang"][1], out["dL"][0], out["dL"][1],
                      " | ATTENTION fin non contenue" if any(out["notc"]) else ""))
        return out

    def _fit_event(self, i):
        a, env, ctx = self.a, self.env, self.ctx
        S, why = build_state(env, a, i)
        if S is None:
            return {"error": why}
        chunk = [S]
        refine_chunk(ctx, chunk, a.lam_geom, self.free, a.max_iter, a.tol)
        S["Lr"], S["Lt"] = None, None
        if a.lrecalc:
            evxy = {"x": S["px"], "y": S["py"]}
            gg = geo_from_params(S["p"], a.y_plane)
            Lk = [half_length(gg, k, evxy, S["amp"], a.y_plane, env.thr, a.k_end, a.axis_dist * env.pitch, a.lmargin)
                  for k in range(2)]
            if Lk[0] is not None and Lk[1] is not None:
                S["Lr"] = Lk
        extend_chunk(ctx, chunk, self.lam, a)
        ext_evt = any(d > 0.0 for d in S["dL"])
        a2 = S["alpha2_f"] if ext_evt else None
        Lfin = [S["Lb"][k] + S["dL"][k] for k in range(2)]
        r, cc, sg, al, mh, ml, G0m, Gm = evaluate(ctx, [(S, S["p"], a2, Lfin)], self.lam, want_G=True)[0]
        B = np.linalg.solve(Gm, G0m)
        Cov = np.linalg.solve(Gm, B.T)
        f = env.f
        dEt = {"H": np.array(f["/H/dE"][i], dtype=float), "L": np.array(f["/L/dE"][i], dtype=float)}
        geo = geo_from_params(S["p"], a.y_plane)
        gt = S["geot"]
        frag = []
        for k in range(2):
            sl = slice(0, mh) if k == 0 else slice(mh, mh + ml)
            M = (mh, ml)[k]
            dEk = dEt[gt["truth"][k]]
            nz = np.nonzero(dEk > 0)[0]
            Ltrue = env.step * (nz[-1] + 1) if len(nz) else 0.0
            Mt = max(M, int(np.ceil(Ltrue / a.delta - 1.0e-9)) + 1)
            ck = cc[sl]
            sck = np.sqrt(np.clip(np.diag(Cov[sl, sl]), 0.0, None))
            t_nodes = truth_nodes(dEk, env.W, env.step, a.delta, Mt)
            if a.win > 0.0:
                Wm, wfull = window_matrix(M, a.delta, a.win)
                qw = Wm @ ck
                sw = np.sqrt(np.clip(np.diag(Wm @ Cov[sl, sl] @ Wm.T), 0.0, None))
                tw = window_truth(dEk, env.W, env.step, a.delta, M, a.win)
            else:
                qw = sw = tw = None
            fr = {"M": M, "Mt": Mt, "c": ck, "sc": sck, "t": t_nodes, "qw": qw, "sw": sw, "tw": tw,
                  "name": gt["truth"][k], "L": Lfin[k]}
            if a.units == "h5":
                # energie par intervalle [j delta, (j+1) delta] : W delta (c_j + c_j+1) / 2 (q lineaire entre noeuds)
                T = np.zeros((M - 1, M))
                jj = np.arange(M - 1)
                T[jj, jj] = 0.5 * env.W * a.delta
                T[jj, jj + 1] = 0.5 * env.W * a.delta
                covk = Cov[sl, sl]
                nbt = Mt - 1
                tb = np.zeros(nbt * self.nint)
                m = min(len(dEk), nbt * self.nint)
                tb[:m] = dEk[:m]
                fr["xb"] = (np.arange(M - 1) + 0.5) * a.delta
                fr["bc"] = T @ ck
                fr["bs"] = np.sqrt(np.clip(np.diag(T @ covk @ T.T), 0.0, None))
                fr["xt"] = (np.arange(nbt) + 0.5) * a.delta
                fr["bt"] = tb.reshape(nbt, self.nint).sum(axis=1)
                if qw is not None:
                    Tw = T @ Wm
                    fr["bw"] = T @ qw
                    fr["bws"] = np.sqrt(np.clip(np.diag(Tw @ covk @ Tw.T), 0.0, None))
                    fr["btw"] = T @ tw
            frag.append(fr)
        gtr = S["geot"]
        return {"S": S, "geo": geo, "frag": frag, "dL": S["dL"], "notc": S["notc"], "M0": S["M0"],
                "chi2d0": S["chi2d0"], "chi2d1": S["chi2d1"],
                "ang": [angle_deg(geo["u"][k], gtr["u"][k]) for k in range(2)],
                "dx": geo["v"][0] - gtr["v"][0], "dz": geo["v"][2] - gtr["v"][2], "gtr": gtr}

    # ------------------------------------------------------------------ dessin
    def _select_ids(self):
        """pads selectionnes : clics, sinon selection automatique de view_pads.py"""
        pads = self.pads
        if self.selected:
            return [p for p in self.selected if p in pads]
        if self.show_all:
            cand = list(pads.keys())
        else:
            cand = [p for p, d in pads.items() if d["q"] >= self.threshold or d["flag"] == 0]
        ids = sorted(cand, key=lambda p: -pads[p]["q"])[:N_AUTO]
        cand = [p for p in cand if pads[p]["v"].size]
        if cand:
            prof = min(cand, key=lambda p: pads[p]["v"].min())
            if pads[prof]["v"].min() < 0.0 and prof not in ids:
                ids.append(prof)
        return ids

    def _draw_curves(self):
        """remplace le panneau des formes d'onde de view_pads.py par les profils dE/dx"""
        a = self.a
        ids = self._select_ids()
        self.shown = ids
        self.ax_qt.clear()
        self.ax_qt.set_visible(False)
        if self.ax_f is None:
            self.ax_f = [self.fig.add_axes([0.58, 0.56, 0.39, 0.36]), self.fig.add_axes([0.58, 0.08, 0.39, 0.36])]
        for ax in self.ax_f:
            ax.clear()
        i = self.event
        if i not in self.cache:
            self.ax_f[0].text(0.5, 0.5, "ajustement de l'evenement %d en cours..." % i, ha="center", va="center",
                              transform=self.ax_f[0].transAxes)
            self.fig.canvas.draw()
        res = self._fit(i)
        if "error" in res:
            self.ax_f[0].text(0.5, 0.5, "evenement %d ecarte :\n%s" % (i, res["error"]), ha="center", va="center",
                              transform=self.ax_f[0].transAxes, color="C3")
            return

        geo, gtr = res["geo"], res["gtr"]
        h = self.pad_size / 2.0

        # contours des pads selectionnes sur le plan d'anode (comme view_pads.py) + axes des fragments
        for n, pad_id in enumerate(ids):
            g = self.pad_geom.get(pad_id) if self.pad_geom else None
            if g is not None and g[3] > 0.0:
                patch = self._pad_patch(pad_id)
            else:
                x, y = self.pads[pad_id]["x"], self.pads[pad_id]["y"]
                patch = mpatches.Rectangle((x - h, y - h), self.pad_size, self.pad_size)
            patch.set_facecolor("none")
            patch.set_edgecolor(COULEURS[n % len(COULEURS)])
            patch.set_linewidth(2.2)
            patch.set_zorder(8)
            self.ax_pad.add_patch(patch)
        for k in range(2):
            for g_, style, lab in ((geo, "-", "axe ajuste" if k == 0 else None),
                                   (gtr, "--", "axe vrai" if k == 0 else None)):
                u = g_["u"][k]
                uxy = np.hypot(u[0], u[1])
                if uxy < 1.0e-9:
                    continue
                ln = res["frag"][k]["L"] * uxy
                x0, y0 = g_["v"][0], g_["v"][1]
                self.ax_pad.plot([x0, x0 + ln * u[0] / uxy], [y0, y0 + ln * u[1] / uxy], style,
                                 color="tab:green" if style == "-" else "k", lw=1.3, zorder=7, label=lab)
        self.ax_pad.legend(fontsize=8)

        # profils
        names = ["frag 0 (y > %.2f)" % a.y_plane, "frag 1 (y < %.2f)" % a.y_plane]
        for k in range(2):
            ax, fr = self.ax_f[k], res["frag"][k]
            M, Mt = fr["M"], fr["Mt"]
            if a.units == "h5":
                nb = len(fr["xb"])
                xmax = fr["xt"][-1] + 0.5 * a.delta
                if self.windowed and "bw" in fr:
                    ax.plot(fr["xt"], fr["bt"], ".", color="0.65", ms=4, label="verite par pas (non fenetree)")
                    ax.fill_between(fr["xb"], fr["bw"] - fr["bws"], fr["bw"] + fr["bws"], color="C2", alpha=0.2)
                    ax.plot(fr["xb"], fr["bw"], "o-", color="C2", ms=3,
                            label="ajuste, fenetre %.1f mm" % (10.0 * a.win))
                    ax.plot(fr["xb"], fr["btw"], "k.-", ms=3, label="verite, meme fenetre")
                else:
                    if len(fr["xt"]) > nb:
                        ax.plot(fr["xt"][nb - 1:], fr["bt"][nb - 1:], ".-", color="0.65", ms=4,
                                label="verite au-dela de la grille")
                    ax.fill_between(fr["xb"], fr["bc"] - fr["bs"], fr["bc"] + fr["bs"], color="C2", alpha=0.2)
                    ax.plot(fr["xb"], fr["bc"], "o-", color="C2", ms=3, label="ajuste (lam = %g)" % self.lam)
                    ax.plot(fr["xt"][:nb], fr["bt"][:nb], "k.-", ms=3, label="verite (somme des dE du .h5)")
                ylab = "dE par pas de %.1f mm (MeV)" % (10.0 * a.delta)
            else:
                ln = np.arange(M) * a.delta
                lt = np.arange(Mt) * a.delta
                xmax = max(lt[-1], ln[-1])
                if self.windowed and fr["qw"] is not None:
                    ax.plot(lt, fr["t"], ".", color="0.65", ms=4, label="verite aux noeuds (non fenetree)")
                    ax.fill_between(ln, fr["qw"] - fr["sw"], fr["qw"] + fr["sw"], color="C2", alpha=0.2)
                    ax.plot(ln, fr["qw"], "o-", color="C2", ms=3, label="ajuste, fenetre %.1f mm" % (10.0 * a.win))
                    ax.plot(ln, fr["tw"], "k.-", ms=3, label="verite, meme fenetre")
                else:
                    if Mt > M:
                        ax.plot(lt[M - 1:], fr["t"][M - 1:], ".-", color="0.65", ms=4,
                                label="verite au-dela de la grille")
                    ax.fill_between(ln, fr["c"] - fr["sc"], fr["c"] + fr["sc"], color="C2", alpha=0.2)
                    ax.plot(ln, fr["c"], "o-", color="C2", ms=3, label="ajuste (lam = %g)" % self.lam)
                    ax.plot(lt[:M], fr["t"][:M], "k.-", ms=3, label="verite (noeuds)")
                ylab = "e-/cm"
            tit = "%s [%s]  chi2/ndl %.2f  angle %.2f deg" % (names[k], fr["name"], res["chi2d1"], res["ang"][k])
            col = "k"
            if res["dL"][k] > 0.0:
                ax.axvspan((res["M0"][k] - 1) * a.delta, (M - 1) * a.delta, color="C1", alpha=0.12,
                           label="zone rallongee")
                tit += "\nGRILLE RALLONGEE (+%.1f cm)" % res["dL"][k]
                col = "C3"
            if res["notc"][k]:
                tit += "\nATTENTION : FIN NON CONTENUE"
                col = "C3"
            ax.set_title(tit, color=col, fontsize=8.5)
            ax.set_xlabel("l (cm)")
            ax.set_ylabel(ylab)
            ax.set_xlim(-0.1, xmax + 0.1)
            ax.axhline(0.0, color="0.7", lw=0.6)

        # pads selectionnes, repere le long de la trace
        v = geo["v"]
        for n, pad_id in enumerate(ids):
            d = self.pads[pad_id]
            k = 0 if d["y"] > a.y_plane else 1
            u = geo["u"][k]
            uxy = np.hypot(u[0], u[1])
            if uxy < 1.0e-9:
                continue
            ex, ey = u[0] / uxy, u[1] / uxy
            s = (d["x"] - v[0]) * ex + (d["y"] - v[1]) * ey
            dperp = abs(-(d["x"] - v[0]) * ey + (d["y"] - v[1]) * ex)
            l = s / uxy
            if l <= 0.0 or l > res["frag"][k]["L"]:
                continue
            col = COULEURS[n % len(COULEURS)]
            used = dperp <= a.pad_dist * self.env.pitch
            ax = self.ax_f[k]
            ax.axvline(l, color=col, lw=1.6, ls="-" if used else ":", zorder=2)
            ax.text(l, 1.0, "%d" % pad_id, color=col, fontsize=7, rotation=90, ha="right", va="top",
                    transform=ax.get_xaxis_transform())
        self.ax_f[0].legend(fontsize=7, loc="upper right")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("h5file")
    add_common_args(p)
    p.add_argument("--event", type=int, default=0)
    p.add_argument("--pad_file", type=str, default=None,
                   help="my_pad.txt : seulement pour un ancien .h5 sans groupe /pads")
    p.add_argument("--lam", type=float, default=1.0e-5, help="lissage du profil affiche (lam_geom : celui de la geometrie)")
    p.add_argument("--units", choices=["h5", "ecm"], default="h5",
                   help="h5 : dE par pas de delta, comme le .h5 (defaut) ; ecm : densite q(l) en e-/cm aux noeuds")
    a = p.parse_args()
    ProfileViewer(a)


if __name__ == "__main__":
    main()
