#!/usr/bin/env python3
"""
view_fit.py : meme viewer que view_pads.py (plan d'anode + courbes des pads), avec en plus, pour chaque pad
tracé, le FIT DE GABARIT de la charge, et la comparaison  pq (charge vraie)  /  Q_fit.

Usage:
    python view_fit.py output.h5 [--event N] [--pad_file my_pad.txt] [--width NS] [--fwhm NS] [--fixed]

Ce fichier doit etre dans le meme dossier que view_pads.py (il en herite : meme navigation, memes touches,
meme selection des pads par clic). Il lit le .h5 de ./process tel quel (aucun champ supplementaire).

Modele ajuste, pour chaque pad ecrit (echantillons s_k aux temps t_k, k = 0..n-1) :

        s(t) = Q * g(t - t0) + bruit

  g(t) = reponse de la chaine a une charge unitaire dont l'arrivee est la CONVOLUTION de deux rectangles :
         - la traversee du gap grille-anode, de duree FIXE wg = dga / ve_ga (lue dans les metadonnees ;
           option --width pour la changer) ;
         - l'etalement du a la trace dans le pad (trace inclinee : plusieurs z sur le meme pad), de duree
           wa >= 0 LIBRE, ajustee sur les donnees de chaque pad (option --fixed : wa = 0, comme avant).
         h = gaussienne causale du preampli (preamp_response = test_gauss : largeur lue dans
         /metadata/preamp_response_info, ou option --fwhm).
  t0   = instant de debut d'arrivee de la charge (parametre non lineaire)
  wa   = duree d'etalement de la trace (parametre non lineaire, grille puis affinage)
  Q    = charge du pad (parametre lineaire).

Le bruit est correle. Sa covariance C est reconstruite d'apres le modele du bruit de ./process (parallele filtre
par h, serie filtre par la derivee de h, bruit du CAN), avec l'ENC propre du pad si le .h5 a /pads/enc (modele
"capacitance") : l'ajustement est un moindres-carres generalise (maximum de vraisemblance pour un bruit
gaussien) :   Q = (g^T C^-1 s) / (g^T C^-1 g),   sigma_Q = 1 / sqrt(g^T C^-1 g).
(t0, wa) sont ceux qui minimisent le chi2 ; chi2/n = (s^T C^-1 s - Q^2/sigma_Q^2) / (n - 3) (n - 2 avec --fixed).

Limites a garder en tete :
  - Q est LIBRE (peut etre negatif), sans contrainte de signe ni de seuil. Les pads ecrits parce que leur bruit
    a franchi le seuil ont un Q_fit biaise vers le haut (effet de selection, voir la discussion).
  - Le gabarit est celui d'une charge COLLECTEE. Un pad d'induction (signal bipolaire, charge nette ~ 0) est mal
    decrit : son chi2/n est grand, a ne pas interpreter comme une charge.
  - Le profil d'arrivee reste schematique (deux rectangles) ; la vraie forme (diffusion, induction) est
    differente. Pour un pad peu charge, wa est mal contraint par le bruit : sa valeur n'est pas fiable.
  - La largeur ajustee n'est qu'un parametre de forme : elle est correlee a t0 et un peu a Q.
"""

import argparse
import re
import sys

import numpy as np
import h5py
import matplotlib.pyplot as plt

import view_pads as vp


# ---------------------------------------------------------------------------
# erf vectorisee (Abramowitz-Stegun 7.1.26, erreur < 1.5e-7) : evite de dependre de scipy
# ---------------------------------------------------------------------------
def _erf(x):
    x = np.asarray(x, dtype=float)
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return s * y


class FitModel:
    """Gabarit g(t) et covariance du bruit ; construit une fois pour tout le fichier."""

    def __init__(self, attrs, width_ns=None, fwhm_ns=None):
        self.dt = float(attrs["delta_t"])
        self.R = int(attrs["adc_ratio"])
        # largeur de la gaussienne du preampli
        if fwhm_ns is None:
            info = attrs["preamp_response_info"] if "preamp_response_info" in attrs else ""
            if isinstance(info, bytes):
                info = info.decode()
            m = re.search(r"fwhm\s*=\s*([0-9.eE+-]+)\s*ns", str(info))
            if m is None:
                raise SystemExit("[view_fit] largeur de la gaussienne introuvable dans /metadata/preamp_response_info "
                                 "(contenu : %r) : utiliser --fwhm NS" % (info,))
            fwhm_ns = float(m.group(1))
        self.fwhm = float(fwhm_ns)
        self.sigma = self.fwhm / 2.3548
        # largeur du profil d'arrivee = traversee du gap grille-anode
        if width_ns is None:
            if "dga" not in attrs or "ve_ga" not in attrs:
                raise SystemExit("[view_fit] dga / ve_ga absents des metadonnees : utiliser --width NS")
            width_ns = float(attrs["dga"]) / float(attrs["ve_ga"])
        self.width = float(width_ns)
        if self.width <= 0.0:
            raise SystemExit("[view_fit] la largeur d'arrivee doit etre > 0")

        # h echantillonnee au pas dt, comme dans ./process (pour la covariance du bruit et la normalisation)
        s = self.sigma
        t0h = 4.0 * s
        n = int(np.ceil((t0h + 4.5 * s) / self.dt)) + 1
        k = np.arange(n)
        h = np.exp(-0.5 * ((k * self.dt - t0h) / s) ** 2)
        self.hnorm = float(h.max())          # ./process normalise h a 1 au maximum de l'echantillonnage
        h = h / self.hnorm
        # integrale de h (pic = 1), en ns : une charge Q donne une sortie dont l'integrale vaut Q * self.area.
        # Reconstruite ici a partir de h echantillonnee ; avec une reponse mesuree, c'est la valeur a remplacer.
        self.area = float(np.sum(h)) * self.dt
        L = len(h)
        hd = np.zeros(L + 1)
        for j in range(L + 1):
            cur = h[j] if j < L else 0.0
            prev = h[j - 1] if j > 0 else 0.0
            hd[j] = cur - prev
        # autocovariances (par bin) : parallele = sigma_par^2 * sum h h ; serie = sigma_ser^2 * sum dh dh
        self.A_par = np.correlate(h, h, mode="full")[len(h) - 1:]
        self.A_ser = np.correlate(hd, hd, mode="full")[len(hd) - 1:]
        self.sig_par = float(attrs["sigma_par"])
        self.sig_ser = float(attrs["sigma_ser"])
        self.enc_can = float(attrs["enc_can"])
        self.enc_ser_ref = float(attrs["enc_ser"])
        self.enc_ref = float(attrs["enc_total"])
        self._cinv = {}

    # --- covariance du bruit de n echantillons consecutifs, pour un pad de facteur serie ser_scale ---
    def cov(self, n, ser_scale):
        """matrice de covariance (n x n) du bruit de n echantillons consecutifs"""
        lags = np.arange(n) * self.R
        c = np.zeros(n)
        for k in range(n):
            l = lags[k]
            ap = self.A_par[l] if l < len(self.A_par) else 0.0
            asr = self.A_ser[l] if l < len(self.A_ser) else 0.0
            c[k] = self.sig_par ** 2 * ap + (ser_scale * self.sig_ser) ** 2 * asr
        c[0] += self.enc_can ** 2
        idx = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
        return c[idx]

    def cinv(self, n, ser_scale):
        key = (n, round(ser_scale, 5))
        if key in self._cinv:
            return self._cinv[key]
        Ci = np.linalg.inv(self.cov(n, ser_scale))
        self._cinv[key] = Ci
        return Ci

    def integral(self, t, v, ser_scale):
        """Charge par integrale de la fenetre ecrite : Q = somme(v) * dt_ech / aire(h).
        Insensible a la forme d'arrivee et a l'induction (charge nette nulle) ; suppose la fenetre
        assez longue pour contenir tout le signal. Renvoie (Q, sigma_Q) ; sigma_Q = bruit seul."""
        n = len(v)
        if n < 2:
            return None
        dts = float(t[1] - t[0])
        one = np.ones(n)
        var = float(one @ self.cov(n, ser_scale) @ one)
        return float(np.sum(v)) * dts / self.area, np.sqrt(var) * dts / self.area

    def ser_scale_pad(self, enc_pad):
        """facteur sur le bruit serie d'un pad d'ENC totale enc_pad (modele capacitance) ; 1 sans /pads/enc"""
        if enc_pad is None:
            return 1.0
        v = self.enc_ser_ref ** 2 + enc_pad ** 2 - self.enc_ref ** 2
        return float(np.sqrt(max(v, 0.0)) / self.enc_ser_ref)

    # --- gabarit : charge unitaire arrivant uniformement sur [t0, t0 + w] ---
    def template(self, t, t0):
        """t : (n,) ; t0 : scalaire ou (m,) -> renvoie (m, n) (ou (n,) pour un scalaire)"""
        t0 = np.asarray(t0, dtype=float)
        scalar = (t0.ndim == 0)
        t0 = np.atleast_1d(t0)
        s = self.sigma
        x1 = (t[None, :] - t0[:, None] - 4.0 * s) / (s * np.sqrt(2.0))
        x2 = (t[None, :] - t0[:, None] - self.width - 4.0 * s) / (s * np.sqrt(2.0))
        g = s * np.sqrt(np.pi / 2.0) * (_erf(x1) - _erf(x2)) / (self.width * self.hnorm)
        return g[0] if scalar else g

    # --- gabarit a deux rectangles : gap (wg fixe) puis etalement de trace (wa libre) ---
    @staticmethod
    def _K(x):
        """primitive de la fonction de repartition normale : K(x) = x Phi(x) + phi(x)"""
        phi = np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
        return x * 0.5 * (1.0 + _erf(x / np.sqrt(2.0))) + phi

    def template2(self, t, t0, wa):
        """charge unitaire arrivant avec le profil (rectangle wg) conv. (rectangle wa) a partir de t0.
        t : (n,) ; t0 : scalaire ou (m,) ; wa : scalaire. Renvoie (m, n) (ou (n,) pour t0 scalaire)."""
        if wa < 1.0:
            return self.template(t, t0)
        t0 = np.asarray(t0, dtype=float)
        scalar = (t0.ndim == 0)
        t0 = np.atleast_1d(t0)
        s = self.sigma
        x0 = (t[None, :] - t0[:, None] - 4.0 * s) / s
        wg = self.width
        K = self._K
        d = K(x0) - K(x0 - wa / s) - K(x0 - wg / s) + K(x0 - (wa + wg) / s)
        g = s * s * np.sqrt(2.0 * np.pi) * d / (wg * wa * self.hnorm)
        return g[0] if scalar else g

    def fit(self, t, v, ser_scale, free_width=True, frac_min=0.8):
        """Ajuste Q, t0 (et wa si free_width) sur les echantillons (t, v).
        Contrainte de fenetre : au moins la fraction frac_min de l'aire du gabarit doit tomber dans la
        fenetre ecrite (sinon le fit place la queue du gabarit au bord avec une charge enorme).
        Renvoie un dict (Q, sigQ, t0, wa, chi2n) ou None."""
        n = len(v)
        if n < 4:
            return None
        Ci = self.cinv(n, ser_scale)
        sCs = float(v @ Ci @ v)
        dts = float(t[1] - t[0])
        aire = self.sigma * np.sqrt(2.0 * np.pi) / self.hnorm      # aire du gabarit complet (charge unitaire)

        def scan(t0s, was):
            best = None
            for wa in was:
                G = self.template2(t, t0s, wa)                # (m, n)
                fr = G.sum(axis=1) * dts / aire               # fraction de l'aire dans la fenetre
                W = G @ Ci
                num = W @ v
                den = np.sum(W * G, axis=1)
                ok = (den > 1.0e-12) & (fr >= frac_min)
                if not ok.any():
                    continue
                gain = np.where(ok, num * num / np.where(den > 1.0e-12, den, 1.0), -1.0)   # reduction de chi2
                k = int(np.argmax(gain))
                if best is None or gain[k] > best[0]:
                    best = (float(gain[k]), float(t0s[k]), float(wa), float(num[k]), float(den[k]))
            return best

        lo = t[0] - 150.0
        hi = t[-1]
        if free_width:
            b = scan(np.arange(lo, hi, 8.0), [0.0, 25.0, 50.0, 100.0, 150.0, 250.0, 400.0, 600.0])
            if b is None:
                return None
            g, c0, wb, num, den = b
            was = [0.0, 10.0, 20.0, 35.0] if wb < 1.0 else [wb * f for f in (0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5)]
        else:
            b = scan(np.arange(lo, hi, 4.0), [0.0])
            if b is None:
                return None
            g, c0, wb, num, den = b
            was = [0.0]
        b = scan(np.arange(c0 - 8.0, c0 + 8.0001, 1.0), was)
        if b is None:
            return None
        g, c1, wb, num, den = b
        if g < 0.0:
            return None
        npar = 3 if free_width else 2
        return {"Q": num / den, "sigQ": 1.0 / np.sqrt(den), "t0": c1, "wa": wb,
                "chi2n": float((sCs - g) / max(n - npar, 1))}


# ---------------------------------------------------------------------------
class ViewerFit(vp.ViewerType2):
    def __init__(self, h5_path, pad_file=None, start_event=0, width_ns=None, fwhm_ns=None, free_width=True):
        self.free_width = free_width
        # parametres du fit : AVANT le constructeur du viewer (il se termine par plt.show(), bloquant)
        with h5py.File(h5_path, "r") as f:
            attrs = f["/metadata"].attrs
            self.fm = FitModel(attrs, width_ns, fwhm_ns)
            self.enc_pad = None
            if "/pads/enc" in f:
                ids = np.array(f["/pads/id"], dtype=int)
                enc = np.array(f["/pads/enc"], dtype=float)
                self.enc_pad = dict((int(ids[k]), float(enc[k])) for k in range(len(ids)))
        print("[view_fit] gabarit : gaussienne fwhm = %.0f ns, gap wg = %.0f ns, etalement de trace wa %s ; "
              "bruit correle %s" % (self.fm.fwhm, self.fm.width, "libre (ajuste par pad)" if free_width else "= 0 (--fixed)",
                                    "avec l'ENC de chaque pad (/pads/enc)" if self.enc_pad else "ENC unique (modele uniform)"))
        print("[view_fit] integrale de h (pic = 1) = %.1f ns : charge par integrale Q_int = somme(v) x dt_ech / %.1f" % (self.fm.area, self.fm.area))
        vp.ViewerType2.__init__(self, h5_path, pad_file=pad_file, start_event=start_event)

    def _load_event(self, i):
        pads = vp.ViewerType2._load_event(self, i)
        for pad_id, d in pads.items():
            enc = self.enc_pad.get(pad_id) if self.enc_pad else None
            sc = self.fm.ser_scale_pad(enc)
            d["fit"] = self.fm.fit(d["t"], d["v"], sc, self.free_width)
            d["int"] = self.fm.integral(d["t"], d["v"], sc)
        self._mark_orphans(pads)
        return pads

    def _mark_orphans(self, pads):
        """Pad orphelin = aucun autre pad ecrit dans le voisinage immediat (8 voisins : ecart en x et en y
        <= 1.1 pas de pad). Un tel pad ne peut pas appartenir a une trace : bruit (ou fantome), exclu des sommes.
        Les pads ronds (r_out > 0) ne sont jamais declares orphelins (voisinage non defini)."""
        if getattr(self, "_pitch", None) is None:
            xs, ys = [], []
            for (px, py, r_in, r_out, sx, sy) in self.pad_geom.values():
                if r_out == 0.0:
                    xs.append(px)
                    ys.append(py)
            xs = np.array(xs)
            ys = np.array(ys)
            dd = np.hypot(xs[:, None] - xs[None, :], ys[:, None] - ys[None, :])
            np.fill_diagonal(dd, 1.0e9)
            self._pitch = float(dd.min())
        ids = list(pads.keys())
        px = np.array([pads[p]["x"] for p in ids])
        py = np.array([pads[p]["y"] for p in ids])
        rond = np.array([self.pad_geom[p][3] > 0.0 for p in ids])
        adj = np.maximum(np.abs(px[:, None] - px[None, :]), np.abs(py[:, None] - py[None, :])) <= 1.1 * self._pitch
        adj = adj & ~np.eye(len(ids), dtype=bool) & ~rond[:, None] & ~rond[None, :]
        has_nb = adj.any(axis=1) | rond
        for k, p in enumerate(ids):
            pads[p]["orphan"] = not bool(has_nb[k])

    def _fit_label(self, d):
        r = d["fit"]
        if r is None:
            return "fit impossible"
        return "fit=%.0f +- %.0f  duree=%.0f ns  chi2/n=%.1f" % (r["Q"], r["sigQ"], self.fm.width + r["wa"], r["chi2n"])

    def _int_label(self, d):
        r = d["int"]
        return "int=%.0f +- %.0f" % (r[0], r[1]) if r is not None else "int impossible"

    def _draw_curves(self):
        """Comme ViewerType2._draw_curves, avec la courbe ajustee de chaque pad (tirets) et pq / Q_fit en legende."""
        pads = self.pads
        if self.selected:
            ids = [p for p in self.selected if p in pads]
        else:
            if self.show_all:
                cand = list(pads.keys())
            else:
                cand = [p for p, d in pads.items() if d["q"] >= self.threshold or d["flag"] == 0]
            ids = sorted(cand, key=lambda p: -pads[p]["q"])[:vp.N_AUTO]
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
            col = vp.COULEURS[n % len(vp.COULEURS)]
            lab = "pad %d [%s]  pq=%.0f  %s  %s%s" % (pad_id, vp.FLAG_TEXTE[d["flag"]], d["q"], self._int_label(d),
                                                      self._fit_label(d), "  ORPHELIN (exclu)" if d["orphan"] else "")
            ax.plot(d["t"], d["v"], color=col, lw=0, marker=".", ms=5, label=lab)
            r = d["fit"]
            if r is not None:
                tt = np.linspace(d["t"][0], d["t"][-1], 400)
                ax.plot(tt, r["Q"] * self.fm.template2(tt, r["t0"], r["wa"]), color=col, lw=1.4, ls="--")

        ax.axhline(0.0, color="0.6", lw=0.6)
        if self.noise_on:
            ax.axhline(self.read_thr, color="0.3", lw=1.0, ls="--")
            ax.text(0.01, self.read_thr, " seuil de lecture (%.0f e-)" % self.read_thr, color="0.3",
                    fontsize=8, va="bottom", transform=ax.get_yaxis_transform())
        ax.set_xlabel("t (ns)")
        ax.set_ylabel("sortie du preampli (e- equiv.)")
        ax.set_title("Points : echantillons ; tirets : fit de gabarit (duree d'arrivee = gap %.0f ns + etalement ajuste) - clic : ajouter / retirer"
                     % self.fm.width, fontsize=8.5)
        ax.legend(fontsize=7.5, loc="upper right")

        h = self.pad_size / 2.0
        for n, pad_id in enumerate(ids):
            g = self.pad_geom.get(pad_id) if self.pad_geom else None
            if g is not None and g[3] > 0.0:
                patch = self._pad_patch(pad_id)
            else:
                x, y = pads[pad_id]["x"], pads[pad_id]["y"]
                patch = vp.mpatches.Rectangle((x - h, y - h), self.pad_size, self.pad_size)
            patch.set_facecolor("none")
            patch.set_edgecolor(vp.COULEURS[n % len(vp.COULEURS)])
            patch.set_linewidth(2.2)
            patch.set_zorder(8)
            self.ax_pad.add_patch(patch)

    def _ke_lines(self, i):
        s = vp.ViewerType2._ke_lines(self, i)
        # sommes sur les pads NON orphelins (aucune information de verite : le flag fantome n'est pas utilise)
        ok = [d for d in self.pads.values() if not d["orphan"] and d["int"] is not None]
        n_orph = sum(1 for d in self.pads.values() if d["orphan"])
        if not ok:
            return s
        q_int = sum(d["int"][0] for d in ok)
        q_fit = sum(d["fit"]["Q"] for d in ok if d["fit"] is not None)
        q_true = sum(d["q"] for d in self.pads.values() if d["flag"] != 0)      # reference : verite
        line = ("KE (hors %d orphelins) : integrale somme Q_int x W = %.3f MeV | fit somme Q_fit x W = %.3f MeV | "
                "somme pq x W = %.3f MeV (verite)" % (n_orph, q_int * self.W, q_fit * self.W, q_true * self.W))
        return (s + "\n" + line) if s else line


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("h5file")
    parser.add_argument("--event",    type=int,   default=0)
    parser.add_argument("--pad_file", type=str,   default=None,
                        help="my_pad.txt : seulement pour un ancien .h5 sans groupe /pads")
    parser.add_argument("--width",    type=float, default=None,
                        help="duree d'arrivee de la charge (ns) ; defaut : dga / ve_ga des metadonnees")
    parser.add_argument("--fixed",    action="store_true",
                        help="pas d'etalement de trace ajuste : arrivee = gap seul (ancien comportement)")
    parser.add_argument("--fwhm",     type=float, default=None,
                        help="largeur (ns) de la gaussienne du preampli ; defaut : lue dans preamp_response_info")
    args = parser.parse_args()
    ViewerFit(args.h5file, pad_file=args.pad_file, start_event=args.event, width_ns=args.width, fwhm_ns=args.fwhm,
              free_width=not args.fixed)


if __name__ == "__main__":
    main()
