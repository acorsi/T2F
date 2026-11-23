#!/usr/bin/env python3
"""
view_fit.py : meme viewer que view_pads.py (plan d'anode + courbes des pads), avec en plus, pour chaque pad
tracé, le FIT DE GABARIT de la charge, et la comparaison  pq (charge vraie)  /  Q_fit.

Usage:
    python view_fit.py output.h5 [--event N] [--pad_file my_pad.txt] [--width NS] [--fwhm NS] [--fixed] [--no_check]

Ce fichier doit etre dans le meme dossier que view_pads.py (il en herite : meme navigation, memes touches,
meme selection des pads par clic). Il lit le .h5 de ./process tel quel (aucun champ supplementaire).

Reponse du preampli h(t) : reconstruite AUTOMATIQUEMENT d'apres /metadata/preamp_response :
  - test_gauss : gaussienne causale, fwhm lue dans /metadata/preamp_response_info (ou option --fwhm) ;
  - minos      : (t/tau)^3 exp(-3t/tau) sin(t/tau), temps du maximum lu dans /metadata/response_peak_ns.
  h est reconstruite comme dans preamp_response.hpp (meme echantillonnage, meme normalisation, meme troncature),
  puis verifiee : nombre de taps = response_n_taps, sqrt(somme h^2) = enc_par / sigma_par et
  sqrt(somme dh^2) = enc_ser / sigma_ser (arret si l'ecart depasse 1e-3, sauf option --no_check).

Modele ajuste, pour chaque pad ecrit (echantillons s_k aux temps t_k, k = 0..n-1) :

        s(t) = Q * g(t - t0) + bruit

  g(t) = reponse de la chaine a une charge unitaire dont l'arrivee est la CONVOLUTION de deux rectangles :
         - la traversee du gap grille-anode, de duree FIXE wg = dga / ve_ga (lue dans les metadonnees ;
           option --width pour la changer) ;
         - l'etalement du a la trace dans le pad (trace inclinee : plusieurs z sur le meme pad), de duree
           wa >= 0 LIBRE, ajustee sur les donnees de chaque pad (option --fixed : wa = 0).
         La sortie est calculee numeriquement avec la reponse h(t) continue (primitives de h sur une grille
         fine), la meme methode pour toutes les reponses.
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
  - Quand le temps de pic de h est grand devant la duree du signal (minos, tp >~ 300 ns), la sortie ressemble a
    h decalee : wa est alors presque indetermine (Q reste bien determine).
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
# lecture d'un attribut texte du .h5 (str ou bytes selon la version de h5py)
# ---------------------------------------------------------------------------
def _attr_str(attrs, key):
    if key not in attrs:
        return None
    v = attrs[key]
    if isinstance(v, bytes):
        v = v.decode()
    return str(v)


# ---------------------------------------------------------------------------
# Reponse du preampli : meme construction que preamp_response.hpp
# ---------------------------------------------------------------------------
def _minos_u_peak():
    """position du maximum de u^3 exp(-3u) sin(u), en unites de tau (1.1664...) : meme bissection que le C++"""
    a, b = 0.5, 1.5
    for i in range(200):
        m = 0.5 * (a + b)
        f = 3.0 / m - 3.0 + np.cos(m) / np.sin(m)
        if f > 0.0:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def _minos_envelope(u):
    return u * u * u * np.exp(-3.0 * u)


def _minos_form(u):
    return _minos_envelope(u) * np.sin(u)


class Response:
    """Reponse h du preampli : tableau h[k] a t = k dt (comme dans ./process) et fonction continue h_cont(t)
    (meme formule, meme constante de normalisation), nulle pour t < 0."""

    def __init__(self, attrs, dt, fwhm_ns=None):
        self.dt = dt
        name = _attr_str(attrs, "preamp_response")
        if name is None:
            raise SystemExit("[view_fit] /metadata/preamp_response absent du .h5 : fichier trop ancien ?")
        self.name = name

        if name == "test_gauss":
            if fwhm_ns is None:
                info = _attr_str(attrs, "preamp_response_info") or ""
                m = re.search(r"fwhm\s*=\s*([0-9.eE+-]+)\s*ns", info)
                if m is None:
                    raise SystemExit("[view_fit] largeur de la gaussienne introuvable dans /metadata/preamp_response_info "
                                     "(contenu : %r) : utiliser --fwhm NS" % (info,))
                fwhm_ns = float(m.group(1))
            self.fwhm = float(fwhm_ns)
            s = self.fwhm / 2.3548200450309493
            t0 = 4.0 * s
            n = int(np.ceil((t0 + 4.5 * s) / dt)) + 1
            self._raw = lambda t: np.exp(-0.5 * ((t - t0) / s) ** 2)
            self.descr = "test_gauss : gaussienne causale, fwhm = %.1f ns, pic a %.1f ns" % (self.fwhm, t0)
            self.peak_ns = t0
        elif name == "minos":
            if fwhm_ns is not None:
                raise SystemExit("[view_fit] --fwhm ne s'applique qu'a la reponse test_gauss (ce fichier est minos)")
            if "response_peak_ns" not in attrs:
                raise SystemExit("[view_fit] /metadata/response_peak_ns absent du .h5 (reponse minos)")
            peak = float(attrs["response_peak_ns"])
            up = _minos_u_peak()
            tau = peak / up
            hpk = float(_minos_form(up))
            a, b = up, 30.0                       # coupure de la queue : meme bissection que le C++
            for i in range(200):
                m = 0.5 * (a + b)
                if _minos_envelope(m) / hpk > 1.0e-4:
                    a = m
                else:
                    b = m
            n = int(np.ceil(b * tau / dt)) + 1
            self._raw = lambda t: _minos_form(t / tau)
            self.tau = tau
            self.descr = "minos : (t/tau)^3 exp(-3t/tau) sin(t/tau), temps du maximum = %.1f ns, tau = %.2f ns" % (peak, tau)
            self.peak_ns = peak
            if "response_tau_ns" in attrs and abs(float(attrs["response_tau_ns"]) - tau) > 1.0e-6 * tau:
                raise SystemExit("[view_fit] tau reconstruit (%.6f ns) different de /metadata/response_tau_ns (%.6f ns)"
                                 % (tau, float(attrs["response_tau_ns"])))
        else:
            raise SystemExit("[view_fit] reponse du preampli inconnue : %r (connues : test_gauss, minos)" % name)

        # h echantillonnee, normalisee a 1 au maximum de l'ECHANTILLONNAGE (comme le C++)
        k = np.arange(n)
        h = self._raw(k * dt)
        self.hmax = float(h.max())
        self.h = h / self.hmax
        self.n_taps = n

        # grille fine pour la reponse continue et ses deux primitives
        self.step = 1.0
        self.tf = np.arange(0.0, (n - 1) * dt + self.step, self.step)
        hf = self.hcont(self.tf)
        self.H1 = np.concatenate(([0.0], np.cumsum(0.5 * (hf[1:] + hf[:-1]) * self.step)))
        self.H2 = np.concatenate(([0.0], np.cumsum(0.5 * (self.H1[1:] + self.H1[:-1]) * self.step)))
        self.area = float(self.H1[-1])          # integrale de h continue (pic = 1), ns

    def hcont(self, t):
        t = np.asarray(t, dtype=float)
        return np.where(t >= 0.0, self._raw(np.maximum(t, 0.0)) / self.hmax, 0.0)

    # primitives, nulles pour x <= 0 ; au-dela de la grille h = 0 : H1 constante, H2 lineaire
    def prim1(self, x):
        return np.interp(x, self.tf, self.H1)

    def prim2(self, x):
        y = np.interp(x, self.tf, self.H2)
        over = x > self.tf[-1]
        if over.any():
            y = y + np.where(over, self.H1[-1] * (x - self.tf[-1]), 0.0)
        return y


class FitModel:
    """Gabarit g(t) et covariance du bruit ; construit une fois pour tout le fichier."""

    def __init__(self, attrs, width_ns=None, fwhm_ns=None, check=True):
        self.dt = float(attrs["delta_t"])
        self.R = int(attrs["adc_ratio"])
        self.resp = Response(attrs, self.dt, fwhm_ns)
        # largeur du profil d'arrivee = traversee du gap grille-anode
        if width_ns is None:
            if "dga" not in attrs or "ve_ga" not in attrs:
                raise SystemExit("[view_fit] dga / ve_ga absents des metadonnees : utiliser --width NS")
            width_ns = float(attrs["dga"]) / float(attrs["ve_ga"])
        self.width = float(width_ns)
        if self.width <= 0.0:
            raise SystemExit("[view_fit] la largeur d'arrivee doit etre > 0")

        h = self.resp.h
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
        self.enc_par_ref = float(attrs["enc_par"])
        self._cinv = {}
        if check:
            self._check_response(attrs)

    def _check_response(self, attrs):
        """h reconstruite = h du C++ ? (nombre de taps, et normes qui ont servi a fixer sigma_par et sigma_ser)"""
        probs = []
        nt = int(attrs["response_n_taps"]) if "response_n_taps" in attrs else None
        if nt is not None and nt != self.resp.n_taps:
            probs.append("nombre de taps : %d reconstruits, %d dans /metadata/response_n_taps" % (self.resp.n_taps, nt))
        h = self.resp.h
        A0 = float(self.A_par[0])
        B0 = float(self.A_ser[0])
        if self.sig_par > 0.0:
            r = np.sqrt(A0) / (self.enc_par_ref / self.sig_par) - 1.0
            if abs(r) > 1.0e-3:
                probs.append("sqrt(somme h^2) : ecart relatif %.2e avec enc_par / sigma_par" % r)
        if self.sig_ser > 0.0:
            r = np.sqrt(B0) / (self.enc_ser_ref / self.sig_ser) - 1.0
            if abs(r) > 1.0e-3:
                probs.append("sqrt(somme dh^2) : ecart relatif %.2e avec enc_ser / sigma_ser" % r)
        if probs:
            raise SystemExit("[view_fit] la reponse reconstruite (%s) ne correspond pas au fichier :\n  - %s\n"
                             "  (option --no_check pour passer outre)" % (self.resp.descr, "\n  - ".join(probs)))

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

    def ser_scale_pad(self, enc_pad):
        """facteur sur le bruit serie d'un pad d'ENC totale enc_pad (modele capacitance) ; 1 sans /pads/enc"""
        if enc_pad is None:
            return 1.0
        v = self.enc_ser_ref ** 2 + enc_pad ** 2 - self.enc_ref ** 2
        return float(np.sqrt(max(v, 0.0)) / self.enc_ser_ref)

    # --- gabarit : charge unitaire arrivant avec le profil (rectangle wg) conv. (rectangle wa) a partir de t0 ---
    def template2(self, t, t0, wa):
        """t : (n,) ; t0 : scalaire ou (m,) ; wa : scalaire (< 1 ns : pas d'etalement). Renvoie (m, n) (ou (n,) pour t0 scalaire).
        g = h (*) p, p = (rect wg) (*) (rect wa) / (wg wa) ; avec H1, H2 primitives de h :
        g = [H2(x) - H2(x-wa) - H2(x-wg) + H2(x-wa-wg)] / (wg wa),  x = t - t0   (wa = 0 : [H1(x) - H1(x-wg)] / wg)."""
        t0 = np.asarray(t0, dtype=float)
        scalar = (t0.ndim == 0)
        t0 = np.atleast_1d(t0)
        x = t[None, :] - t0[:, None]
        wg = self.width
        r = self.resp
        if wa < 1.0:
            g = (r.prim1(x) - r.prim1(x - wg)) / wg
        else:
            g = (r.prim2(x) - r.prim2(x - wa) - r.prim2(x - wg) + r.prim2(x - wa - wg)) / (wg * wa)
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
        aire = self.resp.area                                       # aire du gabarit complet (charge unitaire)

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
    def __init__(self, h5_path, pad_file=None, start_event=0, width_ns=None, fwhm_ns=None, free_width=True, check=True):
        self.free_width = free_width
        # parametres du fit : AVANT le constructeur du viewer (il se termine par plt.show(), bloquant)
        with h5py.File(h5_path, "r") as f:
            attrs = f["/metadata"].attrs
            self.fm = FitModel(attrs, width_ns, fwhm_ns, check)
            self.enc_pad = None
            if "/pads/enc" in f:
                ids = np.array(f["/pads/id"], dtype=int)
                enc = np.array(f["/pads/enc"], dtype=float)
                self.enc_pad = dict((int(ids[k]), float(enc[k])) for k in range(len(ids)))
        print("[view_fit] reponse : %s (%d taps)" % (self.fm.resp.descr, self.fm.resp.n_taps))
        print("[view_fit] gabarit : gap wg = %.0f ns, etalement de trace wa %s ; bruit correle %s"
              % (self.fm.width, "libre (ajuste par pad)" if free_width else "= 0 (--fixed)",
                 "avec l'ENC de chaque pad (/pads/enc)" if self.enc_pad else "ENC unique (modele uniform)"))
        vp.ViewerType2.__init__(self, h5_path, pad_file=pad_file, start_event=start_event)

    def _load_event(self, i):
        pads = vp.ViewerType2._load_event(self, i)
        for pad_id, d in pads.items():
            enc = self.enc_pad.get(pad_id) if self.enc_pad else None
            sc = self.fm.ser_scale_pad(enc)
            d["fit"] = self.fm.fit(d["t"], d["v"], sc, self.free_width)
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
            lab = "pad %d [%s]  pq=%.0f  %s%s" % (pad_id, vp.FLAG_TEXTE[d["flag"]], d["q"], self._fit_label(d),
                                                  "  ORPHELIN (exclu)" if d["orphan"] else "")
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
        ok = [d for d in self.pads.values() if not d["orphan"] and d["fit"] is not None]
        n_orph = sum(1 for d in self.pads.values() if d["orphan"])
        if not ok:
            return s
        q_fit = sum(d["fit"]["Q"] for d in ok)
        q_true = sum(d["q"] for d in self.pads.values() if d["flag"] != 0)      # reference : verite
        line = ("KE (hors %d orphelins) : fit somme Q_fit x W = %.3f MeV | "
                "somme pq x W = %.3f MeV (verite)" % (n_orph, q_fit * self.W, q_true * self.W))
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
                        help="pas d'etalement de trace ajuste : arrivee = gap seul")
    parser.add_argument("--fwhm",     type=float, default=None,
                        help="test_gauss seulement : largeur (ns) de la gaussienne ; defaut : lue dans preamp_response_info")
    parser.add_argument("--no_check", action="store_true",
                        help="ne pas verifier la reponse reconstruite (taps, sigma_par, sigma_ser) contre les metadonnees")
    args = parser.parse_args()
    ViewerFit(args.h5file, pad_file=args.pad_file, start_event=args.event, width_ns=args.width, fwhm_ns=args.fwhm,
              free_width=not args.fixed, check=not args.no_check)


if __name__ == "__main__":
    main()
