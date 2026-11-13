#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Niveau 1 : compare la table Psi(u, v, tau, d) du code a une reference EXACTE
# calculee independamment, et trace simu versus exact.
#
# Usage :  python3 test_level1.py A:2 B:2 ...        (CAS:R, memes arguments que ./test_level1)
#
# Modele defini dans psi_builder.hpp (c'est lui qu'on verifie, pas la realite) :
#   Psi(u,v,tau) = Int d(delta) N(delta; 0, sigma_t)  Lambda(u, v, h)    avec  h = g - ve_ga (tau - delta)
#   h >= g : 0 (pas encore entre dans le gap) ;  h <= 0 : valeur a l'anode (collecte)
#   Lambda(u,v,h) = phi(h) lisse lateralement par une gaussienne de variance
#                   sigma(h)^2 = DT_kg^2 d + DT_ga^2 (g - h)
# Reference exacte de Lambda : transformee de Fourier 2D (le lissage gaussien multiplie la
# transformee par exp(-sigma^2 k^2 / 2)), puis integrale en delta par quadrature de Gauss-Legendre.
# Rien n'est repris du code (ni images, ni angles solides, ni table, ni sigma_min).
# ---------------------------------------------------------------------------
import csv
import sys
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import CubicSpline
from scipy.special import ndtr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOL = 1.0e-3         # tolerance ABSOLUE sur Psi (fraction de la charge du nuage, de 0 a 1)
DK = 0.5             # pas en k de la reference (cm^-1)
NH = 48              # niveaux en h pour la spline de Lambda
OUT = "out/"
XG, WG = leggauss(96)


def read_meta(base):
    M = {}
    with open(base + "_meta.txt") as fh:
        for line in fh:
            k, v = line.strip().split("=")
            try:
                M[k] = float(v)
            except ValueError:
                M[k] = v
    return M


def read_rows(base):
    groups = {}
    with open(base + ".csv") as fh:
        for r in csv.DictReader(fh):
            kind = r["kind"]
            base_kind, sub = kind.rsplit("_", 1)
            key = (base_kind, float(r["u"]), float(r["v"]), float(r["d"]))
            g = groups.setdefault(key, {"tau": [], "psi": [], "psig": [], "sub": []})
            g["tau"].append(float(r["tau"]))
            g["psi"].append(float(r["psi"]))
            g["psig"].append(float(r["psi_g"]))
            g["sub"].append(sub)
    for g in groups.values():
        for k in ("tau", "psi", "psig"):
            g[k] = np.array(g[k])
        g["sub"] = np.array(g["sub"])
    return groups


_kcache = {}


def kmatrix(n):
    if n not in _kcache:
        k = (np.arange(n) + 0.5) * DK
        _kcache[n] = (k, np.hypot(k[:, None], k[None, :]))
    return _kcache[n]


def lambda_spline(M, u, v, d):
    """Lambda(u,v,h) pour h dans [0,g], par Fourier exact, puis spline cubique en h."""
    a, g = M["a"], M["g"]
    s0sq = (M["DT_kg"]) ** 2 * d
    hs = np.linspace(0.0, g, NH + 1)
    vals = np.zeros(NH + 1)
    for j, h in enumerate(hs):
        sig2 = s0sq + M["DT_ga"] ** 2 * (g - h)
        kmax = 8.0 / np.sqrt(sig2)
        if h > 0.0:
            kmax = min(kmax, 25.0 / h)
        n = int(kmax / DK) + 1
        k, K = kmatrix(n)
        if h == 0.0:
            F = 1.0
        else:
            F = np.exp(-K * h) * (1.0 - np.exp(-2.0 * K * (g - h))) / (1.0 - np.exp(-2.0 * K * g))
        Mx = F * np.exp(-0.5 * sig2 * K ** 2)
        A = np.sin(k * a) * np.cos(k * u) / k
        B = np.sin(k * a) * np.cos(k * v) / k
        vals[j] = 4.0 / np.pi ** 2 * DK ** 2 * (A @ Mx @ B)
    return CubicSpline(hs, vals)


def sigma_t_of(M, d):
    """etalement temporel du nuage (modele de PsiTable::sigma_t)"""
    s_kg = M["DL_kg"] * np.sqrt(d) / M["ve_kg"]
    s_ga = M["DL_ga"] * np.sqrt(M["g"]) / M["ve_ga"]
    return np.sqrt(s_kg ** 2 + s_ga ** 2)


def psi_exact(spl, M, sig_t, tau, xg=XG, wg=WG):
    g, ve = M["g"], M["ve_ga"]
    T = g / ve
    out = float(spl(0.0)) * ndtr((tau - T) / sig_t)
    lo = max(tau - T, -9.0 * sig_t)
    hi = min(tau, 9.0 * sig_t)
    if hi > lo:
        x = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
        w = 0.5 * (hi - lo) * wg
        h = np.clip(g - ve * (tau - x), 0.0, g)
        pdf = np.exp(-0.5 * (x / sig_t) ** 2) / (np.sqrt(2.0 * np.pi) * sig_t)
        out += float(np.sum(w * pdf * spl(h)))
    return out


def plateau_exact(M, u, v, d):
    sig = np.sqrt(M["DT_kg"] ** 2 * d + M["DT_ga"] ** 2 * M["g"])
    a = M["a"]
    fx = ndtr((a - u) / sig) - ndtr((-a - u) / sig)
    fy = ndtr((a - v) / sig) - ndtr((-a - v) / sig)
    return fx * fy


def ramp_exact(M, d, tau):
    """somme de Psi sur tout le plan = E[ clip(ve (tau - delta) / g, 0, 1) ], delta ~ N(0, sigma_t)"""
    sig = sigma_t_of(M, d)
    T = M["g"] / M["ve_ga"]

    def m(s):
        z = s / sig
        return s * ndtr(z) + sig * np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    return (m(tau) - m(tau - T)) / T


def analyse(case, R, tag=""):
    base = OUT + "l1_%s_R%d" % (case, R) + (("_" + tag) if tag else "")
    M = read_meta(base)
    groups = read_rows(base)
    rec = {"M": M, "curves": []}
    plateau_err = {"node": 0.0, "rand": 0.0}
    err = {"node_node": 0.0, "node_mid": 0.0, "rand_node": 0.0, "rand_mid": 0.0, "sum_node": 0.0, "sum_mid": 0.0}
    worst = {k: None for k in err}
    float_diff = 0.0

    for key in sorted(groups):
        kind, u, v, d = key
        g = groups[key]
        tau = g["tau"]
        if kind in ("node", "rand"):
            spl = lambda_spline(M, u, v, d)
            sig_t = sigma_t_of(M, d)
            ex = np.array([psi_exact(spl, M, sig_t, t) for t in tau])
            # controle de la reference : 2e quadrature plus fine sur quelques instants
            if kind == "node" and u == 0.0 and v == 0.0:
                x2, w2 = leggauss(256)
                chk = max(abs(psi_exact(spl, M, sig_t, t, x2, w2) - e) for t, e in zip(tau[::5], ex[::5]))
                rec["quad_check"] = max(rec.get("quad_check", 0.0), chk)
        else:  # sum
            ex = np.array([ramp_exact(M, d, t) for t in tau])
        e = g["psi"] - ex
        g["exact"] = ex
        g["err"] = e
        for sub in ("node", "mid"):
            sel = g["sub"] == sub
            k2 = kind + "_" + sub
            m_ = np.max(np.abs(e[sel]))
            if m_ > err[k2]:
                err[k2] = m_
                worst[k2] = (key, tau[sel][np.argmax(np.abs(e[sel]))])
        if kind in ("node", "rand"):
            pe = abs(g["psi"][-1] - plateau_exact(M, u, v, d))     # dernier noeud en tau = plateau
            plateau_err[kind] = max(plateau_err[kind], pe)
            sel = g["sub"] == "node"
            float_diff = max(float_diff, np.max(np.abs(g["psig"][sel] - g["psi"][sel])))
        rec["curves"].append((key, g))
    # --- charge par bin de delta_t (ce que le code ecrit reellement) : Psi(t + delta_t) - Psi(t),
    #     pour deux phases des bords de bin (aux noeuds, et decales d'un demi pas de table)
    Rr = int(M["R"])
    peaks = []
    for key, g in rec["curves"]:
        if key[0] not in ("node", "rand"):
            continue
        for sub in ("node", "mid"):
            sel = g["sub"] == sub
            ex = g["exact"][sel]
            if len(ex) > Rr:
                peaks.append(np.max(np.abs(ex[Rr:] - ex[:-Rr])))
    amax = max(peaks)
    bin_err = {"node": 0.0, "rand": 0.0}
    bin_worst = {"node": None, "rand": None}
    for key, g in rec["curves"]:
        if key[0] not in ("node", "rand"):
            continue
        for sub in ("node", "mid"):
            sel = g["sub"] == sub
            ex, sm = g["exact"][sel], g["psi"][sel]
            dex, dsm = ex[Rr:] - ex[:-Rr], sm[Rr:] - sm[:-Rr]
            pk = np.max(np.abs(dex))
            e_ = np.max(np.abs(dsm - dex)) / amax     # erreur sur la charge d'un bin, / plus grand pic de bin du jeu de test
            if e_ > bin_err[key[0]]:
                bin_err[key[0]] = e_
                bin_worst[key[0]] = (key, sub, pk)
    rec.update(err=err, worst=worst, plateau_err=plateau_err, float_diff=float_diff,
               bin_err=bin_err, bin_worst=bin_worst, bin_peak=amax)
    return rec


def figure(case, R, rec, tag=""):
    M = rec["M"]
    curves = rec["curves"]
    fig, ax = plt.subplots(2, 3, figsize=(17, 8.5), gridspec_kw={"height_ratios": [2, 1]})

    # --- courbes choisies : pire "node", pire "rand", et deux exemples simples
    def worst_of(kind):
        best = None
        for key, g in curves:
            if key[0] == kind:
                m_ = np.max(np.abs(g["err"]))
                if best is None or m_ > best[0]:
                    best = (m_, key, g)
        return best
    sel = []
    dk = sorted(set(k[3] for k, g in curves if k[0] == "node"))
    for key, g in curves:
        if key[0] == "node" and key[1] == 0.0 and key[2] == 0.0 and key[3] in (dk[-1], dk[len(dk) // 2]):
            sel.append(("centre, d=%.3g cm (noeud)" % key[3], key, g))
    wn, wr = worst_of("node"), worst_of("rand")
    sel.append(("pire noeud : u=%.3g v=%.3g d=%.3g" % wn[1][1:], wn[1], wn[2]))
    sel.append(("pire hasard : u=%.3g v=%.3g d=%.3g" % wr[1][1:], wr[1], wr[2]))
    cols = ["tab:blue", "tab:green", "tab:red", "tab:purple"]
    for (lab, key, g), c in zip(sel, cols):
        o = np.argsort(g["tau"])
        ax[0, 0].plot(g["tau"][o], g["exact"][o], "-", color=c, lw=1.2, label="exact " + lab)
        isn = g["sub"] == "node"
        ax[0, 0].plot(g["tau"][isn], g["psi"][isn], "o", color=c, mfc="none", ms=5)
        ax[0, 0].plot(g["tau"][~isn], g["psi"][~isn], "x", color=c, ms=4)
        ax[1, 0].plot(g["tau"][o], g["err"][o], ".-", color=c, ms=3, lw=0.8)
    ax[0, 0].set_title("Psi(tau) : traits = exact, o = simu aux noeuds en tau, x = simu a mi-chemin")
    ax[0, 0].set_ylabel("Psi")
    ax[0, 0].legend(fontsize=6)
    ax[1, 0].set_xlabel("tau (ns)")
    ax[1, 0].set_ylabel("simu - exact")

    # --- nuage simu vs exact, tous points node/rand
    for kind, c, mk in (("node", "tab:blue", "o"), ("rand", "tab:orange", "s")):
        ex = np.concatenate([g["exact"] for k, g in curves if k[0] == kind])
        sm = np.concatenate([g["psi"] for k, g in curves if k[0] == kind])
        ax[0, 1].plot(ex, sm, mk, color=c, ms=2.5, alpha=0.5, label="%s (%d valeurs)" % (
            "points sur les noeuds" if kind == "node" else "points au hasard", ex.size))
        ax[1, 1].plot(ex, sm - ex, mk, color=c, ms=2.5, alpha=0.5)
    ax[0, 1].plot([0, 1], [0, 1], "k-", lw=0.8)
    ax[0, 1].set_xlabel("Psi exact")
    ax[0, 1].set_ylabel("Psi simu")
    ax[0, 1].set_title("Psi simu versus exact (toutes valeurs)")
    ax[0, 1].legend(fontsize=7)
    ax[1, 1].set_xlabel("Psi exact")

    # --- regle de somme
    shown = 0
    cmap = plt.get_cmap("viridis")
    sums = [(k, g) for k, g in curves if k[0] == "sum"]
    ds = sorted(set(k[3] for k, g in sums))
    for key, g in sums:
        c = cmap(ds.index(key[3]) / max(1, len(ds) - 1))
        o = np.argsort(g["tau"])
        if key[1] == 0.0 and key[2] == 0.0:
            ax[0, 2].plot(g["tau"][o], g["exact"][o], "-", color=c, label="exact d=%.3g cm" % key[3])
            isn = g["sub"] == "node"
            ax[0, 2].plot(g["tau"][isn], g["psi"][isn], "o", color=c, mfc="none", ms=4)
            ax[0, 2].plot(g["tau"][~isn], g["psi"][~isn], "x", color=c, ms=3)
        ax[1, 2].plot(g["tau"][o], g["err"][o], ".", color=c, ms=3)
    ax[0, 2].set_title("Somme de Psi sur tout le pavage (= rampe lissee)  (o x = simu)")
    ax[0, 2].legend(fontsize=7)
    ax[1, 2].set_xlabel("tau (ns)")

    for a_ in ax[1, :]:
        a_.axhline(TOL, color="gray", ls="--", lw=0.8)
        a_.axhline(-TOL, color="gray", ls="--", lw=0.8)
        a_.axhline(0, color="k", lw=0.5)
    fig.suptitle("Niveau 1, cas %s, R = %d (dtau = %.3g ns, n_tau = %d, pas u,v = %.3g mm)   traits gris : tolerance +-1e-3" % (
        case, R, M["dtau"], int(M["n_tau"]), 10 * M["step_out"]))
    fig.tight_layout()
    fn = OUT + "l1_%s_R%d" % (case, R) + (("_" + tag) if tag else "") + ".png"
    fig.savefig(fn, dpi=100)
    plt.close(fig)
    return fn


def main():
    args = sys.argv[1:] or ["A:2"]
    summary = []
    nfail = 0
    for arg in args:
        parts = arg.split(":")
        case, R = parts[0], int(parts[1])
        tag = parts[2] if len(parts) > 2 else ""
        rec = analyse(case, R, tag)
        fn = figure(case, R, rec, tag)
        M = rec["M"]
        print("\n=== cas %s, R = %d %s: dtau = %.3g ns, n_tau = %d, n_h = %d, pas u,v = %.3g mm, m_cells = %d, phi_edge = %.1e ===" % (
            case, R, ("[" + tag + "] " if tag else ""), M["dtau"], int(M["n_tau"]), int(M["n_h"]), 10 * M["step_out"], int(M["m_cells"]), M["phi_edge_max"]))
        print("validite de la reference : 2e quadrature (256 pts) ecart %.1e" % rec.get("quad_check", float("nan")))
        tests = [
            ("noeuds (u,v,d) aux noeuds en tau : teste build_psi seul", rec["err"]["node_node"], rec["worst"]["node_node"]),
            ("noeuds (u,v,d), tau a mi-chemin : + interpolation en tau", rec["err"]["node_mid"], rec["worst"]["node_mid"]),
            ("points au hasard, aux noeuds en tau : + interpolation en u,v,d", rec["err"]["rand_node"], rec["worst"]["rand_node"]),
            ("points au hasard, tau a mi-chemin : tout", rec["err"]["rand_mid"], rec["worst"]["rand_mid"]),
            ("somme sur le pavage vs rampe exacte, aux noeuds", rec["err"]["sum_node"], rec["worst"]["sum_node"]),
            ("somme sur le pavage vs rampe exacte, a mi-chemin", rec["err"]["sum_mid"], rec["worst"]["sum_mid"]),
            ("plateau (dernier noeud) vs formule en erf, points noeuds", rec["plateau_err"]["node"], None),
            ("plateau (dernier noeud) vs formule en erf, points hasard", rec["plateau_err"]["rand"], None),
            ("chemin float (psi_vector_g) vs double (psi_vector)", rec["float_diff"], None),
            ("charge d'un bin / pic max, points sur les noeuds (2 phases)", rec["bin_err"]["node"], None),
            ("charge d'un bin / pic max, points au hasard (2 phases)", rec["bin_err"]["rand"], None),
        ]
        for name, e, w in tests:
            tol_ = 1.0e-2 if name.startswith("charge d'un bin") else TOL     # metrique relative au pic de bin : indicative, seuil 1e-2
            ok = e < tol_
            nfail += (not ok)
            extra = ""
            if w is not None and not ok:
                k, t = w
                extra = "   pire : %s u=%.4g v=%.4g d=%.4g tau=%.4g" % (k[0], k[1], k[2], k[3], t)
            print("  %-64s %9.2e  %s%s" % (name, e, "OK" if ok else "ECHEC", extra))
            summary.append((case, R, name, e, ok))
        print("  (charge par bin : plus grand pic de bin = %.3f de la charge du nuage)" % rec["bin_peak"])
        print("  figure : " + fn)
    print("\nBILAN NIVEAU 1 : %s" % ("tout est OK" if nfail == 0 else "%d echec(s) sur %d tests" % (nfail, len(summary))))
    sys.exit(0 if nfail == 0 else 2)


if __name__ == "__main__":
    main()
