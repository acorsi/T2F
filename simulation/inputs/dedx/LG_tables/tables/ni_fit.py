#!/usr/bin/env python3
"""
Ajustement NI = a * (Sn_int)^b a partir des tables dedx (KE/A, Se, Sn) du code T2F.

- Total_Ioniz.dat : blocs "#<A><Symbole>" puis lignes "E0(MeV)  U(MeV)  D(MeV)".
    D = deficit TRIM (= NI), U = transfert nucleaire TRIM (pour comparaison).
- dedx_<Z>.lg     : colonnes "KE/A(MeV/u)  Se  Sn" (meme unite pour Se et Sn).

Sn_int(E0) = A * g_Z(E0/A),  g_Z(e) = int_0^e Sn/(Se+Sn) de'
(les unites de Se et Sn n'interviennent pas, seul le rapport compte).

Usage : python3 ni_fit.py [--dir .] [--data Total_Ioniz.dat] [--log] [--out ni_universel.png]
"""
import argparse, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Z_OF = {"Zr": 40, "Sn": 50, "Nd": 60}


def read_dedx(path):
    d = np.loadtxt(path)
    e, se, sn = d[:, 0], d[:, 1], d[:, 2]
    order = np.argsort(e)
    return e[order], se[order], sn[order]


def make_g(e, se, sn):
    """Retourne g(e') = int_0^e' Sn/Stot de'' (trapezes, rapport constant sous le 1er point)."""
    r = sn / (se + sn)
    c = np.empty_like(e)
    c[0] = r[0] * e[0]
    c[1:] = c[0] + np.cumsum(0.5 * (r[1:] + r[:-1]) * np.diff(e))
    return lambda x: np.interp(x, np.concatenate(([0.0], e)), np.concatenate(([0.0], c)))


def read_data(path):
    blocks, cur = {}, None
    for line in open(path):
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            m = re.match(r"#\s*(\d+)\s*([A-Za-z]{1,2})\s*$", s)
            if m:
                cur = (int(m.group(1)), m.group(2).capitalize())
                blocks[cur] = []
            continue
        if cur is not None:
            blocks[cur].append([float(x) for x in s.split()[:3]])
    return {k: np.array(v) for k, v in blocks.items()}


def fit_power(x, y, log=False):
    """NI = a x^b, c=0. Absolu (moindres carres) par defaut, ou en log si --log."""
    b0, la0 = np.polyfit(np.log(x), np.log(y), 1)   # point de depart = fit en log
    if log:
        return np.exp(la0), b0
    from scipy.optimize import curve_fit
    p, _ = curve_fit(lambda s, a, b: a * s ** b, x, y, p0=[np.exp(la0), b0])
    return p[0], p[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--data", default="Total_Ioniz.dat")
    ap.add_argument("--log", action="store_true", help="ajuster en echelle log (erreur relative)")
    ap.add_argument("--out", default="ni_universel.png")
    args = ap.parse_args()

    data = read_data(os.path.join(args.dir, args.data))
    sets = {}
    for (A, sym), arr in data.items():
        Z = Z_OF.get(sym)
        f = os.path.join(args.dir, f"dedx_{Z}.lg")
        if Z is None or not os.path.exists(f):
            print(f"[!] {A}{sym}: table {f} introuvable, noyau ignore", file=sys.stderr)
            continue
        e, se, sn = read_dedx(f)
        g = make_g(e, se, sn)
        E0, U, D = arr.T
        if (E0 / A).max() > e.max():
            print(f"[!] {A}{sym}: E0/A max ({(E0/A).max():.3f}) > table ({e.max():.3f}), plafonne", file=sys.stderr)
        sets[f"{A}{sym}"] = dict(A=A, E0=E0, U=U, D=D, S=A * g(E0 / A))

    if not sets:
        sys.exit("Aucun noyau exploitable.")

    S = np.concatenate([v["S"] for v in sets.values()])
    D = np.concatenate([v["D"] for v in sets.values()])
    a, b = fit_power(S, D, args.log)
    res = D - a * S ** b
    print(f"\nAjustement ({'log' if args.log else 'absolu'}) : NI = a * S^b   a = {a:.4f}   b = {b:.4f}")
    print(f"Residus : rms = {np.sqrt(np.mean(res**2))*1e3:.1f} keV, max |res| = {np.abs(res).max()*1e3:.1f} keV")
    for k, v in sets.items():
        r = v["D"] - a * v["S"] ** b
        print(f"  {k:6s} rms = {np.sqrt(np.mean(r**2))*1e3:6.1f} keV   "
              f"S_int(E0 max) table = {v['S'][-1]:.3f}   U(TRIM) = {v['U'][-1]:.3f}   "
              f"ecart S/U = {100*(v['S'][-1]/v['U'][-1]-1):+.1f} %")

    # ---- figure
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    mk = dict(zip(sets, ["s", "o", "D"]))
    for k, v in sets.items():
        ax[0].plot(v["E0"], v["D"], mk[k], mfc="none", label=k)
        ax[1].plot(v["S"], v["D"], mk[k], mfc="none", label=k)
        ax[2].plot(v["S"], 1e3 * (v["D"] - a * v["S"] ** b), mk[k], mfc="none", label=k)
    xs = np.linspace(0, S.max() * 1.03, 200)
    ax[1].plot(xs, a * xs ** b, "k-", lw=1, label=f"{a:.3f}·S$^{{{b:.3f}}}$")
    ax[0].set_xscale("log"); ax[0].set_xlabel("E$_0$ (MeV)"); ax[0].set_ylabel("NI (MeV)")
    ax[0].set_title("Une courbe par noyau")
    ax[1].set_xlabel("S$_{Nucl}^{Int}$ (MeV), tables T2F"); ax[1].set_ylabel("NI (MeV)")
    ax[1].set_title("Loi universelle ?")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_xlabel("S$_{Nucl}^{Int}$ (MeV)"); ax[2].set_ylabel("NI − fit (keV)")
    ax[2].set_title("Residus")
    for x in ax:
        x.grid(alpha=0.3); x.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print("Figure :", args.out)


if __name__ == "__main__":
    main()
