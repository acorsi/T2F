#!/usr/bin/env python3
"""
Construit, pour chaque fichier dedx_<Z>.lg (colonnes : KE/A [MeV/u], Se, Sn, meme unite pour Se et Sn),
la table  KE/A  g_Z  h_Z  avec
    g_Z(e) = int_0^e Sn/(Se+Sn) de'     (MeV/u)
    h_Z(e) = int_0^e Se/(Se+Sn) de'     (MeV/u)
Integration par trapezes ; le rapport est suppose constant entre 0 et le premier point de la table.
Une ligne e=0, g=h=0 est ajoutee. Controle : g + h = e (a la precision machine).

Usage : python3 make_gh_tables.py [--indir .] [--outdir gh_tables] [--pattern "dedx_*.lg"]
Sortie : gh_<Z>.dat (3 colonnes) + resume a l'ecran.
Dans le code : S_n^rest = A * g_Z(E/A) et S_e^rest = A * h_Z(E/A)  (MeV), interpolation lineaire en e.
"""
import argparse, glob, os, re, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--indir", default="."); ap.add_argument("--outdir", default="gh_tables")
ap.add_argument("--pattern", default="dedx_*.lg")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)

def cumtrapz0(e, r):
    """int_0^e r de', avec r constant sous le premier point."""
    c = np.empty_like(e)
    c[0] = r[0] * e[0]
    c[1:] = c[0] + np.cumsum(0.5 * (r[1:] + r[:-1]) * np.diff(e))
    return c

print(f"{'Z':>3s} {'npts':>5s} {'e_min':>10s} {'e_max':>8s} {'g(e_max)':>9s} {'h(e_max)':>9s} {'|g+h-e|max':>11s} {'E_max(A=2.5Z)':>14s}")
nok = 0
for f in sorted(glob.glob(os.path.join(args.indir, args.pattern))):
    m = re.search(r"dedx_(\d+)", os.path.basename(f))
    if not m:
        print(f"[!] nom inattendu, ignore : {f}", file=sys.stderr); continue
    Z = int(m.group(1))
    try:
        d = np.loadtxt(f, comments="#", ndmin=2)
    except Exception as ex:
        print(f"[!] Z={Z}: lecture impossible ({ex})", file=sys.stderr); continue
    if d.shape[1] < 3:
        print(f"[!] Z={Z}: moins de 3 colonnes", file=sys.stderr); continue
    e, se, sn = d[:, 0], d[:, 1], d[:, 2]
    o = np.argsort(e); e, se, sn = e[o], se[o], sn[o]
    keep = np.concatenate(([True], np.diff(e) > 0))          # supprime les energies dupliquees
    if not keep.all():
        print(f"[!] Z={Z}: {np.sum(~keep)} energie(s) dupliquee(s) supprimee(s)", file=sys.stderr)
    e, se, sn = e[keep], se[keep], sn[keep]
    if np.any(se < 0) or np.any(sn < 0) or np.any(se + sn <= 0):
        print(f"[!] Z={Z}: valeurs negatives ou Stot nul, ignore", file=sys.stderr); continue
    st = se + sn
    g = cumtrapz0(e, sn / st); h = cumtrapz0(e, se / st)
    E = np.concatenate(([0.0], e)); G = np.concatenate(([0.0], g)); H = np.concatenate(([0.0], h))
    err = np.abs(G + H - E).max()
    out = os.path.join(args.outdir, f"phd_{Z}.dat")
    np.savetxt(out, np.column_stack([E, G]), fmt="%.8e",
               header=f"Z={Z}   KE/A(MeV/u)   g_Z=int Sn/Stot (MeV/u)")
    Emax = e[-1] * 2.5 * Z          # energie max couverte pour un fragment de masse ~2.5 Z
    flag = "  <-- table courte (< 120 MeV)" if Emax < 120 else ""
    print(f"{Z:3d} {len(e):5d} {e[0]:10.3e} {e[-1]:8.3f} {g[-1]:9.5f} {h[-1]:9.5f} {err:11.2e} {Emax:12.0f} MeV{flag}")
    nok += 1
print(f"\n{nok} table(s) ecrite(s) dans {args.outdir}/")
