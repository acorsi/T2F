"""
viz_summary.py : resume par groupe des erreurs de geometrie (depart et apres affinage, par rapport a la verite) pour les evenements
traites par view_list.py. Ne calcule rien de nouveau : lit DOSSIER/work/e<indice>_geom.csv (ecrits par refine_geom.py).
Usage : python3 viz_summary.py liste.txt DOSSIER
Colonnes lues : event, n_iter, chi2d_start, chi2d_ref, ang0_start, ang1_start, ang0_ref, ang1_ref (degres), dx_start, dz_start, dx_ref, dz_ref (cm).
Sorties : tableau a l'ecran (mediane / 90e centile / max par groupe) et DOSSIER/summary.csv (une ligne par evenement).
"""
import os
import sys
import numpy as np

if len(sys.argv) < 3:
    sys.exit(__doc__)
lst, d = sys.argv[1], sys.argv[2]
rows = []
with open(lst) as f:
    for line in f:
        if line.strip() and not line.startswith("#"):
            t = line.split()
            rows.append((int(t[0]), t[1], int(t[2]), float(t[3]), float(t[4])))
COLS = ["event", "n_iter", "chi2d_start", "chi2d_ref", "ang0_start", "ang1_start", "ang0_ref", "ang1_ref",
        "dx_start", "dz_start", "dx_ref", "dz_ref"]
recs, missing, mism = [], [], 0
for i, gr, nit, c0, a0 in rows:
    fn = os.path.join(d, "work", "e%d_geom.csv" % i)
    if not os.path.isfile(fn):
        missing.append(i); continue
    with open(fn) as h:
        head = h.readline().strip().split(",")
    if head != COLS:
        sys.exit("colonnes inattendues dans %s : %s" % (fn, head))
    G = np.loadtxt(fn, delimiter=",", skiprows=1, ndmin=2)
    if G.shape[0] == 0:
        missing.append(i); continue
    g = G[0]
    if int(g[1]) != nit:
        mism += 1
    recs.append((gr, i, nit, a0, g))
print("evenements de la liste : %d ; avec fichier geom : %d ; sans : %d" % (len(rows), len(recs), len(missing)))
if missing:
    print("  sans resultat (ecartes ou en echec) : %s" % ", ".join(str(m) for m in missing[:20]) + (" ..." if len(missing) > 20 else ""))
print("n_iter de refine_geom different de celui de la production : %d evenement(s)" % mism)

def q(x):
    x = np.asarray(x, dtype=float)
    return "%8.3g %8.3g %8.3g" % (np.median(x), np.percentile(x, 90), np.max(x))

groups = []
for r in recs:
    if r[0] not in groups:
        groups.append(r[0])
def col(gr, name):
    k = COLS.index(name)
    return np.array([r[4][k] for r in recs if r[0] == gr])
print("\n(chaque cellule : mediane, 90e centile, maximum)")
items = [("n_iter", lambda gr: col(gr, "n_iter")),
         ("chi2d_start", lambda gr: col(gr, "chi2d_start")),
         ("chi2d_ref", lambda gr: col(gr, "chi2d_ref")),
         ("angle depart (deg), 2 fragments", lambda gr: np.concatenate([col(gr, "ang0_start"), col(gr, "ang1_start")])),
         ("angle affine (deg), 2 fragments", lambda gr: np.concatenate([col(gr, "ang0_ref"), col(gr, "ang1_ref")])),
         ("|dx| depart (cm)", lambda gr: np.abs(col(gr, "dx_start"))),
         ("|dx| affine (cm)", lambda gr: np.abs(col(gr, "dx_ref"))),
         ("|dz| depart (cm)", lambda gr: np.abs(col(gr, "dz_start"))),
         ("|dz| affine (cm)", lambda gr: np.abs(col(gr, "dz_ref")))]
print("%-34s" % "" + "".join("%-30s" % ("%s (n=%d)" % (g, len(col(g, "n_iter")))) for g in groups))
print("%-34s" % "" + "".join("%-30s" % "   mediane     p90      max" for g in groups))
for name, fn in items:
    print("%-34s" % name + "".join("%-30s" % q(fn(g)) for g in groups))

with open(os.path.join(d, "summary.csv"), "w") as g:
    g.write("group,a0_deg," + ",".join(COLS) + "\n")
    for gr, i, nit, a0, row in recs:
        g.write("%s,%.2f," % (gr, a0) + ",".join("%.6g" % v for v in row) + "\n")
print("\ntable par evenement : %s" % os.path.join(d, "summary.csv"))
