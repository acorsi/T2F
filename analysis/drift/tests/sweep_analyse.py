#!/usr/bin/env python3
# Analyse du balayage : pour chaque configuration, plus petit m_cells dont l'erreur sur le profil temporel
# est a moins de MARGE du meilleur resultat obtenu ; puis test de regles m_cells = max(8, ceil(c a / sigma_gap)).
import csv, sys, math
from collections import defaultdict
R = int(sys.argv[1]) if len(sys.argv) > 1 else 8
MARGE = 2.0e-4
rows = list(csv.DictReader(open("out/sweep_R%d.csv" % R)))
conf = defaultdict(list)
for r in rows:
    if r["shape_max"] in ("", "None"):
        continue
    conf[int(r["conf"])].append(dict(m=int(r["m_cells"]), a=float(r["a"]), sg=float(r["sigma_gap"]), g=float(r["g"]),
                                     shape=float(r["shape_max"]), plat=0.0, mb=0.0))
print("conf  g   pad   a/sigma_gap | erreur profil temporel pour m_cells = 8 / 12 / 16 / 24 / 32  | m_min (marge %.0e)" % MARGE)
need = {}
for k in sorted(conf):
    L = sorted(conf[k], key=lambda x: x["m"])
    best = min(x["shape"] for x in L)
    mmin = next(x["m"] for x in L if x["shape"] <= best + MARGE)
    need[k] = (L[0]["a"], L[0]["sg"], mmin)
    print("%3d  %.2f %.2f  %6.1f      | %s | %d" % (k, L[0]["g"], 2 * L[0]["a"] - 0.015, L[0]["a"] / L[0]["sg"],
                                              "  ".join("%.1e" % x["shape"] for x in L), mmin))
print()
for c in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
    bad = []
    for k, (a, sg, mmin) in sorted(need.items()):
        m = max(8, math.ceil(c * a / sg))
        if m < mmin:
            bad.append(k)
    print("regle m_cells = max(8, ceil(%.1f a / sigma_gap)) : insuffisante pour %d configurations %s" % (c, len(bad), bad))
