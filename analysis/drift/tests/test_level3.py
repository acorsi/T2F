#!/usr/bin/env python3
# Niveau 3 : exporte les voxels fusionnes (12 pas, comme main.cpp : merge_trace) d'un fichier d'entree
# vers out/l3_real.txt, lu par ./test_level3.
#   usage : python3 test_level3.py export [fichier.h5] [n_merge]
#   sans fichier.h5, on prend FICHIER_PAR_DEFAUT ci-dessous.
import sys
import h5py
import numpy as np

FICHIER_PAR_DEFAUT = "../input_1000.h5"      # <-- a modifier : chemin du fichier h5 utilise par defaut

if len(sys.argv) < 2 or sys.argv[1] != "export":
    print("usage : python3 test_level3.py export [fichier.h5] [n_merge]")
    sys.exit(1)
fichier = sys.argv[2] if len(sys.argv) > 2 else FICHIER_PAR_DEFAUT
nm = int(sys.argv[3]) if len(sys.argv) > 3 else 12
f = h5py.File(fichier, "r")
n_ev = f["/H/KE"].shape[0]
rows = []
for g in ("H", "L"):
    for i in range(n_ev):
        r = np.asarray(f["/%s/raw_trace" % g][i], dtype=float)
        k = (len(r) + nm - 1) // nm
        r = np.concatenate([r, np.zeros(k * nm - len(r))])
        rows.append(r.reshape(k, nm).sum(axis=1))
with open("out/l3_real.txt", "w") as fh:
    fh.write("%d\n" % len(rows))
    for r in rows:
        fh.write("%d %s\n" % (len(r), " ".join("%.17g" % x for x in r)))
print(fichier, "->  out/l3_real.txt :", len(rows), "traces,", sum(len(r) for r in rows), "voxels")
