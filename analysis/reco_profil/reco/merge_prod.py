"""
merge_prod.py : assemble les fichiers de tranche d'un run Slurm en un seul HDF5.
Usage : python3 merge_prod.py DOSSIER_SORTIE N_EVENEMENTS [SLICE=2000] [--out fichier.h5]
Refuse de fusionner si une tranche manque. Ecrit aussi <out sans .h5>_skipped.csv (par defaut all_prod_skipped.csv) (evenements ecartes).
"""
import os
import sys

import prod_io

args = [x for x in sys.argv[1:]]
out = None
if "--out" in args:
    i = args.index("--out")
    out = args[i + 1]
    del args[i:i + 2]
if len(args) < 2:
    sys.exit(__doc__)
d, n = args[0], int(args[1])
sl = int(args[2]) if len(args) > 2 else 2000
nt = (n + sl - 1) // sl
files, miss = [], []
for k in range(nt):
    f = os.path.join(d, "slices", "s_%05d_prod.h5" % k)
    (files if os.path.isfile(f) and os.path.getsize(f) > 0 else miss).append(f)
if miss:
    sys.exit("[merge] %d tranche(s) manquante(s) sur %d, par exemple %s ; relancer submit_prod.sh" % (len(miss), nt, miss[0]))
if out is None:
    out = os.path.join(d, "all_prod.h5")
ne = prod_io.merge_files(out, files)
sk = out[:-3] + "_skipped.csv" if out.endswith(".h5") else out + "_skipped.csv"
with open(sk, "w") as g:
    g.write("event,reason\n")
    for k in range(nt):
        f = os.path.join(d, "slices", "s_%05d_skipped.csv" % k)
        if os.path.isfile(f):
            with open(f) as h:
                h.readline()
                for line in h:
                    g.write(line)
print("[merge] %d tranches, %d evenements -> %s ; ecartes : %s" % (nt, ne, out, sk))
