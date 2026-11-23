"""
select_events.py : liste d'evenements a visualiser, tires dans all_prod.h5 (sortie du mode production), en 4 groupes :
  az_top   : au maximum d'iterations, azimut a0 dans les bandes données par --az (par defaut les deux classes surrepresentees)
  az_ctrl  : converges en <= 4 iterations, meme azimut (temoins : l'azimut seul fait-il la difference ?)
  oth_top  : au maximum d'iterations, azimut hors des bandes
  oth_ctrl : converges en <= 4 iterations, azimut hors des bandes
Usage : python3 select_events.py all_prod.h5 liste.txt [--az "59.6-75.1,104.9-120.3"] [--n 40,20,30,10] [--seed 1] [--max_iter 8]
  --az : bandes de a0 en degres, modulo 180 (a0 en radians dans le fichier : 1.04-1.31 rad = 59.6-75.1 deg ; 1.83-2.10 rad = 104.9-120.3 deg)
  --n  : nombre d'evenements par groupe, dans l'ordre az_top, az_ctrl, oth_top, oth_ctrl
Sortie : une ligne par evenement : indice groupe n_iter chi2_start a0_deg   (lignes commencant par # = commentaires)
Le tirage est aleatoire a graine fixe : meme fichier + memes options = meme liste.
"""
import sys
import h5py
import numpy as np

args = sys.argv[1:]
def opt(name, default):
    if name in args:
        i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return v
    return default
az = opt("--az", "59.6-75.1,104.9-120.3")
nn = [int(t) for t in opt("--n", "40,20,30,10").split(",")]
seed = int(opt("--seed", "1"))
mxa = opt("--max_iter", None)
if len(args) < 2 or len(nn) != 4:
    sys.exit(__doc__)
f = h5py.File(args[0], "r")
it = f["n_iter"][()]; idx = f["event_index"][()]; p = f["params"][()]; c0 = f["chi2_start"][()]
mx = int(mxa) if mxa is not None else int(it.max())
a0 = np.degrees(p[:, 2]) % 180.0
bands = [tuple(float(t) for t in b.split("-")) for b in az.split(",")]
inb = np.zeros(len(it), dtype=bool)
for lo, hi in bands:
    inb |= (a0 >= lo) & (a0 < hi)
top = it >= mx
low = it <= 4
groups = [("az_top", top & inb), ("az_ctrl", low & inb), ("oth_top", top & ~inb), ("oth_ctrl", low & ~inb)]
rng = np.random.RandomState(seed)
with open(args[1], "w") as g:
    g.write("# fichier %s ; max_iter = %d ; bandes a0 (deg) : %s ; graine %d\n" % (args[0], mx, az, seed))
    g.write("# indice groupe n_iter chi2_start a0_deg\n")
    for (name, sel), k in zip(groups, nn):
        cand = np.where(sel)[0]
        take = np.sort(rng.choice(cand, size=min(k, len(cand)), replace=False)) if len(cand) else cand
        print("%-9s : %6d evenements disponibles, %d tires" % (name, len(cand), len(take)))
        for j in take:
            g.write("%d %s %d %.4f %.2f\n" % (idx[j], name, it[j], c0[j], a0[j]))
print("liste ecrite : %s" % args[1])
