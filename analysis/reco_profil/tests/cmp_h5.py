"""
cmp_h5.py : compare deux sorties HDF5 du mode production (prod_refine.py), par exemple Mac contre cluster, sur les memes evenements.
Usage : python3 cmp_h5.py a_prod.h5 b_prod.h5
Decisions discretes (evenements, nombre d'iterations, nombre de noeuds, rallonges dL, fin non contenue) : doivent etre IDENTIQUES.
Valeurs continues : l'ecart est donne relativement au maximum de |valeur| du tableau ; un changement de compilateur ou de
bibliotheque d'algebre lineaire donne des ecarts de l'ordre de 1e-8 a 1e-5 (l'optimiseur amplifie les arrondis).
"""
import sys
import h5py
import numpy as np

fa, fb = h5py.File(sys.argv[1], "r"), h5py.File(sys.argv[2], "r")
DISCRETE = ["event_index", "event_id", "n_iter", "n_nodes", "not_contained", "prof_off"]
bad = 0
if sorted(fa.keys()) != sorted(fb.keys()):
    sys.exit("tableaux differents : %s / %s" % (sorted(fa.keys()), sorted(fb.keys())))
na, nb = len(fa["event_index"]), len(fb["event_index"])
print("evenements : %d (a) / %d (b)" % (na, nb))
if na != nb or not np.array_equal(fa["event_index"][()], fb["event_index"][()]):
    sys.exit("RESULTAT : ECART (listes d'evenements differentes ; evenements seulement dans a : %d, seulement dans b : %d)"
             % (len(set(fa["event_index"][()]) - set(fb["event_index"][()])), len(set(fb["event_index"][()]) - set(fa["event_index"][()]))))
for k in sorted(fa.keys()):
    a, b = fa[k][()], fb[k][()]
    if a.shape != b.shape:
        print("  %-14s FORMES DIFFERENTES %s %s" % (k, a.shape, b.shape)); bad += 1; continue
    if k in DISCRETE or k == "dL":
        n = int((a != b).sum())
        print("  %-14s %s" % (k, "identique" if n == 0 else "%d valeurs differentes" % n))
        bad += n > 0
    else:
        sc = max(float(np.abs(a).max()), 1e-300) if a.size else 1.0
        d = float(np.abs(a - b).max() / sc) if a.size else 0.0
        print("  %-14s ecart max / max|a| = %.2e" % (k, d))
        bad += d > 1e-4
print("RESULTAT :", "OK" if bad == 0 else "ECART")
