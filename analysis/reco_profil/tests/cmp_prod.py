"""
cmp_prod.py : compare la sortie du mode production (HDF5) aux CSV de refine_fast.py / refine_geom.py (meme lissage, une seule valeur de --lam).
Usage : python3 cmp_prod.py prod200_prod.h5 fast3     (fast3 = prefixe : fast3_geom.csv, fast3_prof.csv, fast3_flags.csv)
Compare, pour les evenements presents dans les deux : nombre d'iterations, chi2/ndl de la geometrie et du profil, rallonges,
fin non contenue, queue/max, sigma de queue. Attendu : identiques (ecarts relatifs <= 1e-9).
"""
import sys
import h5py
import numpy as np

h5, pre = sys.argv[1], sys.argv[2]
f = h5py.File(h5, "r")
ev = f["event_index"][()]
g = np.genfromtxt(pre + "_geom.csv", delimiter=",", names=True)
p = np.loadtxt(pre + "_prof.csv", delimiter=",", skiprows=1)     # l'en-tete contient des virgules : lecture par colonnes
fl = np.genfromtxt(pre + "_flags.csv", delimiter=",", names=True)
pa = p[p[:, 1] == 1]                         # colonnes : event, geom (1 = affinee), lam, chi2_dof, ...
lams = np.unique(pa[:, 2])
if len(lams) != 1:
    sys.exit("plusieurs lissages dans %s_prof.csv : relancer la reference avec une seule valeur de --lam" % pre)
ge = dict((int(e), k) for k, e in enumerate(g["event"]))
pe = dict((int(e), k) for k, e in enumerate(pa[:, 0]))
fe = dict(((int(e), int(k)), j) for j, (e, k) in enumerate(zip(fl["event"], fl["frag"])))
common = [k for k, e in enumerate(ev) if int(e) in ge]
print("evenements : %d dans le HDF5, %d dans la reference, %d communs" % (len(ev), len(g), len(common)))
worst = {}
def upd(name, a, b):
    d = float(abs(a - b) / max(abs(b), 1e-300)) if b != 0 else float(abs(a))
    worst[name] = max(worst.get(name, 0.0), d)
for k in common:
    e = int(ev[k]); j = ge[e]
    upd("n_iter", float(f["n_iter"][k]), float(g["n_iter"][j]))
    upd("chi2_geom", float(f["chi2_geom"][k]), float(g["chi2d_ref"][j]))
    upd("chi2_start", float(f["chi2_start"][k]), float(g["chi2d_start"][j]))
    upd("chi2_prof", float(f["chi2_prof"][k]), float(pa[pe[e], 3]))
    for fr in range(2):
        r = fl[fe[(e, fr)]]
        upd("dL", float(f["dL"][k][fr]), float(r["ext_cm"]))
        upd("not_contained", float(f["not_contained"][k][fr]), float(r["fin_non_contenue"]))
        upd("tail_ratio", float(f["tail_ratio"][k][fr]), float(r["queue_sur_max"]))
        upd("tail_sigma", float(f["tail_sigma"][k][fr]), float(r["sigma_queue_sur_max"]))
bad = 0
for k, v in sorted(worst.items()):
    print("  %-14s ecart relatif max %.2e" % (k, v))
    bad += v > 1e-9
print("RESULTAT :", "OK" if bad == 0 else "ECART")
