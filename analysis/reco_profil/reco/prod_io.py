"""
prod_io.py : ecriture et fusion des fichiers de sortie HDF5 du mode production (prod_refine.py).
Ne depend que de numpy et h5py.

Contenu d'un fichier (n evenements ; fragment 0 = demi-plan y > y_plane, fragment 1 = demi-plan y < y_plane) :
  event_index (n)      indice de l'evenement dans le fichier d'entree
  event_id (n)         /event_id du fichier d'entree
  vertex (n,3)         x, y (= y_plane), z du vertex (cm)
  u0, u1 (n,3)         direction unitaire (ux, uy, uz) des fragments 0 et 1
  params (n,6)         x, z, a0, m0, a1, m1 (parametres ajustes ; a = azimut, m = uz / uxy)
  chi2_start (n)       chi2/ndl de la geometrie de depart (lissage de la phase geometrie)
  chi2_geom (n)        chi2/ndl de la geometrie affinee (meme lissage)
  chi2_prof (n)        chi2/ndl du profil final
  n_iter (n)           iterations de l'affinage de la geometrie
  L_base (n,2)         longueur (cm) de chaque fragment avant rallonge
  dL (n,2)             rallonge (cm)
  not_contained (n,2)  1 = fin de profil non contenue apres rallonge
  tail_ratio (n,2)     moyenne des derniers noeuds / maximum du profil
  tail_sigma (n,2)     ecart-type de cette moyenne / maximum
  n_nodes (n,2)        nombre de noeuds de chaque fragment
  prof_off (n+1)       prof_val[prof_off[e] : prof_off[e+1]] = noeuds du fragment 0 puis du fragment 1 de l'evenement e
  prof_val, prof_sig   dE/dx aux noeuds (e-/noeud, comme le serveur) et ecart-type (non contraint)
Attributs : delta (cm), lam, lam_end_fac, lam_end_zone, y_plane, convention, version.
"""
import h5py
import numpy as np

PER_EVENT = ["event_index", "event_id", "vertex", "u0", "u1", "params", "chi2_start", "chi2_geom", "chi2_prof",
             "n_iter", "L_base", "dL", "not_contained", "tail_ratio", "tail_sigma", "n_nodes"]
SHAPES = {"event_index": (), "event_id": (), "vertex": (3,), "u0": (3,), "u1": (3,), "params": (6,), "chi2_start": (),
          "chi2_geom": (), "chi2_prof": (), "n_iter": (), "L_base": (2,), "dL": (2,), "not_contained": (2,),
          "tail_ratio": (2,), "tail_sigma": (2,), "n_nodes": (2,)}
DTYPES = {"event_index": "i8", "event_id": "i8", "n_iter": "i4", "not_contained": "i1", "n_nodes": "i4"}


def empty_arrays():
    d = {}
    for k in PER_EVENT:
        d[k] = np.zeros((0,) + SHAPES[k], dtype=DTYPES.get(k, "f8"))
    d["prof_off"] = np.zeros(1, dtype="i8")
    d["prof_val"] = np.zeros(0)
    d["prof_sig"] = np.zeros(0)
    return d


def write_file(path, d, attrs):
    with h5py.File(path, "w") as f:
        for k, v in d.items():
            f.create_dataset(k, data=v, compression="gzip", compression_opts=4) if v.size > 1000 else f.create_dataset(k, data=v)
        for k, v in attrs.items():
            f.attrs[k] = v


def merge_files(out, files):
    """concatene des fichiers de tranche (dans l'ordre donne) ; renvoie le nombre d'evenements"""
    parts = []
    attrs = None
    for fn in files:
        with h5py.File(fn, "r") as f:
            if attrs is None:
                attrs = dict((k, f.attrs[k]) for k in f.attrs)
            parts.append(dict((k, np.array(f[k])) for k in f.keys()))
    d = empty_arrays()
    if parts:
        for k in PER_EVENT + ["prof_val", "prof_sig"]:
            d[k] = np.concatenate([p[k] for p in parts])
        offs, base = [np.zeros(1, dtype="i8")], 0
        for p in parts:
            offs.append(p["prof_off"][1:] + base)
            base += int(p["prof_off"][-1])
        d["prof_off"] = np.concatenate(offs)
    write_file(out, d, attrs if attrs is not None else {})
    return len(d["event_index"])
