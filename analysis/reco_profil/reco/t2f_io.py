#!/usr/bin/env python3
"""
t2f_io.py : lecture du format h5 de ./process pour les scripts d'analyse.

Ce que le fichier ne contient plus (retrouve ici) :
  - la position (x, y) du centre d'un pad : a lire dans /pads (id, x, y) a partir de pid, avec pad_geometry(f) ;
  - les points (x, y, z) de la trace de H ou L : reconstruits a partir du vertex, de theta, phi, du pas et
    de l'axe d'emission (voir l'attribut /metadata trace_xyz_description).

Usage :
    import h5py
    from t2f_io import pad_geometry, pad_xy, read_vertex, trace_xyz, event_pads

    f = h5py.File("sortie.h5", "r")
    geom = pad_geometry(f)                 # une fois : lu dans /pads (fichiers anciens : pad_geometry(f, "my_pad.txt"))
    vx, vy, vz = read_vertex(f, i)
    x, y, z = trace_xyz(f, "H", i)
    pads = event_pads(f, i, geom)          # dictionnaire de tableaux, avec x et y des pads

Necessite numpy et h5py.
"""

import numpy as np


def _attr(f, name):
    v = f["/metadata"].attrs[name]
    return v.decode() if isinstance(v, bytes) else v


def load_pad_file(path):
    """my_pad.txt (colonnes : id x y r_in r_out sx sy) -> dictionnaire pad_id -> (x, y, r_in, r_out, sx, sy)."""
    data = np.loadtxt(path)
    return dict((int(r[0]), tuple(float(u) for u in r[1:])) for r in data)


def load_pads_h5(f):
    """Groupe /pads du h5 -> meme dictionnaire que load_pad_file : pad_id -> (x, y, r_in, r_out, sx, sy)."""
    P = f["/pads"]
    ids = np.array(P["id"], dtype=int)
    cols = [np.array(P[k], dtype=float) for k in ("x", "y", "r_in", "r_out")] + \
           [np.array(P[k], dtype=int) for k in ("sx", "sy")]
    return dict((int(ids[k]), tuple(float(c[k]) for c in cols)) for k in range(len(ids)))


def load_pad_capacitances(f):
    """Modele de bruit "capacitance" : dictionnaire pad_id -> (C en pF, ENC en e-) lu dans /pads/c_pF et /pads/enc ;
    None si le fichier n'a pas ces champs (modele "uniform" ou ancien format : tous les pads ont alors l'ENC de /metadata)."""
    if "/pads/c_pF" not in f:
        return None
    P = f["/pads"]
    ids = np.array(P["id"], dtype=int)
    c = np.array(P["c_pF"], dtype=float)
    e = np.array(P["enc"], dtype=float)
    return dict((int(ids[k]), (float(c[k]), float(e[k]))) for k in range(len(ids)))


def pad_geometry(f, pad_file=None):
    """Geometrie des pads : /pads du h5 si present (fichiers recents), sinon pad_file (my_pad.txt, fichiers anciens)."""
    if "/pads" in f:
        return load_pads_h5(f)
    if pad_file is None:
        raise ValueError("ce fichier h5 n'a pas de groupe /pads (ancien format) : donner pad_file=my_pad.txt")
    return load_pad_file(pad_file)


def pad_xy(geom, pids):
    """Positions (x, y) des pads de numeros pids (tableau) ; erreur explicite si un pad est absent du fichier."""
    x = np.empty(len(pids)); y = np.empty(len(pids))
    for k, p in enumerate(pids):
        g = geom.get(int(p))
        if g is None:
            raise KeyError("pad %d absent de la geometrie des pads : mauvais fichier ?" % int(p))
        x[k], y[k] = g[0], g[1]
    return x, y


def read_vertex(f, i):
    """Vertex de fission (cm) de l'evenement i ; accepte aussi les anciens fichiers (/H/vx, ...)."""
    if "/vertex" in f:
        return float(f["/vertex/x"][i]), float(f["/vertex/y"][i]), float(f["/vertex/z"][i])
    return float(f["/H/vx"][i]), float(f["/H/vy"][i]), float(f["/H/vz"][i])


def trace_xyz(f, frag, i, axis=None):
    """Points (x, y, z) en cm de la trace du fragment 'H' ou 'L', evenement i (float64).
    axis : 'Y' ou 'Z' ; par defaut lu dans /metadata emission_axis."""
    if axis is None:
        axis = _attr(f, "emission_axis")
    step = float(f["/metadata"].attrs["step_size_eff"])
    th = float(f["/%s/theta" % frag][i]); ph = float(f["/%s/phi" % frag][i])
    n = len(f["/%s/dE" % frag][i])
    s = step * (np.arange(n) + 0.5)
    ux = np.sin(th) * np.cos(ph)
    uy = np.cos(th) if axis == "Y" else np.sin(th) * np.sin(ph)
    uz = np.cos(th) if axis == "Z" else np.sin(th) * np.sin(ph)
    vx, vy, vz = read_vertex(f, i)
    return vx + s * ux, s * uy, vz + s * uz


def event_pads(f, i, geom):
    """Liste de pads de l'evenement i : pid, x, y (du fichier de pads), pq, pflag, pk0, pnt, pqn (concatenes)."""
    P = f["/P"]
    pid = np.array(P["pid"][i], dtype=int)
    x, y = pad_xy(geom, pid)
    return {"pid": pid, "x": x, "y": y,
            "pq": np.array(P["pq"][i], dtype=float), "pflag": np.array(P["pflag"][i], dtype=int),
            "pk0": np.array(P["pk0"][i], dtype=int), "pnt": np.array(P["pnt"][i], dtype=int),
            "pqn": np.array(P["pqn"][i], dtype=float)}
