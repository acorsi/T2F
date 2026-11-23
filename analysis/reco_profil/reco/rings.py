#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rings.py : pads non lus utilises comme contraintes de signal nul (couronne 2).

Definitions (uniquement a partir des donnees : aucun drapeau de simulation) :
  - pads ecrits : ceux de /P/pid de l'evenement ;
  - fantome : pad ecrit sans AUCUN autre pad ecrit a moins de radius (1.5 x pitch) ; retire (une seule passe) ;
  - pads touches : pads ecrits non retires ;
  - couronne 1 : pads non touches dont le centre est a moins de radius du centre d'un pad touche ;
  - couronne 2 : pads non touches, hors couronne 1, dont le centre est a moins de radius du centre d'un pad
    de couronne 1.
Seule la couronne 2 sert de contrainte (signal nul) ; la couronne 1 peut porter une charge sous le seuil et reste libre.
"""

import numpy as np


class PadRings:
    def __init__(self, geom, radius):
        """geom : dictionnaire pad_id -> (x, y, ...) (t2f_io.pad_geometry) ; radius en cm"""
        ids = sorted(geom.keys())
        self.ids = np.array(ids, dtype=int)
        self.x = np.array([geom[k][0] for k in ids], dtype=float)
        self.y = np.array([geom[k][1] for k in ids], dtype=float)
        self.index = dict((int(k), n) for n, k in enumerate(self.ids))
        D = np.hypot(self.x[:, None] - self.x[None, :], self.y[:, None] - self.y[None, :])
        np.fill_diagonal(D, np.inf)
        self.near = D < radius

    def split(self, written_pids):
        """renvoie (keep, ring1, ring2) : keep = booleen sur written_pids (False : fantome retire) ;
        ring1, ring2 = tableaux de pad_id"""
        w = np.array([self.index[int(p)] for p in written_pids], dtype=int)
        n = len(self.ids)
        if len(w) == 0:
            return np.zeros(0, dtype=bool), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        keep = self.near[np.ix_(w, w)].any(axis=1)
        tmask = np.zeros(n, dtype=bool)
        tmask[w[keep]] = True
        if not tmask.any():
            return keep, np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        r1 = (~tmask) & self.near[:, tmask].any(axis=1)
        if r1.any():
            r2 = (~tmask) & (~r1) & self.near[:, r1].any(axis=1)
        else:
            r2 = np.zeros(n, dtype=bool)
        return keep, self.ids[r1], self.ids[r2]


def zero_window(pk0, pnt, uz, ext_max, ve, dt_can):
    """fenetre commune (j0, nt) des pads a signal nul, en echantillons CAN.
    pk0, pnt : fenetres des pads touches ; la fenetre va du plus petit j0 au plus grand j0+nt, elargie de chaque
    cote par le decalage temporel maximal d'une rallonge de ext_max cm : ext_max x max|uz| / ve / dt_can.
    j0 >= 0. Renvoie None si aucun pad touche n'a d'echantillon."""
    pk0 = np.asarray(pk0, dtype=int)
    pnt = np.asarray(pnt, dtype=int)
    ok = pnt > 0
    if not ok.any():
        return None
    jmin = int(pk0[ok].min())
    jmax = int((pk0[ok] + pnt[ok]).max())
    dj = int(np.ceil(ext_max * max(abs(float(u)) for u in uz) / ve / dt_can)) + 2
    j0 = max(0, jmin - dj)
    return j0, jmax + dj - j0
