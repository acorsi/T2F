"""
whiten_fast.py : blanchiment exact et rapide, SANS modifier fit_profile.py ni refine_geom.py.

Principe : Whitener.apply fait, pour chaque pad, np.linalg.solve(L, A_bloc.T) et np.linalg.solve(L, y_bloc),
c'est-a-dire une factorisation LU de L a chaque appel. Ici on calcule UNE fois par (n, ser_scale) l'inverse
de la matrice triangulaire L (deja en cache dans Whitener.chol), puis on blanchit par un produit matriciel :
    A_bloc blanchi = A_bloc @ (L^-1)^T          y_bloc blanchi = y_bloc @ (L^-1)^T
Resultat identique a l'arrondi pres (aucune approximation, aucune troncature).

Usage :
    import fit_profile, whiten_fast
    whiten_fast.install(fit_profile)        # avant de lancer refine_geom.main()
Verification : variable d'environnement WHITEN_CHECK=N compare les N premiers appels a l'ancienne methode
et affiche l'ecart relatif maximal a la fin du programme (stderr).
"""
import atexit
import os
import sys

import numpy as np
from scipy.linalg import solve_triangular


def _inv_t(self, n, ser_scale):
    """(L^-1)^T pour la fenetre de n echantillons, en cache"""
    cache = self.__dict__.setdefault("_inv_t_cache", {})
    key = (n, round(ser_scale, 5))
    if key not in cache:
        L = self.chol(n, ser_scale)
        Li = solve_triangular(L, np.eye(n), lower=True)
        cache[key] = np.ascontiguousarray(Li.T)
    return cache[key]


def install(fp):
    if getattr(fp.Whitener, "_fast_installed", False):
        return
    old_apply = fp.Whitener.apply
    n_check = int(os.environ.get("WHITEN_CHECK", "0") or 0)
    stat = {"calls": 0, "dA": 0.0, "dy": 0.0}

    plans = {}

    def plan_of(self, pids, nts):
        """liste (debut, n, (L^-1)^T) par pad, en cache : les pads d'un evenement ne changent pas entre evaluations"""
        key = (id(self), np.asarray(pids).tobytes(), np.asarray(nts).tobytes())
        pl = plans.get(key)
        if pl is None:
            pl = []
            off = 0
            for pid, n in zip(pids, nts):
                n = int(n)
                pl.append((off, n, _inv_t(self, n, self.ser_scale(pid))))
                off += n
            if len(plans) > 4000:
                plans.clear()
            plans[key] = pl
        return pl

    def apply_fast(self, pids, nts, A, y):
        if not self.noise_on:
            return A.copy(), y.copy()
        Aw = np.empty_like(A)
        yw = np.empty_like(y)
        for off, n, M in plan_of(self, pids, nts):
            Aw[:, off:off + n] = A[:, off:off + n] @ M
            yw[off:off + n] = y[off:off + n] @ M
        if stat["calls"] < n_check:
            stat["calls"] += 1
            Aw0, yw0 = old_apply(self, pids, nts, A, y)
            stat["dA"] = max(stat["dA"], float(np.abs(Aw - Aw0).max() / max(np.abs(Aw0).max(), 1e-300)))
            stat["dy"] = max(stat["dy"], float(np.abs(yw - yw0).max() / max(np.abs(yw0).max(), 1e-300)))
        return Aw, yw

    fp.Whitener.apply = apply_fast
    fp.Whitener._fast_installed = True
    if n_check > 0:
        def report():
            sys.stderr.write("[whiten_fast] verification sur %d appels : ecart relatif max  Aw %.2e   yw %.2e\n"
                             % (stat["calls"], stat["dA"], stat["dy"]))
        atexit.register(report)
