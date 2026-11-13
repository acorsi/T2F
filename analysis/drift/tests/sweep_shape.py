#!/usr/bin/env python3
# Profil temporel : table (C++) contre reference exacte (Fourier + quadrature), points pres des bords de pad.
import sys, glob, time
import numpy as np
sys.path.insert(0, ".")
import test_level1 as t

def shape_error(base):
    M = t.read_meta(base)
    rows = np.genfromtxt(base + ".csv", delimiter=",", names=True)
    keys = {}
    for i, r in enumerate(rows):
        keys.setdefault((r["u"], r["v"], r["d"]), []).append(i)
    worst = 0.0
    for (u, v, d), idx in keys.items():
        spl = t.lambda_spline(M, u, v, d)
        sig_t = t.sigma_t_of(M, d)
        tau = rows["tau"][idx]
        ex = np.array([t.psi_exact(spl, M, sig_t, x) for x in tau])
        e = np.abs(rows["psi"][idx] - ex)
        if np.max(e) > worst:
            worst = np.max(e)
            info = "u=%.4f v=%.4f d=%.3f tau=%.1f" % (u, v, d, tau[np.argmax(e)])
    shape_error.info = info
    return worst

if __name__ == "__main__":
    for b in sys.argv[1:]:
        t0 = time.time()
        print("%s  profil temporel : erreur max %.2e  pire %s  (%.0f s)" % (b, shape_error(b), shape_error.info, time.time() - t0))
