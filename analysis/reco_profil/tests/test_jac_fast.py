"""
test_jac_fast.py : compare refine_geom.refine_chunk (original) et jac_fast.refine_chunk_fast sur des evenements SYNTHETIQUES
(vertex, angles et profils tires au hasard ; signal = reponse du serveur + bruit gaussien correle ; aucune donnee reelle).
Usage (depuis le dossier fast_basis) :  python3 tests/test_jac_fast.py [n_evenements]
Attendu : memes parametres et memes nombres d'iterations (ecarts ~1e-9 au plus), moins de calcul et d'echanges.
"""
import copy
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "py"))
sys.path.insert(0, HERE)
import fit_profile as fp
import refine_geom as rg
import jac_fast
import make_req

if os.environ.get("WHITEN_FAST") == "1":
    import whiten_fast
    whiten_fast.install(fp)

NEV = int(sys.argv[1]) if len(sys.argv) > 1 else 6
EXE = os.path.join(HERE, "..", "build", "profile_basis_fast")
CFG = os.path.join(HERE, "..", "drift.cfg")
TMP = os.path.join(HERE, "tmp_jt")
STEP, DELTA = 0.05, 0.2


class Model:                                   # covariance de bruit synthetique, Toeplitz, ecart type 300 e-
    def cov(self, n, s):
        i = np.arange(n)
        return (300.0 ** 2) * s * (0.6 ** np.abs(i[:, None] - i[None, :]))

    def ser_scale_pad(self, enc):
        return 1.0


rng = np.random.RandomState(5)
wh = fp.Whitener(Model(), None, True)
server = rg.Server(EXE, CFG, [])
nreq = [0, 0]
run0 = server.run


def counting_run(req, rsp):
    run0(req, rsp)
    nreq[0] += 1
    nreq[1] += os.path.getsize(rsp)
server.run = counting_run
ctx = rg.Ctx(types.SimpleNamespace(lam_end_fac=1.0, lam_end_zone=0.0), server, wh, STEP, DELTA, 0.0, TMP)

P = make_req.pads()
pitch = make_req.pitch
states, ptrue = [], []
for e in range(NEV):
    pt = np.array([rng.uniform(-0.5, 0.5), rng.uniform(6, 14),
                   rng.uniform(0.2, 1.3), rng.uniform(-0.4, 0.4), np.pi - rng.uniform(0.2, 1.3), rng.uniform(-0.4, 0.4)])
    geo = rg.geo_from_params(pt, 0.0)
    L = [rng.uniform(2.0, 3.5), rng.uniform(2.0, 3.5)]
    sel = np.zeros(len(P), dtype=bool)
    for k in range(2):
        u = geo["u"][k]; uxy = np.hypot(u[0], u[1]); ex, ey = u[0] / uxy, u[1] / uxy
        s = np.clip((P[:, 0] - geo["v"][0]) * ex + P[:, 1] * ey, 0, L[k] * uxy)
        sel |= np.hypot(P[:, 0] - (geo["v"][0] + s * ex), P[:, 1] - (geo["v"][1] + s * ey)) < 2.0 * pitch
    idx = np.where(sel)[0]
    rows = [(int(i), 20, 116) for i in idx]
    S = {"event": e, "phase": e % 2, "L": L, "rows": rows, "pids": np.array([r[0] for r in rows]),
         "nts": np.array([r[2] for r in rows]), "y": np.zeros(116 * len(rows)), "wcol": None, "nr": 116 * len(rows),
         "full": None}
    # signal vrai : reponse du serveur a la vraie geometrie, profil positif aleatoire, bruit correle
    recs = [{"tag": 0, "phase": S["phase"], "v": geo["v"], "u": geo["u"], "L": L, "pads": rows}]
    fp.write_request(TMP + "_t.txt", STEP, DELTA, recs)
    server.run(TMP + "_t.txt", TMP + "_t.bin")
    for tag, mh, ml, A in fp.read_response_iter(TMP + "_t.bin"):
        c = np.concatenate([1.5e4 * np.exp(-0.5 * ((np.arange(mh) * DELTA - rng.uniform(0.5, 1.5)) / 0.5) ** 2),
                            1.5e4 * np.exp(-0.5 * ((np.arange(ml) * DELTA - rng.uniform(0.5, 1.5)) / 0.5) ** 2)])
        y = A.T @ c
    off = 0
    for pid, n in zip(S["pids"], S["nts"]):
        Lc = np.linalg.cholesky(Model().cov(int(n), 1.0))
        y[off:off + n] += Lc @ rng.randn(int(n))
        off += n
    S["y"] = y
    S["p"] = pt + np.array([0.02, 0.03, 0.01, 0.05, -0.01, -0.05])     # depart ecarte de la verite
    states.append(S)
    ptrue.append(pt)
nreq[0] = nreq[1] = 0

free = [0, 1, 2, 3, 4, 5]
orig = rg.refine_chunk
A_states = [copy.deepcopy(S) for S in states]
orig(ctx, A_states, 1.0e-4, free, 8, 1.0e-4)
a_req, a_mb = nreq[0], nreq[1] / 1e6
nreq[0] = nreq[1] = 0
jac_fast.install(rg)
B_states = [copy.deepcopy(S) for S in states]
rg.refine_chunk(ctx, B_states, 1.0e-4, free, 8, 1.0e-4)
b_req, b_mb = nreq[0], nreq[1] / 1e6
server.close()

dp = max(np.abs(a["p"] - b["p"]).max() for a, b in zip(A_states, B_states))
dr = max(np.abs(a["r"] - b["r"]).max() / np.abs(a["r"]).max() for a, b in zip(A_states, B_states))
print("evenements : %d ; pads par evenement (mediane) : %d" % (NEV, np.median([len(S["rows"]) for S in states])))
print("iterations (original / jac_fast) :", [a["nit"] for a in A_states], [b["nit"] for b in B_states])
print("ecart max sur les parametres finaux : %.2e  | sur le residu (relatif) : %.2e" % (dp, dr))
print("requetes serveur : %d / %d ; reponses : %.1f Mo / %.1f Mo (jac_fast = %.0f %% du volume)" % (a_req, b_req, a_mb, b_mb, 100.0 * b_mb / a_mb))
for f in ("_t.txt", "_t.bin", "_req.txt", "_rsp.bin"):
    try:
        os.remove(TMP + f)
    except OSError:
        pass
print("RESULTAT : %s" % ("OK" if dp < 1e-7 and [a["nit"] for a in A_states] == [b["nit"] for b in B_states] else "A EXAMINER"))
