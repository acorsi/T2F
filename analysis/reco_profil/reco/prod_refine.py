#!/usr/bin/env python3
"""
prod_refine.py : MODE PRODUCTION de la reconstruction des profils dE/dx. Meme algorithme que refine_geom.py
(geometrie de depart par reco_split.estimate, affinage par Levenberg-Marquardt, rallonge de fin de profil, profil final
avec lissage relache en fin de trace), mais :
  - aucune information de verite n'est utilisee par la reconstruction : elle n'a besoin que de /metadata, /pads,
    /event_id et /P/{pid, pk0, pnt, pqn, phase}. Si le fichier a /H et /L (simulation), Z, A et KE vrais sont
    ajoutes a la sortie APRES la reconstruction de chaque evenement (fonction truth_labels : recouvrement entre les
    pads touches par les trajectoires vraies et les pads de chaque demi-plan), sans jamais l'influencer ;
  - aucun evenement n'est ecarte d'apres la verite ;
  - pas de figures, pas de diagnostics ; une seule geometrie (affinee) et un seul lissage (--lam) ;
  - le profil final est celui de la derniere evaluation de la phase de rallonge (meme geometrie, meme longueurs, meme lissage,
    meme penalite : le resultat serait identique s'il etait recalcule) : pas de calcul supplementaire.
Convention de fragment : 0 = demi-plan y > y_plane, 1 = demi-plan y < y_plane (aucune identite H/L).

Usage :
    python3 prod_refine.py sortie.h5 --exe ../build/profile_basis_fast --cfg ../drift.cfg [--start 0] [--n 1000000]
            [--lam 1e-5] [--chunk 10] [--out prod] [options de refine_geom.py : --delta, --lam_geom, --lam_end_fac, ...]
Sorties : <out>_prod.h5 (contenu : voir prod_io.py) et <out>_skipped.csv (evenement, raison de l'ecart).
Necessite le serveur profile_basis_fast. Fichiers dans le meme dossier : fit_profile.py, refine_geom.py, view_fit.py,
view_pads.py, reco_dir.py, reco_split.py, rings.py, t2f_io.py, whiten_fast.py, jac_fast.py, prod_io.py.
Variables d'environnement : WHITEN_FAST=0 / JAC_FAST=0 pour desactiver les accelerations (JAC_FAST=0 : serveur ordinaire).
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np

import fit_profile

if os.environ.get("WHITEN_FAST", "1") != "0":
    import whiten_fast
    whiten_fast.install(fit_profile)

import refine_geom as rg

if os.environ.get("JAC_FAST", "1") != "0":
    import jac_fast
    jac_fast.install(rg)

import reco_split
from reco_dir import pad_waveforms
from rings import zero_window
from t2f_io import pad_xy, read_vertex
import prod_io


def read_event(f, i, geom):
    """pads de l'evenement i : UNIQUEMENT pid, x, y, pk0, pnt, pqn (rien de la verite)"""
    P = f["/P"]
    pid = np.array(P["pid"][i], dtype=int)
    x, y = pad_xy(geom, pid)
    return {"pid": pid, "x": x, "y": y, "pk0": np.array(P["pk0"][i], dtype=int),
            "pnt": np.array(P["pnt"][i], dtype=int), "pqn": np.array(P["pqn"][i], dtype=float)}


def build_state_prod(env, a, i):
    """etat d'un evenement : comme refine_geom.build_state, sans verite ; (None, raison) si ecarte"""
    f = env.f
    ev = read_event(f, i, env.geom)
    phase = int(f["/P/phase"][i])
    wfs = pad_waveforms(ev, phase, env.R, env.dt)
    amp = np.array([w[1].max() if len(w[1]) else 0.0 for w in wfs])
    res, why = reco_split.estimate(wfs, ev["x"], ev["y"], env.resp.area, env.m_h, env.thr, a,
                                   env.pitch, env.ve, env.dkg, env.t_off)
    if res is None:
        return None, "estimation : " + why
    g0 = {"v": np.array([res["xv"], a.y_plane, res["zv"]]), "u": [res["u_p"], res["u_m"]]}
    L = [rg.half_length(g0, k, ev, amp, a.y_plane, env.thr, a.k_end, a.axis_dist * env.pitch, a.lmargin)
         for k in range(2)]
    if L[0] is None or L[1] is None:
        return None, "longueur : aucun pad fort dans un demi-plan"
    keep = np.zeros(len(ev["pid"]), dtype=bool)
    for k in range(2):
        u = g0["u"][k]
        uxy = np.hypot(u[0], u[1])
        d = rg.dist_to_segment(ev["x"], ev["y"], g0["v"][0], g0["v"][1], u[0] / uxy, u[1] / uxy, L[k] * uxy)
        keep |= d <= a.pad_dist * env.pitch
    idx = np.where(keep & (ev["pnt"] > 0))[0]
    if len(idx) == 0:
        return None, "aucun pad proche de la trace"
    offs = np.concatenate([[0], np.cumsum(ev["pnt"])])
    y = np.concatenate([ev["pqn"][offs[k]:offs[k] + ev["pnt"][k]] for k in idx])
    S = {"event": i, "phase": phase, "L": L, "p": rg.params_from_geo(g0),
         "pids": ev["pid"][idx], "nts": ev["pnt"][idx], "y": y, "nr": len(y),
         "rows": [(int(ev["pid"][k]), int(ev["pk0"][k]), int(ev["pnt"][k])) for k in idx],
         "px": ev["x"].copy(), "py": ev["y"].copy(), "amp": amp, "nz": 0, "n1": 0, "ntz": 0, "wcol": None, "full": None}
    if env.rings is not None:
        keep_w, ring1, ring2 = env.rings.split(ev["pid"])
        win = zero_window(ev["pk0"][keep_w], ev["pnt"][keep_w],
                          [g0["u"][0][2], g0["u"][1][2]], a.ext_max, env.ve, env.dt * env.R)
        if win is not None:
            used = set(int(r[0]) for r in S["rows"])
            zp = [int(k) for k in ring2 if int(k) not in used]
            j0z, ntz = win
            rows, pids, nts, yv, wcol = list(S["rows"]), S["pids"], S["nts"], S["y"], None
            if zp:
                rows = rows + [(k, j0z, ntz) for k in zp]
                pids = np.concatenate([pids, np.array(zp, dtype=int)])
                nts = np.concatenate([nts, np.full(len(zp), ntz, dtype=int)])
                yv = np.concatenate([yv, np.zeros(len(zp) * ntz)])
                S["nz"], S["ntz"] = len(zp), ntz
            if a.ring1_k > 0.0:
                zp1 = [int(k) for k in ring1 if int(k) not in used]
                if zp1:
                    rows = rows + [(k, j0z, ntz) for k in zp1]
                    pids = np.concatenate([pids, np.array(zp1, dtype=int)])
                    nts = np.concatenate([nts, np.full(len(zp1), ntz, dtype=int)])
                    yv = np.concatenate([yv, np.zeros(len(zp1) * ntz)])
                    S["n1"], S["ntz"] = len(zp1), ntz
                    wcol = np.ones(len(yv))
                    wcol[len(yv) - len(zp1) * ntz:] = 1.0 / a.ring1_k
            if len(yv) > len(S["y"]):
                S["full"] = {"rows": rows, "pids": pids, "nts": nts, "y": yv, "wcol": wcol}
                if a.zero_in_geom:
                    rg.use_full_rows(S)
    return S, ""


def extend_chunk_prod(ctx, chunk, lam, a):
    """comme refine_geom.extend_chunk, mais garde pour chaque evenement le resultat de sa DERNIERE evaluation
    (S["fin"] = (r, c, sg, mh, ml)) : c'est le profil final (meme geometrie, longueurs, lissage et penalite)."""
    for S in chunk:
        S["dL"] = [0.0, 0.0]
        S["Lb"] = S["Lr"] if S["Lr"] is not None else S["L"]
    res = rg.evaluate(ctx, [(S, S["p"], None, S["Lb"]) for S in chunk], lam)
    for S, rr in zip(chunk, res):
        S["alpha2_f"] = rr[3]
        S["fin"] = (rr[0], rr[1], rr[2], rr[4], rr[5])
        st = rg.tail_state(rr[1], rr[2], rr[4], rr[5], a.n_tail, a.f_tail)
        S["notc"], S["tailr"], S["tails"] = [s[0] for s in st], [s[1] for s in st], [s[2] for s in st]
    if not a.ext:
        return
    nit_max = int(np.ceil(a.ext_max / a.ext_step)) + 1
    for it in range(nit_max):
        act = []
        for S in chunk:
            grew = False
            for k in range(2):
                if S["notc"][k] and S["dL"][k] + a.ext_step <= a.ext_max + 1.0e-9:
                    S["dL"][k] += a.ext_step
                    grew = True
            if grew:
                act.append(S)
        if not act:
            break
        items = [(S, S["p"], S["alpha2_f"], [S["Lb"][k] + S["dL"][k] for k in range(2)]) for S in act]
        res = rg.evaluate(ctx, items, lam)
        for S, rr in zip(act, res):
            S["fin"] = (rr[0], rr[1], rr[2], rr[4], rr[5])
            st = rg.tail_state(rr[1], rr[2], rr[4], rr[5], a.n_tail, a.f_tail)
            S["notc"], S["tailr"], S["tails"] = [s[0] for s in st], [s[1] for s in st], [s[2] for s in st]


LABEL_STEP = 0.2   # pas (cm) le long de la trajectoire vraie projetee dans le plan (x, y)


def truth_labels(env, a, i, u_rec):
    """Labels de verite (Z, A, KE), attribues aux fragments 0/1 APRES la reconstruction de l'evenement i :
    ils ne sont jamais utilises par elle.
    Methode : pour H et pour L, pads de l'evenement (/P/pid) a moins d'un pitch de la trajectoire vraie projetee dans
    (x, y) (pas de LABEL_STEP depuis le vertex, jusqu'au pad touche le plus eloigne) ; P_0, P_1 = pads de l'evenement dans
    y > y_plane, y < y_plane (groupes utilises par la reconstruction). On retient l'affectation conjointe (H->0, L->1) ou
    (H->1, L->0) qui maximise le nombre de pads communs. Un pad proche du vertex, touche par les deux traces, compte
    pareil dans les deux affectations : il ne decide rien.
    Renvoie (Z[2], A[2], KE[2], cos[2], egalite) ; Z = -1, A = -1, KE = NaN, cos = NaN si le fichier n'a pas de /H, /L
    (donnees reelles) ; egalite = 1 si les deux affectations ont le meme score (a doit rester 0).
    cos[k] = cosinus entre la direction reconstruite du fragment k et la direction vraie du fragment attribue (controle)."""
    f = env.f
    nul = ([-1, -1], [-1, -1], [np.nan, np.nan], [np.nan, np.nan], 0)
    for g in ("H", "L"):
        if ("/" + g) not in f or "Z" not in f["/" + g] or "theta" not in f["/" + g]:
            return nul
    pid = np.array(f["/P"]["pid"][i], dtype=int)
    if len(pid) == 0:
        return nul
    px, py = pad_xy(env.geom, pid)
    vx, vy, vz = read_vertex(f, i)
    s = np.arange(0.0, float(np.hypot(px - vx, py - vy).max()) + env.pitch, LABEL_STEP)
    side = [py > a.y_plane, py < a.y_plane]
    u3, score = {}, {}
    for g in ("H", "L"):
        t, p = float(f["/%s/theta" % g][i]), float(f["/%s/phi" % g][i])
        ux = np.sin(t) * np.cos(p)
        if env.axis == "Y":
            uy, uz = np.cos(t), np.sin(t) * np.sin(p)
        else:
            uy, uz = np.sin(t) * np.sin(p), np.cos(t)
        u3[g] = np.array([ux, uy, uz])
        n = np.hypot(ux, uy)
        tx, ty = vx + s * ux / n, vy + s * uy / n
        d2 = ((px[:, None] - tx[None, :]) ** 2 + (py[:, None] - ty[None, :]) ** 2).min(axis=1)
        touched = d2 <= env.pitch ** 2
        score[g] = [int((touched & side[k]).sum()) for k in range(2)]
    sa = score["H"][0] + score["L"][1]
    sb = score["H"][1] + score["L"][0]
    kh = 0 if sa >= sb else 1
    Z, A, KE, cs = [-1, -1], [-1, -1], [np.nan, np.nan], [np.nan, np.nan]
    for g, k in (("H", kh), ("L", 1 - kh)):
        Z[k], A[k], KE[k] = int(f["/%s/Z" % g][i]), int(f["/%s/A" % g][i]), float(f["/%s/KE" % g][i])
        cs[k] = float(np.dot(u3[g], u_rec[k]))
    return (Z, A, KE, cs, 1 if sa == sb else 0)


class Collector:
    def __init__(self):
        self.rows = []
        self.val, self.sig = [], []
        self.n_tie = 0

    def add(self, S, a, event_id, env):
        r, c, sg, mh, ml = S["fin"]
        geo = rg.geo_from_params(S["p"], a.y_plane)
        lab = truth_labels(env, a, S["event"], geo["u"])
        self.n_tie += lab[4]
        self.rows.append((S["event"], event_id, geo["v"], geo["u"][0], geo["u"][1], S["p"], S["chi2d0"], S["chi2d1"],
                          float(r[:S["nr"]] @ r[:S["nr"]]) / S["nr"], S["nit"], list(S["Lb"]), list(S["dL"]),
                          [1 if b else 0 for b in S["notc"]], list(S["tailr"]), list(S["tails"]), (mh, ml))
                         + lab[:4])
        self.val.append(np.asarray(c, dtype=float))
        self.sig.append(np.asarray(sg, dtype=float))

    def arrays(self):
        d = prod_io.empty_arrays()
        if not self.rows:
            return d
        cols = list(zip(*self.rows))
        for k, name in enumerate(prod_io.PER_EVENT):
            d[name] = np.array(cols[k], dtype=prod_io.DTYPES.get(name, "f8"))
        n_all = [int(len(v)) for v in self.val]
        d["prof_off"] = np.concatenate([[0], np.cumsum(n_all)]).astype("i8")
        d["prof_val"] = np.concatenate(self.val)
        d["prof_sig"] = np.concatenate(self.sig)
        return d


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("h5file")
    rg.add_common_args(p)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n", type=int, default=1000000000)
    p.add_argument("--chunk", type=int, default=10, help="evenements traites ensemble (taille des fichiers d'echange)")
    p.add_argument("--lam", type=float, default=1.0e-5, help="lissage du profil final (et de la rallonge)")
    p.add_argument("--out", type=str, default="prod")
    p.add_argument("--plot_n", type=int, default=0, help="ignore (compatibilite avec run_par.py)")
    p.add_argument("--tmpdir", type=str, default="",
                   help="dossier des fichiers d'echange requete/reponse (disque local du noeud) ; defaut : a cote de --out")
    a = p.parse_args()
    free = rg.parse_free(a)
    t_start = time.time()

    env = rg.Env(a)
    f = env.f
    if "/event_id" not in f:
        sys.exit("[prod_refine] /event_id absent du fichier")
    i1 = min(env.N, a.start + a.n)
    print("[prod_refine] %s : evenements %d a %d | lam_geom = %g, lam = %g, lissage de fin x %g sur %.2f cm"
          % (a.h5file, a.start, i1 - 1, a.lam_geom, a.lam, a.lam_end_fac, a.lam_end_zone))
    attrs = {"delta": a.delta, "lam": a.lam, "lam_geom": a.lam_geom, "lam_end_fac": a.lam_end_fac,
             "lam_end_zone": a.lam_end_zone, "y_plane": a.y_plane, "ext_step": a.ext_step, "ext_max": a.ext_max,
             "convention": "fragment 0 : demi-plan y > y_plane ; fragment 1 : demi-plan y < y_plane ; aucune identite H/L",
             "source": a.h5file, "start": a.start, "version": "prod_refine 2",
             "labels": "frag_Z, frag_A, frag_KE_true : verite de simulation ajoutee apres la reconstruction (H et L affectes aux fragments 0/1 par recouvrement entre pads touches par la trace vraie et pads de chaque demi-plan) ; -1 si absente"}
    # prefixe des fichiers d'echange (requete/reponse)
    if a.tmpdir:
        if not os.path.isdir(a.tmpdir):
            os.makedirs(a.tmpdir)
        xprefix = os.path.join(a.tmpdir, os.path.basename(a.out))
    else:
        xprefix = a.out
    coll = Collector()
    skipped = []
    server = None
    try:
        for c0 in range(a.start, i1, a.chunk):
            chunk = []
            for i in range(c0, min(c0 + a.chunk, i1)):
                S, why = build_state_prod(env, a, i)
                if S is None:
                    skipped.append((i, why))
                else:
                    chunk.append(S)
            if not chunk:
                continue
            if server is None:
                server = rg.Server(a.exe, a.cfg, a.exe_args.split())
                ctx = rg.Ctx(a, server, env.wh, env.step, a.delta, a.y_plane, xprefix)
            rg.refine_chunk(ctx, chunk, a.lam_geom, free, a.max_iter, a.tol)
            for S in chunk:
                S["Lr"] = None
                if a.lrecalc:
                    gg = rg.geo_from_params(S["p"], a.y_plane)
                    evxy = {"x": S["px"], "y": S["py"]}
                    Lk = [rg.half_length(gg, k, evxy, S["amp"], a.y_plane, env.thr, a.k_end,
                                         a.axis_dist * env.pitch, a.lmargin) for k in range(2)]
                    if Lk[0] is not None and Lk[1] is not None:
                        S["Lr"] = Lk
            extend_chunk_prod(ctx, chunk, a.lam, a)
            for S in chunk:
                coll.add(S, a, int(f["/event_id"][S["event"]]), env)
            if (c0 - a.start) % (10 * a.chunk) == 0:
                print("[prod_refine] evenements %d-%d traites (%d ecartes), %.0f s"
                      % (a.start, min(c0 + a.chunk, i1) - 1, len(skipped), time.time() - t_start), flush=True)
    finally:
        if server is not None:
            server.close()
        for ext in ("_req.txt", "_rsp.bin"):
            try:
                os.remove(xprefix + ext)
            except OSError:
                pass

    d = coll.arrays()
    # ecriture sous un nom provisoire puis renommage : un fichier _prod.h5 existant est donc toujours complet
    prod_io.write_file(a.out + "_prod.h5.part", d, attrs)
    os.rename(a.out + "_prod.h5.part", a.out + "_prod.h5")
    with open(a.out + "_skipped.csv", "w") as g:
        g.write("event,reason\n")
        for i, why in skipped:
            g.write("%d,%s\n" % (i, why.replace(",", ";")))
    n = len(d["event_index"])
    print("[prod_refine] labels de verite : %d evenements etiquetes ; affectations a egalite : %d (doit rester 0)"
          % (int((d["frag_Z"][:, 0] >= 0).sum()), coll.n_tie))
    print("[prod_refine] %d evenements ecrits dans %s_prod.h5, %d ecartes (%s_skipped.csv), %.0f s"
          % (n, a.out, len(skipped), a.out, time.time() - t_start))
    if n:
        print("[prod_refine] chi2/ndl geometrie : mediane %.3f ; iterations : mediane %d ; fragments rallonges : %d / %d ; "
              "fin non contenue apres rallonge : %d"
              % (np.median(d["chi2_geom"]), int(np.median(d["n_iter"])), int((d["dL"] > 0).sum()), d["dL"].size,
                 int(d["not_contained"].sum())))


if __name__ == "__main__":
    main()
