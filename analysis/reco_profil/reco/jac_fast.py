"""
jac_fast.py : jacobien par fragment pour refine_geom.py, SANS modifier refine_geom.py ni fit_profile.py.
(necessite le serveur profile_basis_fast : format de requete etendu "#fast 1")

Les parametres libres sont p = [x, z, a0, m0, a1, m1]. Les angles a0, m0 ne changent que le signal du fragment 0
(hats 0 .. mh-1), a1, m1 que celui du fragment 1 : le signal d'un chapeau ne depend que de SA demi-droite et du vertex.
Pour une difference finie en a0 ou m0, on ne demande donc au serveur que le fragment 0 ; les colonnes blanchies du
fragment 1 sont celles de l'evaluation de base (le point courant), deja calculees. Idem en a1, m1 (fragment 1 seul).
Le residu, l'ajustement du profil (NNLS) et le jacobien sont ceux de refine_geom.py : seul le calcul des colonnes
change. Gain : -29 % de calcul C++, de blanchiment et d'echanges par iteration.

Usage :   import refine_geom, jac_fast ;  jac_fast.install(refine_geom)   (avant refine_geom.main())
"""
import struct

import numpy as np

_rg = None
FRAG_MASK = {0: 3, 1: 3, 2: 1, 3: 1, 4: 2, 5: 2}     # parametre -> fragments a recalculer (bit 0 : fragment 0, bit 1 : fragment 1)


def write_request_ext(path, step_eff, delta, recs):
    with open(path, "w") as g:
        g.write("#fast 1\n")
        g.write("%.10g %.10g %d\n" % (step_eff, delta, len(recs)))
        for r in recs:
            g.write("%d %d\n" % (r["tag"], r["phase"]))
            g.write("%.10g %.10g %.10g\n" % tuple(r["v"]))
            g.write("%d\n" % r["mask"])
            for k in range(2):
                g.write("%.12g %.12g %.12g %.10g\n" % (r["u"][k][0], r["u"][k][1], r["u"][k][2], r["L"][k]))
            g.write("%d\n" % len(r["pads"]))
            for (pid, j0, nt) in r["pads"]:
                g.write("%d %d %d\n" % (pid, j0, nt))


def read_response_ext(path, masks):
    """produit (tag, mh, ml, A, mask) ; A n'a que les lignes des fragments demandes (fragment 0 d'abord)"""
    with open(path, "rb") as g:
        while True:
            h = g.read(16)
            if len(h) < 16:
                return
            tag, mh, ml, nr = struct.unpack("<4i", h)
            mask = masks[tag]
            nc = (mh if mask & 1 else 0) + (ml if mask & 2 else 0)
            n = nc * nr
            A = np.frombuffer(g.read(8 * n), dtype="<f8", count=n).reshape(nc, nr)
            yield tag, mh, ml, A, mask


def evaluate_fast(ctx, items, lam, want_G=False):
    """meme contrat que refine_geom.evaluate ; un item peut avoir un 5e element : mask (defaut 3).
    mask 1 ou 2 : seul ce fragment est calcule par le serveur, l'autre vient de S["Aw_base"] (colonnes blanchies
    du point courant). Remplit ctx.aw_list : Aw complet de chaque item (meme ordre que items)."""
    rg = _rg
    recs, endws, masks = [], {}, {}
    fac, zone = ctx.a.lam_end_fac, ctx.a.lam_end_zone
    for n, it in enumerate(items):
        S, p, alpha2, Lov = it[:4]
        mask = it[4] if len(it) > 4 else 3
        masks[n] = mask
        geo = rg.geo_from_params(p, ctx.y_plane)
        Ls = S["L"] if Lov is None else Lov
        recs.append({"tag": n, "phase": S["phase"], "v": geo["v"], "u": geo["u"], "L": Ls, "pads": S["rows"],
                     "mask": mask})
        if fac != 1.0 and zone > 0.0 and S.get("dL") is not None:
            endws[n] = (fac, [max(1, int(np.ceil((Ls[k] - S["dL"][k] - zone) / ctx.delta - 1.0e-9)))
                              for k in range(2)])
    req, rsp = ctx.tmp + "_req.txt", ctx.tmp + "_rsp.bin"
    write_request_ext(req, ctx.step, ctx.delta, recs)
    ctx.server.run(req, rsp)
    out = [None] * len(items)
    ctx.aw_list = [None] * len(items)
    for tag, mh, ml, A, mask in rg.read_response_ext(rsp, masks):
        S, p, alpha2, Lov = items[tag][:4]
        Aw, yw = ctx.wh.apply(S["pids"], S["nts"], A, S["y"])
        if S["wcol"] is not None:                 # couronne 1 : contrainte faible (--ring1_k)
            Aw = Aw * S["wcol"][None, :]
            yw = yw * S["wcol"]
        if mask != 3:
            base = S["Aw_base"]
            if base.shape[0] != mh + ml or base.shape[1] != Aw.shape[1]:
                raise RuntimeError("jac_fast : cache Aw_base incoherent (%s contre %d x %d)" % (base.shape, mh + ml, Aw.shape[1]))
            Aw = np.vstack([Aw, base[mh:]]) if mask == 1 else np.vstack([base[:mh], Aw])
        ctx.aw_list[tag] = Aw
        ew = None if tag not in endws else (endws[tag][0], endws[tag][1][0], endws[tag][1][1])
        if want_G:
            c, sig, r, al, G0, G = rg.solve_profile_r(Aw, yw, mh, ml, ctx.delta, lam, alpha2, want_G=True, endw=ew)
            out[tag] = (r, c, sig, al, mh, ml, G0, G)
        else:
            c, sig, r, al = rg.solve_profile_r(Aw, yw, mh, ml, ctx.delta, lam, alpha2, endw=ew)
            out[tag] = (r, c, sig, al, mh, ml)
    return out


def refine_chunk_fast(ctx, chunk, lam, free, max_iter, tol):
    """refine_geom.refine_chunk avec jacobien par fragment (meme algorithme, memes parametres)"""
    rg = _rg
    FD_STEP = rg.FD_STEP
    res = rg.evaluate(ctx, [(S, S["p"], None, None) for S in chunk], lam)
    for S, rr, aw in zip(chunk, res, ctx.aw_list):
        S["Aw_base"] = aw
        S["r"], S["alpha2"] = rr[0], rr[3]
        S["F"] = float(S["r"] @ S["r"])
        S["F0"] = S["F"]
        S["chi2d0"] = float(S["r"][:S["nr"]] @ S["r"][:S["nr"]]) / S["nr"]
        S["mu"] = 1.0e-2
        S["done"] = False
        S["nit"] = 0
    items = []
    for S in chunk:
        for k in free:
            pk = S["p"].copy()
            pk[k] += FD_STEP[k]
            items.append((S, pk, S["alpha2"], None, FRAG_MASK[k]))
    res = rg.evaluate(ctx, items, lam)
    n = 0
    for S in chunk:
        S["J"] = np.zeros((len(S["r"]), 6))
        for k in free:
            S["J"][:, k] = (res[n][0] - S["r"]) / FD_STEP[k]
            n += 1

    for it in range(max_iter):
        act = [S for S in chunk if not S["done"]]
        if not act:
            break
        trials = []
        for S in act:
            step = rg.lm_step(S["J"], S["r"], S["mu"], free)
            trials.append(S["p"] + step)
        res = rg.evaluate(ctx, [(S, pt, S["alpha2"], None) for S, pt in zip(act, trials)], lam)
        aws = ctx.aw_list
        need_J = []
        for S, pt, rr, aw in zip(act, trials, res, aws):
            Fn = float(rr[0] @ rr[0])
            S["nit"] += 1
            if Fn < S["F"]:
                rel = (S["F"] - Fn) / S["F"]
                S["p"], S["r"], S["F"] = pt, rr[0], Fn
                S["Aw_base"] = aw
                S["mu"] = max(S["mu"] / 3.0, 1.0e-4)
                if rel < tol:
                    S["done"] = True
                else:
                    need_J.append(S)
            else:
                S["mu"] *= 5.0
                if S["mu"] > 1.0e6:
                    S["done"] = True
        if need_J:
            items = []
            for S in need_J:
                for k in free:
                    pk = S["p"].copy()
                    pk[k] += FD_STEP[k]
                    items.append((S, pk, S["alpha2"], None, FRAG_MASK[k]))
            res = rg.evaluate(ctx, items, lam)
            n = 0
            for S in need_J:
                S["J"] = np.zeros((len(S["r"]), 6))
                for k in free:
                    S["J"][:, k] = (res[n][0] - S["r"]) / FD_STEP[k]
                    n += 1
    for S in chunk:
        S["chi2d1"] = float(S["r"][:S["nr"]] @ S["r"][:S["nr"]]) / S["nr"]
        S.pop("Aw_base", None)            # les lignes changent ensuite (pseudo-donnees nulles) : le cache n'est plus valable
        rg.use_full_rows(S)


def install(rg):
    global _rg
    for nm in ("evaluate", "refine_chunk", "geo_from_params", "solve_profile_r", "lm_step", "use_full_rows", "FD_STEP"):
        if not hasattr(rg, nm):
            raise RuntimeError("jac_fast : refine_geom n'a pas de %s (version differente de celle fournie)" % nm)
    _rg = rg
    rg.read_response_ext = read_response_ext
    rg.evaluate = evaluate_fast
    rg.refine_chunk = refine_chunk_fast
