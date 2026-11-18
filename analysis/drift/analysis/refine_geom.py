#!/usr/bin/env python3
"""
refine_geom.py : etape 2. Affine vertex et directions en minimisant le chi2 PROFILE des echantillons des pads.

Pour chaque evenement, 6 parametres : x_v, z_v (le vertex reste dans le plan y = y_plane) et, pour chaque
demi-droite, l'angle dans le plan alpha (direction de (ux, uy)) et la pente en z m = uz / |u_xy|.
Valeurs de depart : reco_split.estimate. Pas de colinearite imposee.

Pour chaque jeu de parametres, le profil q(l) est resolu a l'interieur (meme solveur que fit_profile.py, lissage
lam_geom fixe, c >= 0) : l'objectif est le residu complet  ||C^(-1/2)(A c - y)||^2 + penalite  (projection
variable). Minimisation de Levenberg-Marquardt, jacobien par differences finies. Les longueurs des demi-droites
et les pads utilises sont fixes (ceux de la geometrie de depart).

Toutes les evaluations passent par ./profile_basis en mode serveur (la table n'est construite qu'une fois).

Sorties : tableau de la geometrie avant / apres (erreurs vs verite), profil (estime initial, affine, vrai),
diagnostic biais de lissage / variance (matrice de resolution), charge ajustee vs /P/pq,
profil fenetre (moyenne glissante --win, sortie au pas delta, compare a la verite fenetree au niveau voxel),
fichiers <out>_geom.csv, <out>_prof.csv, <out>_diag.csv, <out>_wsum.csv, <out>_win.csv, <out>_flags.csv,
figures <out>_charge.png, <out>_evt*.png, <out>_wevt*.png.

Fin de profil (grille adaptee a la trace) : la longueur L de chaque demi-droite est estimee depuis les pads
(sans la verite). Un profil ajuste qui ne retombe pas a zero en bout de grille (moyenne des --n_tail derniers
noeuds > --f_tail x maximum du fragment) signale de la charge au-dela de la grille, repliee sur la fin du profil :
la grille de ce fragment est alors rallongee par pas de --ext_step (max --ext_max) jusqu'a ce que le profil
retombe. Rien n'est modifie en silence : <out>_flags.csv (par fragment : rallonge, ext_cm, fin_non_contenue,
queue/max, deficit vs verite) ; <out>_win.csv marque par noeud la zone rallongee ; liste ATTENTION dans le resume.
Le range pour l'identification est le point ou le profil ajuste retombe, jamais la fin de la grille.

Usage :
  python3 refine_geom.py sortie.h5 --exe ../build/profile_basis --cfg ../drift.cfg --n 100 --out rg1
"""

import argparse
import os
import subprocess
import sys

import h5py
import numpy as np

from t2f_io import pad_geometry, read_vertex, event_pads, load_pad_capacitances
from view_fit import Response, FitModel
from reco_dir import pad_waveforms
import reco_split
from fit_profile import (geometry_true, geometry_est, half_length, dist_to_segment, write_request,
                         read_response_iter, Whitener, solve_profile_r, truth_nodes)

PNAMES = ["x", "z", "a0", "m0", "a1", "m1"]
FD_STEP = np.array([0.02, 0.05, 0.005, 0.02, 0.005, 0.02])       # pas des differences finies (cm, cm, rad, -, rad, -)
MAX_STEP = np.array([0.15, 0.4, 0.06, 0.25, 0.06, 0.25])         # pas maximal d'une iteration


# ---------------------------------------------------------------------------
def params_from_geo(geo):
    p = np.zeros(6)
    p[0], p[1] = geo["v"][0], geo["v"][2]
    for k in range(2):
        u = geo["u"][k]
        uxy = np.hypot(u[0], u[1])
        p[2 + 2 * k] = np.arctan2(u[1], u[0])
        p[3 + 2 * k] = u[2] / uxy
    return p


def geo_from_params(p, y_plane):
    u = []
    for k in range(2):
        a, m = p[2 + 2 * k], p[3 + 2 * k]
        w = np.array([np.cos(a), np.sin(a), m])
        u.append(w / np.linalg.norm(w))
    return {"v": np.array([p[0], y_plane, p[1]]), "u": u}


def angle_deg(u, v):
    return float(np.degrees(np.arccos(max(min(float(u @ v), 1.0), -1.0))))


def pct(v):
    v = np.asarray(v, dtype=float)
    v = np.abs(v[np.isfinite(v)])
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    return (float(np.median(v)), float(np.percentile(v, 68)), float(np.percentile(v, 95)))


# ---------------------------------------------------------------------------
# moyenne glissante (fenetre de largeur w, sortie aux noeuds, pas delta)
# ---------------------------------------------------------------------------
def window_matrix(M, delta, w):
    """S (M, M) : moyenne glissante de largeur w de l'interpolation lineaire des chapeaux, evaluee
    aux noeuds ; fenetre tronquee a [0, (M-1) delta] et renormalisee aux bords.
    Renvoie (S, full) : full[i] vrai si la fenetre entiere tient dans le domaine."""
    Lmax = (M - 1) * delta
    g = np.linspace(0.0, Lmax, 20 * (M - 1) + 1)
    HV = np.clip(1.0 - np.abs(g[:, None] - delta * np.arange(M)[None, :]) / delta, 0.0, None)
    S = np.zeros((M, M))
    for i in range(M):
        aa, bb = max(0.0, i * delta - 0.5 * w), min(Lmax, i * delta + 0.5 * w)
        sel = (g >= aa - 1.0e-12) & (g <= bb + 1.0e-12)
        S[i] = HV[sel].mean(axis=0)
    ln = delta * np.arange(M)
    full = (ln >= 0.5 * w - 1.0e-9) & (ln <= Lmax - 0.5 * w + 1.0e-9)
    return S, full


def window_truth(dE, W, step, delta, M, w):
    """moyenne glissante de largeur w du profil vrai des voxels (e-/cm), aux noeuds ;
    fenetre tronquee a [0, (M-1) delta] comme dans window_matrix."""
    q = np.asarray(dE, dtype=float) / W / step
    s = step * (np.arange(len(q)) + 0.5)
    Lmax = (M - 1) * delta
    h = 0.5 * step
    out = np.zeros(M)
    for i in range(M):
        aa, bb = max(0.0, i * delta - 0.5 * w), min(Lmax, i * delta + 0.5 * w)
        ov = np.clip(np.minimum(bb, s + h) - np.maximum(aa, s - h), 0.0, None)
        out[i] = (ov * q).sum() / (bb - aa)
    return out


# ---------------------------------------------------------------------------
class Server:
    """./profile_basis en mode serveur"""

    def __init__(self, exe, cfg, extra):
        cmd = [exe, cfg, "--serve"] + extra
        print("[refine_geom] " + " ".join(cmd) + "  (construction de la table...)")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True, bufsize=1)
        while True:
            line = self.p.stdout.readline()
            if line == "":
                sys.exit("[refine_geom] profile_basis s'est arrete avant @@READY (voir les messages ci-dessus)")
            if line.startswith("@@READY"):
                break
            if line.startswith("reponse") or line.startswith("PadPlane"):
                print("  | " + line.rstrip())
        print("[refine_geom] serveur pret")

    def run(self, req, rsp):
        self.p.stdin.write("%s %s\n" % (req, rsp))
        self.p.stdin.flush()
        while True:
            line = self.p.stdout.readline()
            if line == "":
                sys.exit("[refine_geom] profile_basis s'est arrete pendant une requete")
            if line.startswith("@@DONE"):
                return
            if line.startswith("@@ERROR"):
                sys.exit("[refine_geom] " + line.strip())

    def close(self):
        try:
            self.p.stdin.write("quit\n")
            self.p.stdin.flush()
            self.p.wait(timeout=20)
        except Exception:
            self.p.kill()


# ---------------------------------------------------------------------------
class Ctx:
    """tout ce qui est commun a une evaluation"""

    def __init__(self, a, server, wh, step, delta, y_plane, tmp):
        self.a, self.server, self.wh, self.step, self.delta, self.y_plane, self.tmp = a, server, wh, step, delta, y_plane, tmp


def evaluate(ctx, items, lam, want_G=False):
    """items : liste de (S, p, alpha2, Lov) avec Lov = longueurs a utiliser (None : S["L"]) ;
    renvoie une liste de (r, c, sig, alpha2, mh, ml) dans le meme ordre
    (avec want_G : (r, c, sig, alpha2, mh, ml, G0, G), pour la matrice de resolution)"""
    recs, byt = [], {}
    for n, (S, p, alpha2, Lov) in enumerate(items):
        geo = geo_from_params(p, ctx.y_plane)
        recs.append({"tag": n, "phase": S["phase"], "v": geo["v"], "u": geo["u"],
                     "L": S["L"] if Lov is None else Lov, "pads": S["rows"]})
    req, rsp = ctx.tmp + "_req.txt", ctx.tmp + "_rsp.bin"
    write_request(req, ctx.step, ctx.delta, recs)
    ctx.server.run(req, rsp)
    out = [None] * len(items)
    for tag, mh, ml, A in read_response_iter(rsp):
        S, p, alpha2, Lov = items[tag]
        Aw, yw = ctx.wh.apply(S["pids"], S["nts"], A, S["y"])
        if want_G:
            c, sig, r, al, G0, G = solve_profile_r(Aw, yw, mh, ml, ctx.delta, lam, alpha2, want_G=True)
            out[tag] = (r, c, sig, al, mh, ml, G0, G)
        else:
            c, sig, r, al = solve_profile_r(Aw, yw, mh, ml, ctx.delta, lam, alpha2)
            out[tag] = (r, c, sig, al, mh, ml)
    return out


def lm_step(J, r, mu, free):
    Jf = J[:, free]
    H = Jf.T @ Jf
    g = Jf.T @ r
    d = np.diag(H).copy()
    d = np.maximum(d, 1.0e-12 * max(d.max(), 1.0e-300))
    try:
        dl = -np.linalg.solve(H + mu * np.diag(d), g)
    except np.linalg.LinAlgError:
        dl = np.zeros(len(free))
    step = np.zeros(6)
    step[free] = dl
    return np.clip(step, -MAX_STEP, MAX_STEP)


def refine_chunk(ctx, chunk, lam, free, max_iter, tol):
    # evaluation initiale (alpha2 calcule ici, puis fige)
    res = evaluate(ctx, [(S, S["p"], None, None) for S in chunk], lam)
    for S, rr in zip(chunk, res):
        S["r"], S["alpha2"] = rr[0], rr[3]
        S["F"] = float(S["r"] @ S["r"])
        S["F0"] = S["F"]
        S["chi2d0"] = float(S["r"][:S["nr"]] @ S["r"][:S["nr"]]) / S["nr"]
        S["mu"] = 1.0e-2
        S["done"] = False
        S["nit"] = 0
    # jacobiens initiaux : une evaluation par parametre libre et par evenement, en une seule requete
    items = []
    for S in chunk:
        for k in free:
            pk = S["p"].copy()
            pk[k] += FD_STEP[k]
            items.append((S, pk, S["alpha2"], None))
    res = evaluate(ctx, items, lam)
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
            step = lm_step(S["J"], S["r"], S["mu"], free)
            trials.append(S["p"] + step)
        res = evaluate(ctx, [(S, pt, S["alpha2"], None) for S, pt in zip(act, trials)], lam)
        need_J = []
        for S, pt, rr in zip(act, trials, res):
            Fn = float(rr[0] @ rr[0])
            S["nit"] += 1
            if Fn < S["F"]:
                rel = (S["F"] - Fn) / S["F"]
                S["p"], S["r"], S["F"] = pt, rr[0], Fn
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
                    items.append((S, pk, S["alpha2"], None))
            res = evaluate(ctx, items, lam)
            n = 0
            for S in need_J:
                S["J"] = np.zeros((len(S["r"]), 6))
                for k in free:
                    S["J"][:, k] = (res[n][0] - S["r"]) / FD_STEP[k]
                    n += 1
    for S in chunk:
        S["chi2d1"] = float(S["r"][:S["nr"]] @ S["r"][:S["nr"]]) / S["nr"]


# ---------------------------------------------------------------------------
# fin de profil : test de confinement et rallonge de la grille
# ---------------------------------------------------------------------------
def tail_state(c, sg, mh, ml, n_tail, f_tail):
    """par fragment : (fin non contenue, moyenne des n_tail derniers noeuds / max, sigma de cette moyenne / max).
    sigma : ecart-type non contraint des noeuds de queue, supposes independants (indicatif, pas dans le critere)."""
    out = []
    for sl in (slice(0, mh), slice(mh, mh + ml)):
        ck, sk = c[sl], sg[sl]
        top = float(ck.max()) if len(ck) else 0.0
        nt = max(1, min(n_tail, len(ck)))
        tm = float(ck[-nt:].mean())
        ts = float(np.sqrt((sk[-nt:] ** 2).sum())) / nt
        ratio = tm / top if top > 0 else 0.0
        out.append((bool(top > 0 and ratio > f_tail), ratio, ts / top if top > 0 else float("nan")))
    return out


def extend_chunk(ctx, chunk, lam, a):
    """geometrie affinee, lissage lam. Pose S["Lb"] (longueurs de base), S["dL"] (rallonge par fragment, cm),
    S["notc"] (fin non contenue apres rallonge), S["tailr"], S["tails"], S["M0"] (noeuds avant rallonge),
    S["alpha2_f"] (poids de la penalite de la premiere passe, garde pendant les rallonges)."""
    for S in chunk:
        S["dL"] = [0.0, 0.0]
        S["Lb"] = S["Lr"] if S["Lr"] is not None else S["L"]
    res = evaluate(ctx, [(S, S["p"], None, S["Lb"]) for S in chunk], lam)
    for S, rr in zip(chunk, res):
        S["M0"] = (rr[4], rr[5])
        S["alpha2_f"] = rr[3]
        st = tail_state(rr[1], rr[2], rr[4], rr[5], a.n_tail, a.f_tail)
        S["notc"] = [s[0] for s in st]
        S["tailr"] = [s[1] for s in st]
        S["tails"] = [s[2] for s in st]
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
        res = evaluate(ctx, items, lam)
        for S, rr in zip(act, res):
            st = tail_state(rr[1], rr[2], rr[4], rr[5], a.n_tail, a.f_tail)
            S["notc"] = [s[0] for s in st]
            S["tailr"] = [s[1] for s in st]
            S["tails"] = [s[2] for s in st]


# ---------------------------------------------------------------------------
def add_common_args(p):
    """options communes a refine_geom.py et view_profile.py"""
    p.add_argument("--exe", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--exe_args", type=str, default="", help="cle=valeur pour profile_basis (ex. n_threads=4 table_R=5 table_tau_tol=0)")
    p.add_argument("--delta", type=float, default=0.2)
    p.add_argument("--lam_geom", type=float, default=1.0e-4, help="lissage pendant l'ajustement de la geometrie")
    p.add_argument("--free", type=str, default="x,z,a0,m0,a1,m1", help="parametres libres, parmi " + ",".join(PNAMES))
    p.add_argument("--max_iter", type=int, default=8)
    p.add_argument("--tol", type=float, default=1.0e-4, help="arret : diminution relative du residu < tol")
    p.add_argument("--lmargin", type=float, default=0.5)
    p.add_argument("--k_end", type=float, default=3.0)
    p.add_argument("--axis_dist", type=float, default=1.5)
    p.add_argument("--pad_dist", type=float, default=1.5)
    p.add_argument("--no_check", action="store_true")
    p.add_argument("--k_thr", type=float, default=10.0)
    p.add_argument("--f_max", type=float, default=0.1)
    p.add_argument("--min_half", type=int, default=3)
    p.add_argument("--sign_z", type=float, default=1.0)
    p.add_argument("--y_plane", type=float, default=0.0)
    p.add_argument("--t_off", type=float, default=None)
    p.add_argument("--lrecalc", type=int, default=1,
                   help="1 = L recalcule avec la geometrie affinee (resp. vraie) pour les profils finaux ; "
                        "0 = L de la geometrie de depart (comportement de rg4)")
    p.add_argument("--ext", type=int, default=1, help="1 = rallonge des fragments dont le profil ne retombe pas ; "
                   "0 = detection et drapeaux seulement, sans rallonge")
    p.add_argument("--f_tail", type=float, default=0.10,
                   help="fin non contenue si moyenne des n_tail derniers noeuds > f_tail x max du fragment")
    p.add_argument("--n_tail", type=int, default=3, help="nombre de noeuds de fin pour le test de confinement")
    p.add_argument("--ext_step", type=float, default=0.5, help="pas de rallonge (cm)")
    p.add_argument("--ext_max", type=float, default=3.0, help="rallonge maximale cumulee par fragment (cm)")
    p.add_argument("--win", type=float, default=0.5,
                   help="largeur (cm) de la moyenne glissante du profil fenetre, sortie au pas delta (0 = desactive)")


def parse_free(a):
    free = []
    for nm in a.free.split(","):
        nm = nm.strip()
        if nm not in PNAMES:
            sys.exit("--free : parametre inconnu %r (valeurs : %s)" % (nm, ",".join(PNAMES)))
        free.append(PNAMES.index(nm))
    return sorted(set(free))


class Env:
    """fichier .h5 ouvert + modele de reponse et de bruit (commun a refine_geom.py et view_profile.py)"""

    def __init__(self, a):
        f = h5py.File(a.h5file, "r")
        at = f["/metadata"].attrs
        self.f, self.at = f, at
        self.dt = float(at["delta_t"]); self.R = int(at["adc_ratio"])
        self.ve = float(at["ve_kg"]); self.dkg = float(at["dkg"])
        self.W = float(at["W"]); self.step = float(at["step_size_eff"])
        self.noise_on = int(at["noise_enable"]) == 1
        self.thr = float(at["read_thr_e"]) if self.noise_on else 0.0
        self.pitch = float(at["pad_size"]) + float(at["inter_pad"])
        axis = at["emission_axis"]
        self.axis = axis.decode() if isinstance(axis, bytes) else str(axis)
        self.t_off = (0.5 * float(at["dga"]) / float(at["ve_ga"]) - 0.5 * self.dt) if a.t_off is None else a.t_off
        a.t_off = self.t_off
        self.resp = Response(at, self.dt)
        hf = self.resp.hcont(self.resp.tf)
        self.m_h = float((self.resp.tf * hf).sum() / hf.sum())
        self.model = FitModel(at, check=not a.no_check)
        self.geom = pad_geometry(f)
        self.wh = Whitener(self.model, load_pad_capacitances(f), self.noise_on)
        self.N = f["/event_id"].shape[0]


def build_state(env, a, i):
    """etat d'un evenement (geometrie de depart, pads, longueurs) ; (None, raison) si ecarte"""
    f = env.f
    ev = event_pads(f, i, env.geom)
    phase = int(f["/P/phase"][i])
    wfs = pad_waveforms(ev, phase, env.R, env.dt)
    amp = np.array([w[1].max() if len(w[1]) else 0.0 for w in wfs])
    gt = geometry_true(f, i, env.axis, a.y_plane)
    if gt is None:
        return None, "verite : fragments du meme cote du plan"
    res, why = reco_split.estimate(wfs, ev["pflag"], ev["x"], ev["y"], env.resp.area, env.m_h, env.thr, a,
                                   env.pitch, env.ve, env.dkg, env.t_off)
    if res is None:
        return None, "estimation : " + why
    g0 = geometry_est(res, a.y_plane, gt["truth"])
    L = [half_length(g0, k, ev, amp, a.y_plane, env.thr, a.k_end, a.axis_dist * env.pitch, a.lmargin)
         for k in range(2)]
    if L[0] is None or L[1] is None:
        return None, "longueur : aucun pad fort dans un demi-plan"
    keep = np.zeros(len(ev["pid"]), dtype=bool)
    for k in range(2):
        u = g0["u"][k]
        uxy = np.hypot(u[0], u[1])
        d = dist_to_segment(ev["x"], ev["y"], g0["v"][0], g0["v"][1], u[0] / uxy, u[1] / uxy, L[k] * uxy)
        keep |= d <= a.pad_dist * env.pitch
    idx = np.where(keep & (ev["pnt"] > 0))[0]
    if len(idx) == 0:
        return None, "aucun pad proche de la trace"
    offs = np.concatenate([[0], np.cumsum(ev["pnt"])])
    y = np.concatenate([ev["pqn"][offs[k]:offs[k] + ev["pnt"][k]] for k in idx])
    S = {"event": i, "phase": phase, "L": L, "geo0": g0, "geot": gt,
         "p": params_from_geo(g0), "p0": params_from_geo(g0), "pt": params_from_geo(gt),
         "pids": ev["pid"][idx], "nts": ev["pnt"][idx], "y": y, "nr": len(y),
         "rows": [(int(ev["pid"][k]), int(ev["pk0"][k]), int(ev["pnt"][k])) for k in idx],
         "pq": float(ev["pq"].sum()),
         "pq_side": [float(ev["pq"][ev["y"] > a.y_plane].sum()),
                     float(ev["pq"][ev["y"] < a.y_plane].sum())],
         "px": ev["x"].copy(), "py": ev["y"].copy(), "amp": amp}
    return S, ""


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("h5file")
    add_common_args(p)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--chunk", type=int, default=25, help="evenements traites ensemble (memoire du fichier de reponse)")
    p.add_argument("--lam", type=str, default="1e-5,1e-4,1e-3", help="lissages du tableau final des profils")
    p.add_argument("--lam_plot", type=float, default=None, help="lissage des figures (defaut : lam_geom)")
    p.add_argument("--out", type=str, default="refine_geom")
    p.add_argument("--plot_n", type=int, default=6)
    a = p.parse_args()
    lams = [float(s) for s in a.lam.split(",") if s.strip()]
    lam_plot = a.lam_plot if a.lam_plot is not None else a.lam_geom
    free = parse_free(a)

    env = Env(a)
    f, W, step = env.f, env.W, env.step
    thr, pitch = env.thr, env.pitch
    wh = env.wh
    N = env.N
    i1 = min(N, a.start + a.n)
    print("[refine_geom] %s : evenements %d a %d | parametres libres : %s | lam_geom = %g"
          % (a.h5file, a.start, i1 - 1, [PNAMES[k] for k in free], a.lam_geom))

    # --- evenements : geometrie de depart, pads, longueurs ---
    states, skipped = [], {}

    def skip(why):
        skipped[why] = skipped.get(why, 0) + 1

    for i in range(a.start, i1):
        S, why = build_state(env, a, i)
        if S is None:
            skip(why)
            continue
        states.append(S)
    print("[refine_geom] %d evenements a traiter, ecartes : %s" % (len(states), skipped if skipped else "aucun"))
    if not states:
        sys.exit(1)

    server = Server(a.exe, a.cfg, a.exe_args.split())
    ctx = Ctx(a, server, wh, step, a.delta, a.y_plane, a.out)
    prof_rows, plots, geom_rows, diag_rows, chg_rows = [], {}, [], [], []
    win_rows, winprof_rows, wplots, wm_cache, flag_rows = [], [], {}, {}, []
    try:
        for c0 in range(0, len(states), a.chunk):
            chunk = states[c0:c0 + a.chunk]
            refine_chunk(ctx, chunk, a.lam_geom, free, a.max_iter, a.tol)
            print("[refine_geom] evenements %d-%d : chi2/ndl %.3f -> %.3f (mediane), iterations : mediane %d"
                  % (chunk[0]["event"], chunk[-1]["event"], np.median([S["chi2d0"] for S in chunk]),
                     np.median([S["chi2d1"] for S in chunk]), int(np.median([S["nit"] for S in chunk]))))
            # profils finaux : depart, affine, vrai
            # (lrecalc : L recalcule avec la geometrie affinee (resp. vraie), meme marge lmargin)
            for S in chunk:
                S["Lr"], S["Lt"] = None, None
                if a.lrecalc:
                    evxy = {"x": S["px"], "y": S["py"]}
                    for key, pp in (("Lr", S["p"]), ("Lt", S["pt"])):
                        gg = geo_from_params(pp, a.y_plane)
                        Lk = [half_length(gg, k, evxy, S["amp"], a.y_plane, thr, a.k_end,
                                          a.axis_dist * pitch, a.lmargin) for k in range(2)]
                        if Lk[0] is not None and Lk[1] is not None:
                            S[key] = Lk
            # test de confinement et rallonge (geometrie affinee, lissage des figures) ; meme grille ensuite pour
            # tous les lissages et les trois geometries d'un evenement
            extend_chunk(ctx, chunk, lam_plot, a)
            for lam in lams:
                is_plot_lam = (lam == lam_plot) or (lam_plot not in lams and lam == lams[0])
                items = []
                for S in chunk:
                    ext_evt = any(d > 0.0 for d in S["dL"])
                    a2 = S["alpha2_f"] if ext_evt else None
                    Lt0 = S["Lt"] if S["Lt"] is not None else S["L"]
                    items += [(S, S["p0"], a2, [S["L"][k] + S["dL"][k] for k in range(2)]),
                              (S, S["p"], a2, [S["Lb"][k] + S["dL"][k] for k in range(2)]),
                              (S, S["pt"], a2, [Lt0[k] + S["dL"][k] for k in range(2)])]
                res = evaluate(ctx, items, lam, want_G=True)
                for n, S in enumerate(chunk):
                    i = S["event"]
                    dEt = {"H": np.array(f["/H/dE"][i], dtype=float), "L": np.array(f["/L/dE"][i], dtype=float)}
                    for v, mode in enumerate(("est0", "ref", "true")):
                        r, cc, sg, al, mh, ml, G0m, Gm = res[3 * n + v]
                        cs, ss, qt, Qe = [cc[:mh], cc[mh:]], [sg[:mh], sg[mh:]], [], []
                        row = [i, v, lam, float(r[:S["nr"]] @ r[:S["nr"]]) / S["nr"]]
                        for k in range(2):
                            M = (mh, ml)[k]
                            qtr = truth_nodes(dEt[S["geot"]["truth"][k]], W, step, a.delta, M)
                            qt.append(qtr)
                            if v == 1 and lam == lams[0]:
                                dEk = dEt[S["geot"]["truth"][k]]
                                nz = np.nonzero(dEk > 0)[0]
                                Ltrue = step * (nz[-1] + 1) if len(nz) else 0.0
                                flag_rows.append([i, k, 1.0 if S["dL"][k] > 0.0 else 0.0, S["dL"][k],
                                                  1.0 if S["notc"][k] else 0.0, S["tailr"][k], S["tails"][k],
                                                  Ltrue - (S["M0"][k] - 1) * a.delta, Ltrue - (M - 1) * a.delta,
                                                  1.0 if any(d > 0.0 for d in S["dL"]) else 0.0])
                            wgt = np.full(M, a.delta); wgt[0] = 0.5 * a.delta
                            Qt = float((qtr * wgt).sum())
                            Qe.append(float((cs[k] * wgt).sum()))
                            top = qtr.max()
                            row += [Qe[k] / Qt - 1.0 if Qt > 0 else np.nan,
                                    float(np.sqrt(np.mean((cs[k] - qtr) ** 2)) / top) if top > 0 else np.nan]
                        row += [Qe[0], S["pq_side"][0], Qe[1], S["pq_side"][1],
                                float(S["dL"][0] > 0.0), float(S["notc"][0]), float(S["dL"][1] > 0.0), float(S["notc"][1])]
                        prof_rows.append(row)
                        # diagnostic biais de lissage / variance (geometrie affinee et vraie)
                        if v in (1, 2):
                            t = np.concatenate(qt)
                            Rt = np.linalg.solve(Gm, G0m @ t)              # R.verite, R = G^-1 G0
                            B = np.linalg.solve(Gm, G0m)
                            Cov = np.linalg.solve(Gm, B.T)                 # G^-1 G0 G^-1 (covariance lineaire)
                            dcov = np.diag(Cov)
                            for k in range(2):
                                sl = slice(0, mh) if k == 0 else slice(mh, mh + ml)
                                top = qt[k].max()
                                if top <= 0:
                                    continue
                                diag_rows.append([i, v, lam, k,
                                                  float(np.sqrt(np.mean((cs[k] - qt[k]) ** 2)) / top),
                                                  float(np.sqrt(np.mean((Rt[sl] - qt[k]) ** 2)) / top),
                                                  float(np.sqrt(np.mean((cs[k] - Rt[sl]) ** 2)) / top),
                                                  float(np.sqrt(np.mean(np.clip(dcov[sl], 0.0, None))) / top)])
                                # profil fenetre : moyenne glissante de largeur a.win, sortie aux noeuds
                                if a.win > 0.0:
                                    M = (mh, ml)[k]
                                    if M not in wm_cache:
                                        wm_cache[M] = window_matrix(M, a.delta, a.win)
                                    Wm, wfull = wm_cache[M]
                                    tw = window_truth(dEt[S["geot"]["truth"][k]], W, step, a.delta, M, a.win)
                                    qw = Wm @ cs[k]
                                    sw = np.sqrt(np.clip(np.diag(Wm @ Cov[sl, sl] @ Wm.T), 0.0, None))
                                    msk = wfull if wfull.any() else np.ones(M, dtype=bool)
                                    topw = tw.max()
                                    if topw > 0:
                                        win_rows.append([i, v, lam, k,
                                                         float(np.sqrt(np.mean((qw[msk] - tw[msk]) ** 2)) / topw),
                                                         float(np.sqrt(np.mean(sw[msk] ** 2)) / topw),
                                                         float(np.sqrt(np.mean((qt[k][msk] - tw[msk]) ** 2)) / topw)])
                                    if v == 1 and is_plot_lam:
                                        for jn in range(M):
                                            winprof_rows.append([i, k, jn * a.delta, qw[jn], sw[jn], tw[jn],
                                                                 qt[k][jn], 1.0 if wfull[jn] else 0.0,
                                                                 1.0 if jn >= S["M0"][k] else 0.0,
                                                                 1.0 if S["dL"][k] > 0.0 else 0.0,
                                                                 1.0 if S["notc"][k] else 0.0,
                                                                 1.0 if any(d > 0.0 for d in S["dL"]) else 0.0])
                                        wplots[(i, k)] = (qw, sw, tw, qt[k], S["M0"][k], S["dL"][k] > 0.0, S["notc"][k])
                        if v == 1 and is_plot_lam:
                            chg_rows.append([S["pq_side"][0], Qe[0], S["pq_side"][1], Qe[1]])
                        if is_plot_lam:
                            plots[(i, mode)] = (cs, ss, qt, S["geot"]["truth"])
            for S in chunk:
                gt_, g0_, g1_ = S["geot"], geo_from_params(S["p0"], a.y_plane), geo_from_params(S["p"], a.y_plane)
                geom_rows.append([S["event"], S["nit"], S["chi2d0"], S["chi2d1"]]
                                 + [angle_deg(g0_["u"][k], gt_["u"][k]) for k in range(2)]
                                 + [angle_deg(g1_["u"][k], gt_["u"][k]) for k in range(2)]
                                 + [g0_["v"][0] - gt_["v"][0], g0_["v"][2] - gt_["v"][2],
                                    g1_["v"][0] - gt_["v"][0], g1_["v"][2] - gt_["v"][2]])
    finally:
        server.close()

    gcols = ["event", "n_iter", "chi2d_start", "chi2d_ref", "ang0_start", "ang1_start", "ang0_ref", "ang1_ref",
             "dx_start", "dz_start", "dx_ref", "dz_ref"]
    pcols = ["event", "geom(0=depart,1=affine,2=vrai)", "lam", "chi2_dof", "dQ_frag0", "rms_frag0", "dQ_frag1", "rms_frag1",
             "Qest_frag0", "Qpq_frag0", "Qest_frag1", "Qpq_frag1",
             "rallonge_frag0", "fin_non_contenue_frag0", "rallonge_frag1", "fin_non_contenue_frag1"]
    dcols = ["event", "geom(1=affine,2=vrai)", "lam", "frag", "rms_total", "rms_biais_lissage", "rms_stat", "rms_stat_prevu"]
    G = np.array(geom_rows, dtype=float)
    P = np.array(prof_rows, dtype=float)
    DG = np.array(diag_rows, dtype=float)
    np.savetxt(a.out + "_geom.csv", G, delimiter=",", header=",".join(gcols), comments="")
    np.savetxt(a.out + "_prof.csv", P, delimiter=",", header=",".join(pcols), comments="")
    np.savetxt(a.out + "_diag.csv", DG, delimiter=",", header=",".join(dcols), comments="")
    if flag_rows:
        FL = np.array(flag_rows, dtype=float)
        flcols = ["event", "frag", "rallonge", "ext_cm", "fin_non_contenue", "queue_sur_max", "sigma_queue_sur_max",
                  "deficit_initial_cm", "deficit_final_cm", "evt_rallonge"]
        np.savetxt(a.out + "_flags.csv", FL, delimiter=",", header=",".join(flcols), comments="")
    if win_rows:
        WS = np.array(win_rows, dtype=float)
        wscols = ["event", "geom(1=affine,2=vrai)", "lam", "frag", "rms_fenetre", "bruit_prevu", "rms_tnoeuds_tfenetre"]
        np.savetxt(a.out + "_wsum.csv", WS, delimiter=",", header=",".join(wscols), comments="")
    if winprof_rows:
        WP = np.array(winprof_rows, dtype=float)
        wpcols = ["event", "frag", "l_cm", "q_fen_est", "sigma_prevu", "q_fen_vrai", "q_noeud_vrai", "fenetre_entiere",
                  "zone_rallongee", "rallonge", "fin_non_contenue", "evt_rallonge"]
        np.savetxt(a.out + "_win.csv", WP, delimiter=",", header=",".join(wpcols), comments="")

    print("\n=== Geometrie : %d evenements (ecartes : %s) ===" % (len(G), skipped if skipped else "aucun"))
    print("chi2/ndl (donnees, lam_geom = %g) : depart %.3f -> affine %.3f (mediane) ; iterations : mediane %d, max %d"
          % (a.lam_geom, np.median(G[:, 2]), np.median(G[:, 3]), int(np.median(G[:, 1])), int(G[:, 1].max())))
    print("                                      mediane     68 %      95 %")
    for name, v0, v1, unit in (("angle 3D (2 fragments)", np.concatenate([G[:, 4], G[:, 5]]), np.concatenate([G[:, 6], G[:, 7]]), "deg"),
                               ("x_v", G[:, 8], G[:, 10], "cm"), ("z_v", G[:, 9], G[:, 11], "cm")):
        print("  %-22s depart   %9.3f %9.3f %9.3f  %s" % ((name,) + pct(v0) + (unit,)))
        print("  %-22s affine   %9.3f %9.3f %9.3f  %s" % ((name,) + pct(v1) + (unit,)))

    print("\n=== Profils (erreur rms / maximum du profil vrai ; |dQ| = erreur relative sur la charge totale) ===")
    print("geometrie  lam        chi2/ndl   rms : med   68%    95%    >10%    |dQ| : med   95%")
    for v, name in ((0, "depart"), (1, "affinee"), (2, "vraie")):
        for lam in lams:
            sel = (P[:, 1] == v) & (P[:, 2] == lam)
            if not sel.any():
                continue
            r = P[sel]
            rms = np.concatenate([r[:, 5], r[:, 7]])
            dq = np.concatenate([r[:, 4], r[:, 6]])
            m, p68, p95 = pct(rms)
            q = pct(dq)
            fr = float(np.mean(rms[np.isfinite(rms)] > 0.10))
            print("%-9s  %-9g  %8.3f   %13.3f %6.3f %6.3f  %5.1f %%   %12.4f %6.4f"
                  % (name, lam, np.median(r[:, 3]), m, p68, p95, 100.0 * fr, q[0], q[2]))

    if flag_rows:
        FL = np.array(flag_rows, dtype=float)
        nfr = len(FL)
        isr, notc, di, dfin = FL[:, 2] > 0, FL[:, 4] > 0, FL[:, 7], FL[:, 8]
        DEFT = a.delta                                    # deficit significatif : au moins un noeud
        print("\n=== Fin de profil : confinement et rallonges (geometrie affinee, lam = %g) ===" % lam_plot)
        print("critere : moyenne des %d derniers noeuds > %.0f %% du max du fragment => fin non contenue ;" % (a.n_tail, 100.0 * a.f_tail))
        print("rallonge par pas de %.2f cm jusqu'a %.2f cm (--ext %d). Drapeaux par fragment : %s_flags.csv ;"
              % (a.ext_step, a.ext_max, a.ext, a.out))
        print("par noeud : colonne zone_rallongee de %s_win.csv." % a.out)
        print("fragments rallonges : %d / %d" % (int(isr.sum()), nfr))
        for r in FL[isr]:
            print("  rallonge : evt %d frag %d  +%.2f cm  (queue/max %.3f, deficit vs verite %+.2f -> %+.2f cm)"
                  % (int(r[0]), int(r[1]), r[3], r[5], r[7], r[8]))
        if notc.any():
            print("ATTENTION : fin de profil NON CONTENUE (apres rallonge) pour %d fragment(s) : le range n'est pas mesurable :" % int(notc.sum()))
            for r in FL[notc]:
                print("  ATTENTION evt %d frag %d : queue/max = %.3f (sigma %.3f), rallonge %.2f cm"
                      % (int(r[0]), int(r[1]), r[5], r[6], r[3]))
        else:
            print("fin non contenue apres rallonge : aucun fragment")
        print("validation par la verite (deficit initial = L_vrai(dE > 0) - fin de grille de depart ; significatif si > %.2f cm) :" % DEFT)
        print("  rallonge et deficit initial significatif (detection juste)     : %d" % int((isr & (di > DEFT)).sum()))
        print("  rallonge sans deficit initial significatif (fausse alerte)     : %d" % int((isr & (di <= DEFT)).sum()))
        print("  pas de rallonge malgre un deficit initial significatif (rate)  : %d" % int((~isr & (di > DEFT)).sum()))
        print("  deficit apres rallonge : max %+.3f cm ; fragments encore en deficit significatif : %d"
              % (float(dfin.max()), int((dfin > DEFT).sum())))

    print("\n=== Diagnostic biais de lissage / variance (rms / max du profil vrai, medianes) ===")
    print("R = (G0+P)^-1 G0 (estimateur lineaire) ; biais de lissage : R.verite - verite ; stat (mesure) :")
    print("ajuste - R.verite (contient l'effet de c >= 0) ; stat (prevu) : sqrt(diag((G0+P)^-1 G0 (G0+P)^-1)).")
    print("geometrie  lam        rms total   biais lissage   stat (mesure)   stat (prevu)")
    for v, name in ((1, "affinee"), (2, "vraie")):
        for lam in lams:
            sel = (DG[:, 1] == v) & (DG[:, 2] == lam)
            if not sel.any():
                continue
            r = DG[sel]
            print("%-9s  %-9g  %9.4f   %13.4f   %13.4f   %12.4f"
                  % (name, lam, np.median(r[:, 4]), np.median(r[:, 5]), np.median(r[:, 6]), np.median(r[:, 7])))

    if win_rows:
        WS = np.array(win_rows, dtype=float)
        print("\n=== Profils fenetres : moyenne glissante w = %.2f cm, sortie au pas delta = %.2f cm ===" % (a.win, a.delta))
        print("rms de (estime - vrai fenetre) / max du vrai fenetre, noeuds a fenetre entiere ; vrai fenetre =")
        print("fenetre appliquee aux voxels de /H/dE,/L/dE ; bruit prevu = sqrt(diag(S Cov S^T)) (sans c >= 0).")
        print("geometrie  lam        rms : med    68%     95%     bruit prevu (med)")
        for v, name in ((1, "affinee"), (2, "vraie")):
            for lam in lams:
                sel = (WS[:, 1] == v) & (WS[:, 2] == lam)
                if not sel.any():
                    continue
                r = WS[sel]
                m, p68, p95 = pct(r[:, 4])
                print("%-9s  %-9g  %9.4f %7.4f %7.4f   %13.4f" % (name, lam, m, p68, p95, np.median(r[:, 5])))
        sel = (WS[:, 1] == 1) & (WS[:, 2] == lams[0])
        if sel.any():
            m, p68, p95 = pct(WS[sel][:, 6])
            print("structure vraie entre les noeuds et la fenetre, rms(t_noeuds - t_fenetre)/max :")
            print("  mediane %.4f  68%% %.4f  95%% %.4f  (fluctuations de dE seul : straggling simule, sans Fano ni bruit)"
                  % (m, p68, p95))

    if chg_rows:
        C = np.array(chg_rows, dtype=float)
        qp = [C[:, 0], C[:, 2], C[:, 0] + C[:, 2]]
        qe = [C[:, 1], C[:, 3], C[:, 1] + C[:, 3]]
        print("\n=== Charge : ajustement vs charge arrivee sur les pads (geometrie affinee, lam = %g) ===" % lam_plot)
        print("Qpq (par fragment) = somme de /P/pq sur les pads ecrits du demi-plan du fragment.")
        print("Reserves : pads sous le seuil absents de pq ; pres du vertex la diffusion melange les demi-plans")
        print("(le total n'a pas ce probleme).")
        print("(Qest-Qpq)/Qpq :          mediane     68 %      95 %")
        for k, name in ((0, "fragment 0"), (1, "fragment 1"), (2, "total")):
            d = qe[k] / qp[k] - 1.0
            sm = float(np.median(d))
            print("  %-22s %+9.4f %9.4f %9.4f" % ((name, sm) + pct(d)[1:]))
    print("\nTableaux : %s_geom.csv, %s_prof.csv, %s_diag.csv, %s_flags.csv%s" %
          (a.out, a.out, a.out, a.out, (", %s_wsum.csv, %s_win.csv" % (a.out, a.out)) if win_rows else ""))
    print("cache du blanchiment : %s" % wh.cache_info())

    if chg_rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        C = np.array(chg_rows, dtype=float)
        pts = [(C[:, 0], C[:, 1], "C0", "o", "fragment 0"),
               (C[:, 2], C[:, 3], "C1", "s", "fragment 1"),
               (C[:, 0] + C[:, 2], C[:, 1] + C[:, 3], "k", "^", "total")]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for qp_, qe_, col, mk, lab in pts:
            ax[0].plot(qp_, qe_, mk, color=col, ms=4, mfc="none", label=lab)
            ax[1].plot(qp_, 100.0 * (qe_ / qp_ - 1.0), mk, color=col, ms=4, mfc="none", label=lab)
        lim = [0.0, 1.05 * max(float((C[:, 0] + C[:, 2]).max()), float((C[:, 1] + C[:, 3]).max()))]
        ax[0].plot(lim, lim, "k--", lw=0.8)
        ax[0].set_xlim(lim); ax[0].set_ylim(lim)
        ax[0].set_xlabel("Qpq (e-) : somme de /P/pq du demi-plan"); ax[0].set_ylabel("Qest (e-) : integrale du profil ajuste")
        ax[0].set_title("charge par fragment (geometrie affinee, lam=%g)" % lam_plot)
        ax[1].axhline(0.0, color="k", lw=0.8, ls="--")
        ax[1].set_xlabel("Qpq (e-)"); ax[1].set_ylabel("(Qest - Qpq) / Qpq (%)")
        ax[1].set_title("ecart relatif")
        ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(a.out + "_charge.png", dpi=110)
        plt.close(fig)
        print("figure : %s_charge.png" % a.out)

    if a.plot_n > 0 and plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {"est0": "C0", "ref": "C2", "true": "C3"}
        labels = {"est0": "geometrie estimee (depart)", "ref": "geometrie affinee", "true": "geometrie vraie"}
        events = sorted(set(k[0] for k in plots))[:a.plot_n]
        for i in events:
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            qbest = [None, None]
            names = None
            for mode in ("est0", "ref", "true"):
                if (i, mode) not in plots:
                    continue
                cs, ss, qt, names = plots[(i, mode)]
                for k in range(2):
                    l = np.arange(len(cs[k])) * a.delta
                    ax[k].fill_between(l, cs[k] - ss[k], cs[k] + ss[k], color=colors[mode], alpha=0.12)
                    ax[k].plot(l, cs[k], "o-", color=colors[mode], ms=3, label=labels[mode])
                    if qbest[k] is None or len(qt[k]) > len(qbest[k]):
                        qbest[k] = qt[k]
            for k in range(2):
                ax[k].plot(np.arange(len(qbest[k])) * a.delta, qbest[k], "k.-", ms=3, label="verite (moyenne sur chapeaux)")
                ax[k].set_xlabel("l (cm)"); ax[k].set_ylabel("e-/cm")
                ax[k].set_title("fragment %d (%s), evt %d, lam=%g" % (k, names[k], i, lam_plot))
            ax[0].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig("%s_evt%d.png" % (a.out, i), dpi=110)
            plt.close(fig)
        print("figures : %s_evt*.png" % a.out)
        # profils fenetres (geometrie affinee, lam = lam_plot)
        wevents = sorted(set(k[0] for k in wplots))[:a.plot_n]
        for i in wevents:
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            for k in range(2):
                if (i, k) not in wplots:
                    continue
                qw, sw, tw, tn, m0k, rallk, notck = wplots[(i, k)]
                l = np.arange(len(qw)) * a.delta
                if rallk:
                    ax[k].axvline((m0k - 1) * a.delta, color="C1", ls="--", lw=1.0, label="fin de grille initiale")
                ax[k].plot(l, tn, ".", color="0.6", ms=4, label="verite aux noeuds (non fenetree)")
                ax[k].fill_between(l, qw - sw, qw + sw, color="C2", alpha=0.2)
                ax[k].plot(l, qw, "o-", color="C2", ms=3, label="ajuste, fenetre %.1f mm" % (10 * a.win))
                ax[k].plot(l, tw, "k.-", ms=3, label="verite, meme fenetre")
                ax[k].set_xlabel("l (cm)"); ax[k].set_ylabel("e-/cm")
                tit = "fragment %d, evt %d, lam=%g (geometrie affinee)" % (k, i, lam_plot)
                if rallk:
                    tit += "\nGRILLE RALLONGEE"
                if notck:
                    tit += "\nATTENTION : FIN NON CONTENUE"
                ax[k].set_title(tit, color="C3" if (rallk or notck) else "k", fontsize=9)
            ax[0].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig("%s_wevt%d.png" % (a.out, i), dpi=110)
            plt.close(fig)
        if wevents:
            print("figures : %s_wevt*.png (bande = bruit prevu)" % a.out)


if __name__ == "__main__":
    main()
