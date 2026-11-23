#!/usr/bin/env python3
"""
fit_profile.py : etape 1. Profil de charge q(l) (electrons par cm) de chaque fragment, ajuste directement
sur les echantillons des pads, par moindres carres lineaires ponderes par la covariance du bruit.

Chaine (voir profile_basis.cpp) :
  1. direction et vertex : reco_split.estimate (pads forts, temps des pads) ; option --mode true : valeurs vraies ;
  2. longueur modelisee de chaque demi-droite : pad fort le plus lointain + marge (--lmargin) ;
  3. pads utilises : pads ecrits a moins de --pad_dist pitchs de la trace (tous leurs echantillons ecrits) ;
  4. ./profile_basis calcule, pour chaque chapeau de pas --delta, la reponse exacte (induction, preampli, CAN) ;
  5. solution de  min  || C^(-1/2) (A c - y) ||^2 + penalite de courbure (ordre 2),  c >= 0 ;
  6. comparaison avec la verite /H/dE, /L/dE (moyennee sur les memes chapeaux).

Convention : fragment 0 = demi-plan y > y_plane, fragment 1 = demi-plan y < y_plane.
Penalite : lam * alpha^2 * (1 cm)^4 * somme (c[m-1] - 2 c[m] + c[m+1])^2 / delta^3  (approche l'integrale de q''^2),
alpha^2 = mediane des ||colonne de A ponderee||^2 / delta^2 : lam n'a pas d'unite et le resultat ne depend pas de delta.

Usage :
  python3 fit_profile.py sortie.h5 --exe ./build/profile_basis --cfg drift.cfg --n 20 --lam 1e-5,1e-4,1e-3 --mode both

Necessite numpy, scipy, h5py, matplotlib ; t2f_io.py, view_fit.py, view_pads.py, reco_dir.py, reco_split.py dans le meme dossier.
"""

import argparse
import os
import struct
import subprocess
import sys

import h5py
import numpy as np
from scipy.optimize import nnls

from t2f_io import pad_geometry, read_vertex, event_pads, load_pad_capacitances
from view_fit import Response, FitModel
from reco_dir import true_direction, pad_waveforms
import reco_split


# ---------------------------------------------------------------------------
# geometrie d'un evenement : vertex et deux directions (fragment 0 : y > plan, fragment 1 : y < plan)
# ---------------------------------------------------------------------------
def geometry_true(f, i, axis, y_plane):
    uH = true_direction(f, "H", i, axis)
    uL = true_direction(f, "L", i, axis)
    if uH[1] * uL[1] >= 0.0 or uH[1] == 0.0 or uL[1] == 0.0:
        return None
    vx, vy, vz = read_vertex(f, i)
    if uH[1] > 0.0:
        return {"v": np.array([vx, vy, vz]), "u": [uH, uL], "truth": ["H", "L"]}
    return {"v": np.array([vx, vy, vz]), "u": [uL, uH], "truth": ["L", "H"]}


def geometry_est(res, y_plane, truth_names):
    v = np.array([res["xv"], y_plane, res["zv"]])
    return {"v": v, "u": [res["u_p"], res["u_m"]], "truth": truth_names}


def dist_to_segment(px, py, x0, y0, ex, ey, L):
    """distance de points (px, py) au segment x0 + s e, s dans [0, L] (e unitaire dans le plan)"""
    s = (px - x0) * ex + (py - y0) * ey
    s = np.clip(s, 0.0, L)
    return np.hypot(px - (x0 + s * ex), py - (y0 + s * ey))


def half_length(geo, k, ev, amp, y_plane, thr, k_end, axis_dist, lmargin, pitch=0.0):
    """longueur l (3D) de la demi-droite k : pad fort (amp >= k_end * seuil) le plus lointain, a moins de
    axis_dist de l'axe, + lmargin. None si aucun pad.
    pitch > 0 : marge adaptative lmargin + 0.5 * pitch / uxy (plafonnee a uxy = 0.15), car l'etendue dans
    le plan est quantifiee au pitch et l'erreur est amplifiee par 1/uxy a incidence raide."""
    u = geo["u"][k]
    uxy = np.hypot(u[0], u[1])
    if uxy < 1.0e-6:
        return None
    ex, ey = u[0] / uxy, u[1] / uxy
    sgn = 1.0 if k == 0 else -1.0
    x0, y0 = geo["v"][0], geo["v"][1]
    s = (ev["x"] - x0) * ex + (ev["y"] - y0) * ey
    dperp = np.abs(-(ev["x"] - x0) * ey + (ev["y"] - y0) * ex)
    ok = (amp >= k_end * thr) & (sgn * (ev["y"] - y_plane) > 0.0) & (dperp <= axis_dist) & (s > 0.0)
    if not ok.any():
        return None
    marge = lmargin
    if pitch > 0.0:
        marge += 0.5 * pitch / max(uxy, 0.15)
    return float(s[ok].max() / uxy + marge)


# ---------------------------------------------------------------------------
# requete / reponse du pilote C++
# ---------------------------------------------------------------------------
def write_request(path, step_eff, delta, recs):
    with open(path, "w") as g:
        g.write("%.10g %.10g %d\n" % (step_eff, delta, len(recs)))
        for r in recs:
            g.write("%d %d\n" % (r["tag"], r["phase"]))
            g.write("%.10g %.10g %.10g\n" % tuple(r["v"]))
            for k in range(2):
                g.write("%.12g %.12g %.12g %.10g\n" % (r["u"][k][0], r["u"][k][1], r["u"][k][2], r["L"][k]))
            g.write("%d\n" % len(r["pads"]))
            for (pid, j0, nt) in r["pads"]:
                g.write("%d %d %d\n" % (pid, j0, nt))


def read_response_iter(path):
    """lit le fichier de reponse evenement par evenement : produit (tag, mh, ml, A)"""
    with open(path, "rb") as g:
        while True:
            h = g.read(16)
            if len(h) < 16:
                return
            tag, mh, ml, nr = struct.unpack("<4i", h)
            n = (mh + ml) * nr
            A = np.frombuffer(g.read(8 * n), dtype="<f8", count=n).reshape(mh + ml, nr)
            yield tag, mh, ml, A


def read_response(path):
    out = {}
    with open(path, "rb") as g:
        buf = g.read()
    pos = 0
    while pos < len(buf):
        tag, mh, ml, nr = struct.unpack_from("<4i", buf, pos)
        pos += 16
        n = (mh + ml) * nr
        A = np.frombuffer(buf, dtype="<f8", count=n, offset=pos).reshape(mh + ml, nr).copy()
        pos += 8 * n
        out[tag] = (mh, ml, A)
    return out


# ---------------------------------------------------------------------------
# ponderation par la covariance du bruit
# ---------------------------------------------------------------------------
class Whitener:
    def __init__(self, model, caps, noise_on):
        self.model = model
        self.caps = caps
        self.noise_on = noise_on
        self._chol = {}

    def chol(self, n, ser_scale):
        key = (n, round(ser_scale, 5))
        if key not in self._chol:
            self._chol[key] = np.linalg.cholesky(self.model.cov(n, ser_scale))
        return self._chol[key]

    def ser_scale(self, pid):
        if self.caps is None or int(pid) not in self.caps:
            return 1.0
        return self.model.ser_scale_pad(self.caps[int(pid)][1])

    def apply(self, pids, nts, A, y):
        """A : (nb, nr), y : (nr) ; renvoie (Aw, yw) blanchis pad par pad (blocs d'echantillons consecutifs)"""
        if not self.noise_on:
            return A.copy(), y.copy()
        Aw = np.empty_like(A)
        yw = np.empty_like(y)
        off = 0
        for pid, n in zip(pids, nts):
            L = self.chol(int(n), self.ser_scale(pid))
            Aw[:, off:off + n] = np.linalg.solve(L, A[:, off:off + n].T).T
            yw[off:off + n] = np.linalg.solve(L, y[off:off + n])
            off += n
        return Aw, yw


# ---------------------------------------------------------------------------
# ajustement
# ---------------------------------------------------------------------------
def second_diff(M):
    """matrice (M-2, M) des differences secondes"""
    if M < 3:
        return np.zeros((0, M))
    D = np.zeros((M - 2, M))
    for m in range(M - 2):
        D[m, m], D[m, m + 1], D[m, m + 2] = 1.0, -2.0, 1.0
    return D


def solve_profile_r(Aw, yw, Mh, Ml, delta, lam, alpha2=None, want_G=False, endw=None):
    """min ||Aw^T c - yw||^2 + penalite, c >= 0. Aw : (nb, nr).
    Renvoie c, sigma (non contraint), residu complet r = [Aw^T c - yw ; racine(penalite) D c], alpha2 utilise.
    alpha2 : si None, calcule (mediane des ||colonne||^2 / delta^2) ; le fixer rend deux evaluations comparables.
    want_G : renvoie en plus G0 = Aw Aw^T et G = G0 + penalite (pour la matrice de resolution).
    endw : None, ou (fac, js_h, js_l) : le poids de la penalite est multiplie par fac pour les differences
    secondes dont le noeud central est d'indice >= js (js_h pour le fragment 0, js_l pour le fragment 1) ;
    fac < 1 relache le lissage en fin de trace."""
    nb, nr = Aw.shape
    G0 = Aw @ Aw.T
    rhs = Aw @ yw
    if alpha2 is None:
        cn = np.diag(G0)
        good = cn > 1.0e-12 * max(cn.max(), 1.0e-300)
        alpha2 = float(np.median(cn[good])) / delta ** 2 if good.any() else 1.0
    D = np.zeros((max(Mh - 2, 0) + max(Ml - 2, 0), nb))
    r = 0
    for kf, (M0, M) in enumerate(((0, Mh), (Mh, Ml))):
        d = second_diff(M)
        if endw is not None and d.shape[0] > 0:
            # la ligne m de d est centree sur le noeud m + 1
            d = d * np.where(np.arange(d.shape[0]) + 1 >= endw[1 + kf], np.sqrt(endw[0]), 1.0)[:, None]
        D[r:r + d.shape[0], M0:M0 + M] = d
        r += d.shape[0]
    wreg = lam * alpha2 / delta ** 3
    G = G0 + wreg * (D.T @ D) + 1.0e-10 * np.trace(G0) / nb * np.eye(nb)
    try:
        Lc = np.linalg.cholesky(G)
    except np.linalg.LinAlgError:
        G = G + 1.0e-6 * np.trace(G0) / nb * np.eye(nb)
        Lc = np.linalg.cholesky(G)
    # min c^T G c - 2 rhs^T c  <=>  min || Lc^T c - Lc^-1 rhs ||^2
    b = np.linalg.solve(Lc, rhs)
    c, _ = nnls(Lc.T, b, maxiter=50 * nb)
    sig = np.sqrt(np.diag(np.linalg.inv(G)))
    resid = np.concatenate([Aw.T @ c - yw, np.sqrt(wreg) * (D @ c)])
    if want_G:
        return c, sig, resid, alpha2, G0, G
    return c, sig, resid, alpha2


def solve_profile(Aw, yw, Mh, Ml, delta, lam):
    """comme solve_profile_r ; renvoie c, sigma, chi2 des donnees (sans la penalite), nombre d'echantillons"""
    c, sig, resid, _ = solve_profile_r(Aw, yw, Mh, Ml, delta, lam)
    nr = Aw.shape[1]
    d = resid[:nr]
    return c, sig, float(d @ d), nr


def truth_nodes(dE, W, step, delta, M):
    """valeur moyenne (e-/cm), ponderee par chaque chapeau, du profil vrai ; poids integral des chapeaux"""
    q = np.asarray(dE, dtype=float) / W / step
    s = step * (np.arange(len(q)) + 0.5)
    out = np.zeros(M)
    for m in range(M):
        w = np.clip(1.0 - np.abs(s - m * delta) / delta, 0.0, None)
        norm = (0.5 if m == 0 else 1.0) * delta
        out[m] = (w * q).sum() * step / norm
    return out


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("h5file")
    p.add_argument("--exe", required=True, help="executable profile_basis")
    p.add_argument("--cfg", required=True, help="le drift.cfg utilise pour produire le .h5")
    p.add_argument("--exe_args", type=str, default="", help="arguments cle=valeur supplementaires pour profile_basis (ex. n_threads=4)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--mode", choices=["est", "true", "both"], default="est",
                   help="direction/vertex estimes (est), vrais (true) ou les deux (both)")
    p.add_argument("--delta", type=float, default=0.2, help="pas des chapeaux (cm)")
    p.add_argument("--lam", type=str, default="1e-5,1e-4,1e-3", help="liste de coefficients de lissage (sans unite)")
    p.add_argument("--lmargin", type=float, default=0.5, help="marge au-dela du pad fort le plus lointain (cm)")
    p.add_argument("--k_end", type=float, default=3.0, help="pad 'fort' pour la longueur : amplitude >= k_end * seuil de lecture")
    p.add_argument("--axis_dist", type=float, default=1.5, help="distance max a l'axe (en pitchs) pour la longueur")
    p.add_argument("--pad_dist", type=float, default=1.5, help="pads utilises : distance a la trace <= pad_dist pitchs")
    p.add_argument("--no_check", action="store_true")
    # options de reco_split.estimate
    p.add_argument("--k_thr", type=float, default=10.0)
    p.add_argument("--f_max", type=float, default=0.1)
    p.add_argument("--n_nb", type=int, default=2, help="pad utilise pour l'estimation de depart : au moins n_nb voisins (a moins de 1.5 pitch) eux-memes forts ; 0 = sans condition")
    p.add_argument("--min_half", type=int, default=3)
    p.add_argument("--sign_z", type=float, default=1.0)
    p.add_argument("--y_plane", type=float, default=0.0)
    p.add_argument("--t_off", type=float, default=None)
    p.add_argument("--out", type=str, default="profile_fit")
    p.add_argument("--plot_n", type=int, default=6, help="nombre d'evenements traces")
    a = p.parse_args()
    lams = [float(s) for s in a.lam.split(",") if s.strip()]

    f = h5py.File(a.h5file, "r")
    at = f["/metadata"].attrs
    dt = float(at["delta_t"]); R = int(at["adc_ratio"]); ve = float(at["ve_kg"]); dkg = float(at["dkg"])
    W = float(at["W"]); step = float(at["step_size_eff"])
    noise_on = int(at["noise_enable"]) == 1
    thr = float(at["read_thr_e"]) if noise_on else 0.0
    pitch = float(at["pad_size"]) + float(at["inter_pad"])
    axis = at["emission_axis"]
    axis = axis.decode() if isinstance(axis, bytes) else str(axis)
    t_off = (0.5 * float(at["dga"]) / float(at["ve_ga"]) - 0.5 * dt) if a.t_off is None else a.t_off
    a.t_off = t_off
    resp = Response(at, dt)
    hf = resp.hcont(resp.tf)
    m_h = float((resp.tf * hf).sum() / hf.sum())
    model = FitModel(at, check=not a.no_check)
    geom = pad_geometry(f)
    caps = load_pad_capacitances(f)
    wh = Whitener(model, caps, noise_on)
    N = f["/event_id"].shape[0]
    i1 = min(N, a.start + a.n)
    modes = ["est", "true"] if a.mode == "both" else [a.mode]
    print("[fit_profile] %s : evenements %d a %d | reponse : %s | delta = %.3f cm | lam = %s"
          % (a.h5file, a.start, i1 - 1, resp.descr, a.delta, lams))

    # --- preparation des evenements ---
    recs, info, skipped = [], {}, {}

    def skip(why):
        skipped[why] = skipped.get(why, 0) + 1

    for i in range(a.start, i1):
        ev = event_pads(f, i, geom)
        phase = int(f["/P/phase"][i])
        wfs = pad_waveforms(ev, phase, R, dt)
        amp = np.array([w[1].max() if len(w[1]) else 0.0 for w in wfs])
        gt = geometry_true(f, i, axis, a.y_plane)
        if gt is None:
            skip("verite : fragments du meme cote du plan")
            continue
        geos = {}
        if "true" in modes:
            geos["true"] = gt
        if "est" in modes:
            res, why = reco_split.estimate(wfs, ev["x"], ev["y"], resp.area, m_h, thr, a, pitch, ve, dkg, t_off)
            if res is None:
                skip("estimation : " + why)
                continue
            geos["est"] = geometry_est(res, a.y_plane, gt["truth"])
        for mode in modes:
            geo = geos[mode]
            L = [half_length(geo, k, ev, amp, a.y_plane, thr, a.k_end, a.axis_dist * pitch, a.lmargin) for k in range(2)]
            if L[0] is None or L[1] is None:
                skip("longueur : aucun pad fort dans un demi-plan (%s)" % mode)
                continue
            # pads proches de la trace
            keep = np.zeros(len(ev["pid"]), dtype=bool)
            for k in range(2):
                u = geo["u"][k]
                uxy = np.hypot(u[0], u[1])
                d = dist_to_segment(ev["x"], ev["y"], geo["v"][0], geo["v"][1], u[0] / uxy, u[1] / uxy, L[k] * uxy)
                keep |= d <= a.pad_dist * pitch
            idx = np.where(keep & (ev["pnt"] > 0))[0]
            if len(idx) == 0:
                skip("aucun pad proche de la trace (%s)" % mode)
                continue
            tag = 2 * i + (1 if mode == "true" else 0)
            recs.append({"tag": tag, "phase": phase, "v": geo["v"], "u": geo["u"], "L": L,
                         "pads": [(int(ev["pid"][k]), int(ev["pk0"][k]), int(ev["pnt"][k])) for k in idx]})
            info[tag] = {"event": i, "mode": mode, "idx": idx, "ev": ev, "geo": geo, "L": L}
    if not recs:
        print("[fit_profile] aucun evenement exploitable :", skipped)
        sys.exit(1)

    # --- pilote C++ ---
    req, rsp = a.out + "_request.txt", a.out + "_response.bin"
    write_request(req, step, a.delta, recs)
    cmd = [a.exe, a.cfg, req, rsp] + a.exe_args.split()
    print("[fit_profile] " + " ".join(cmd))
    rc = subprocess.run(cmd)
    if rc.returncode != 0:
        sys.exit("[fit_profile] profile_basis a echoue (code %d)" % rc.returncode)
    resp_all = read_response(rsp)

    # --- ajustement et comparaison ---
    rows = []
    store = {}
    for rec in recs:
        tag = rec["tag"]
        d = info[tag]
        mh, ml, A = resp_all[tag]
        ev = d["ev"]
        idx = d["idx"]
        offs = np.concatenate([[0], np.cumsum(ev["pnt"])])
        y = np.concatenate([ev["pqn"][offs[k]:offs[k] + ev["pnt"][k]] for k in idx])
        pids = ev["pid"][idx]
        nts = ev["pnt"][idx]
        if A.shape[1] != len(y):
            sys.exit("[fit_profile] evenement %d : %d colonnes dans la reponse, %d echantillons de donnees"
                     % (d["event"], A.shape[1], len(y)))
        Aw, yw = wh.apply(pids, nts, A, y)
        i = d["event"]
        dEtrue = {"H": np.array(f["/H/dE"][i], dtype=float), "L": np.array(f["/L/dE"][i], dtype=float)}
        qtruth = []
        for k in range(2):
            M = (mh, ml)[k]
            qtruth.append(truth_nodes(dEtrue[d["geo"]["truth"][k]], W, step, a.delta, M))
        qpq = float(ev["pq"].sum())
        qdE = (dEtrue["H"].sum() + dEtrue["L"].sum()) / W
        for lam in lams:
            c, sig, chi2, nr = solve_profile(Aw, yw, mh, ml, a.delta, lam)
            cs = [c[:mh], c[mh:]]
            ss = [sig[:mh], sig[mh:]]
            row = [i, 1 if d["mode"] == "true" else 0, lam, nr, chi2 / nr, qpq / qdE if qdE > 0 else np.nan]
            for k in range(2):
                M = (mh, ml)[k]
                w = np.full(M, a.delta); w[0] = 0.5 * a.delta
                Qf = float((cs[k] * w).sum())
                Qt = float((qtruth[k] * w).sum())
                top = qtruth[k].max()
                rms = float(np.sqrt(np.mean((cs[k] - qtruth[k]) ** 2)) / top) if top > 0 else np.nan
                row += [Qf / Qt - 1.0 if Qt > 0 else np.nan, rms]
            rows.append(row)
            store[(tag, lam)] = (cs, ss, qtruth)

    rows = np.array(rows, dtype=float)
    cols = ["event", "true_geom", "lam", "n_samples", "chi2_dof", "Qpads_over_QdE",
            "dQ_frag0", "rms_frag0", "dQ_frag1", "rms_frag1"]
    np.savetxt(a.out + ".csv", rows, delimiter=",", header=",".join(cols), comments="")

    print("\n=== Resume : %d requetes (evenements %d a %d), ecartes : %s ===" % (len(recs), a.start, i1 - 1, skipped if skipped else "aucun"))
    print("sum(pq) / sum(dE/W) : mediane %.3f  (controle des unites de /H/dE : doit valoir ~1)" % np.nanmedian(rows[:, 5]))
    print("\nmode   lam        chi2/ndl   |dQ| frag0 (med / 68%)   |dQ| frag1   rms/max frag0   rms/max frag1")
    for mode, flag in (("est", 0), ("true", 1)):
        for lam in lams:
            sel = (rows[:, 1] == flag) & (rows[:, 2] == lam)
            if not sel.any():
                continue
            r = rows[sel]
            def pc(v):
                v = np.abs(v[np.isfinite(v)])
                return (np.median(v), np.percentile(v, 68)) if len(v) else (np.nan, np.nan)
            q0 = pc(r[:, 6]); q1 = pc(r[:, 8])
            print("%-5s  %-9g  %8.3f   %8.3f / %-8.3f     %8.3f / %-8.3f  %12.3f  %12.3f"
                  % (mode, lam, np.nanmedian(r[:, 4]), q0[0], q0[1], q1[0], q1[1], np.nanmedian(r[:, 7]), np.nanmedian(r[:, 9])))
    print("\nTableau : %s.csv ; requete : %s ; reponse : %s" % (a.out, req, rsp))

    # --- figures ---
    if a.plot_n > 0:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        events = sorted(set(info[r["tag"]]["event"] for r in recs))[:a.plot_n]
        lam_plot = lams[len(lams) // 2]
        colors = {"est": "C0", "true": "C3"}
        labels = {"est": "ajustement, geometrie estimee", "true": "ajustement, geometrie vraie"}
        for i in events:
            tags = dict((info[r["tag"]]["mode"], r["tag"]) for r in recs if info[r["tag"]]["event"] == i)
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            for k in range(2):
                qbest = None
                for mode in ("est", "true"):
                    if mode not in tags:
                        continue
                    cs, ss, qt = store[(tags[mode], lam_plot)]
                    l = np.arange(len(cs[k])) * a.delta
                    ax[k].fill_between(l, cs[k] - ss[k], cs[k] + ss[k], color=colors[mode], alpha=0.15)
                    ax[k].plot(l, cs[k], "o-", color=colors[mode], ms=3, label=labels[mode])
                    if qbest is None or len(qt[k]) > len(qbest):
                        qbest = qt[k]
                lt = np.arange(len(qbest)) * a.delta
                ax[k].plot(lt, qbest, "k.-", ms=3, label="verite (moyenne sur chapeaux)")
                name = info[list(tags.values())[0]]["geo"]["truth"][k]
                ax[k].set_xlabel("l (cm)"); ax[k].set_ylabel("e-/cm")
                ax[k].set_title("fragment %d (%s), evt %d, lam=%g" % (k, name, i, lam_plot))
            ax[0].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig("%s_evt%d.png" % (a.out, i), dpi=110)
            plt.close(fig)
        print("figures : %s_evt*.png (bande = sigma non contraint)" % a.out)


if __name__ == "__main__":
    main()
