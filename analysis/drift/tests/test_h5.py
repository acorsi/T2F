#!/usr/bin/env python3
"""Tests sur de vraies sorties de ./process (h5py, numpy ; matplotlib pour le viewer).

Lancement normal : bash run_tests_h5.sh (il fabrique les runs). A la main :
    python3 test_h5.py --pair bruite.h5 propre.h5 [--pair bruite2.h5 propre2.h5] [--analysis ../analysis] [--viewer view_pads.py]

  H7   contenu du h5 : groupes, types, longueurs, valeurs, attributs obligatoires, coherence interne
  H8   reconstruction de trace (t2f_io.trace_xyz) : normes, axes Y et Z (fichier synthetique), barycentre
       de la trace compare a celui des pads
  H9   make_clean_cfg.py : le cfg propre est bien sans bruit, --check passe, et --check detecte une alteration
  H11  reproductibilite : des runs a 1, 2 et 8 threads donnent des fichiers identiques bit a bit (--same REF AUTRE)
  H12  modele capacitance : /pads/c_pF, /pads/enc, attributs, rms du bruit pur par classe de capacite (--cap BRUITE PROPRE)
  H10  viewer view_pads.py sans affichage : pad trouve au clic (carre, transition, anneau, hors detecteur),
       selection, dessin d'evenements (saute si --viewer n'est pas donne)

Code retour : 0 si tout passe. Python simple.
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy as np

n_ok = 0
n_fail = 0


def check(cond, what, value=None, expected=None):
    global n_ok, n_fail
    extra = ""
    if value is not None:
        extra = "  (obtenu %s, attendu %s)" % (value, expected)
    if cond:
        n_ok += 1
        print("  ok   %s%s" % (what, extra))
    else:
        n_fail += 1
        print("  ECHEC %s%s" % (what, extra))


def attr(f, name):
    v = f["/metadata"].attrs[name]
    return v.decode() if isinstance(v, bytes) else v


def has_attr(f, name):
    return name in f["/metadata"].attrs


# ---------------------------------------------------------------------------
# H7 : contenu du h5
# ---------------------------------------------------------------------------
VLEN_INT = ["pid", "pflag", "pk0", "pnt"]
VLEN_FLOAT = ["pq", "pqn"]
ATTRS_REQUIRED = [
    "emission_axis", "git_version", "git_branch", "step_size_eff", "adc_ratio", "delta_t", "noise_enable",
    "read_thr_e", "n_bins", "readout_grid_bins", "window_tail_bins", "window_margin_bins", "W",
    "pad_size", "inter_pad", "frag_mask", "eps_qt", "ghost_enable", "config_text",
    "P_description", "vertex_description", "trace_xyz_description", "pads_description",
]
REMOVED = ["/P/px", "/P/py", "/H/x", "/H/y", "/H/z", "/L/x", "/L/y", "/L/z",
           "/H/vx", "/H/vy", "/H/vz", "/L/vx", "/L/vy", "/L/vz"]


def vlen_base(ds):
    return h5py.check_vlen_dtype(ds.dtype)


def test_h5_content(path, label, noisy):
    print("\nH7 contenu de %s (%s)" % (os.path.basename(path), label))
    f = h5py.File(path, "r")
    N = f["/event_id"].shape[0]
    check(N > 0, "H7 au moins un evenement", N, ">0")

    missing = [n for n in ["/event_id", "/vertex/x", "/vertex/y", "/vertex/z", "/P/phase", "/P/n_ghost"]
               + ["/P/" + k for k in VLEN_INT + VLEN_FLOAT]
               + ["/%s/%s" % (g, k) for g in "HL" for k in ("KE", "theta", "phi", "dE")]
               + ["/pads/" + k for k in ("id", "x", "y", "size", "r_in", "r_out", "sx", "sy")] if n not in f]
    check(not missing, "H7 tous les datasets attendus sont presents", missing, [])
    present_removed = [n for n in REMOVED if n in f]
    check(not present_removed, "H7 les datasets supprimes (px, py, x/y/z, v*) sont absents", present_removed, [])
    missing_a = [a for a in ATTRS_REQUIRED if not has_attr(f, a)]
    check(not missing_a, "H7 attributs obligatoires presents", missing_a, [])
    if missing or missing_a:
        return

    # types
    bad = []
    for k in VLEN_INT:
        if vlen_base(f["/P/" + k]) != np.dtype("int32"):
            bad.append(k)
    for k in VLEN_FLOAT:
        if vlen_base(f["/P/" + k]) != np.dtype("float32"):
            bad.append(k)
    for k in ("phase", "n_ghost"):
        if f["/P/" + k].dtype != np.dtype("int32"):
            bad.append(k)
    for g in "HL":
        if vlen_base(f["/%s/dE" % g]) is None:
            bad.append(g + "/dE")
        for k in ("KE", "theta", "phi"):
            if f["/%s/%s" % (g, k)].dtype != np.dtype("float32"):
                bad.append("%s/%s" % (g, k))
    check(not bad, "H7 types : /P int32 et float32, /H et /L float32", bad, [])

    # longueurs 1re dimension
    firsts = [f[n].shape[0] for n in ["/P/phase", "/P/n_ghost", "/P/pid", "/P/pqn", "/vertex/x", "/vertex/y", "/vertex/z"]
              + ["/%s/%s" % (g, k) for g in "HL" for k in ("KE", "theta", "phi", "dE")]]
    check(all(n == N for n in firsts), "H7 tous les datasets par evenement ont N evenements", sorted(set(firsts)), [N])

    R = int(attr(f, "adc_ratio"))
    grid = int(attr(f, "readout_grid_bins"))
    nb = int(attr(f, "n_bins"))
    tail = int(attr(f, "window_tail_bins"))
    marg = int(attr(f, "window_margin_bins"))
    check(grid == nb + tail + marg, "H7 readout_grid_bins = n_bins + window_tail_bins + window_margin_bins", grid, nb + tail + marg)
    check(int(attr(f, "noise_enable")) == (1 if noisy else 0), "H7 attribut noise_enable coherent", int(attr(f, "noise_enable")), 1 if noisy else 0)
    check(attr(f, "emission_axis") in ("Y", "Z"), "H7 emission_axis = Y ou Z", attr(f, "emission_axis"), "Y|Z")
    check(len(attr(f, "git_version")) > 0, "H7 git_version non vide", attr(f, "git_version"), "non vide")
    check("noise_enable" in attr(f, "config_text"), "H7 config_text contient la config du run", "", "")

    # geometrie
    ids = np.array(f["/pads/id"])
    size = np.array(f["/pads/size"]); rin = np.array(f["/pads/r_in"]); rout = np.array(f["/pads/r_out"])
    check(len(np.unique(ids)) == len(ids), "H7 /pads : identifiants uniques", len(np.unique(ids)), len(ids))
    sq = (rout == 0)
    check(np.all(size[sq] > 0) and np.all(size[~sq] == 0), "H7 /pads : carres de taille > 0, anneaux et transition de taille 0", "", "")
    check(np.all(rout[~sq] > rin[~sq]), "H7 /pads : r_out > r_in pour transition et anneaux", "", "")
    check(np.allclose(size[sq], float(attr(f, "pad_size"))), "H7 /pads : taille des carres = attribut pad_size", float(size[sq][0]), float(attr(f, "pad_size")))
    idset = set(int(i) for i in ids)

    # contenu par evenement
    n_bad = {"longueurs": 0, "pqn": 0, "pnt": 0, "pk0": 0, "fenetre": 0, "ordre": 0, "flag": 0, "q_fantome": 0,
             "n_ghost": 0, "pid": 0, "phase": 0, "fantome_propre": 0, "pq": 0}
    for i in range(N):
        pid = np.array(f["/P/pid"][i]); fl = np.array(f["/P/pflag"][i]); k0 = np.array(f["/P/pk0"][i])
        nt = np.array(f["/P/pnt"][i]); q = np.array(f["/P/pq"][i]); qn = np.array(f["/P/pqn"][i])
        ph = int(f["/P/phase"][i]); ng = int(f["/P/n_ghost"][i])
        if not (len(pid) == len(fl) == len(k0) == len(nt) == len(q)):
            n_bad["longueurs"] += 1
            continue
        if len(qn) != int(nt.sum()):
            n_bad["pqn"] += 1
        if len(nt) and nt.min() < 1:
            n_bad["pnt"] += 1
        if len(k0) and k0.min() < 0:
            n_bad["pk0"] += 1
        ns = (grid - ph + R - 1) // R
        if len(k0) and np.any(k0 + nt > ns):
            n_bad["fenetre"] += 1
        if len(pid) > 1 and not np.all(np.diff(pid) > 0):
            n_bad["ordre"] += 1
        if len(fl) and (fl.min() < 0 or fl.max() > 3):
            n_bad["flag"] += 1
        if np.any(q[fl == 0] != 0.0):
            n_bad["q_fantome"] += 1
        if ng != int((fl == 0).sum()):
            n_bad["n_ghost"] += 1
        if not all(int(p) in idset for p in pid):
            n_bad["pid"] += 1
        if not (0 <= ph < R):
            n_bad["phase"] += 1
        if not noisy and (ng != 0 or np.any(fl == 0)):
            n_bad["fantome_propre"] += 1
        if np.any(q[fl != 0] < -1.0):      # un pad d'induction pure a pq = 0 (a -0 pres) ; jamais une charge negative
            n_bad["pq"] += 1
    texts = {"longueurs": "len(pid) = len(pflag) = len(pk0) = len(pnt) = len(pq)",
             "pqn": "len(pqn) = somme(pnt)", "pnt": "pnt >= 1", "pk0": "pk0 >= 0",
             "fenetre": "pk0 + pnt <= nombre d'echantillons de la grille de lecture", "ordre": "pid strictement croissants",
             "flag": "pflag dans 0..3", "q_fantome": "fantome (pflag 0) => pq = 0", "n_ghost": "n_ghost = nombre de pflag 0",
             "pid": "tous les pid existent dans /pads", "phase": "0 <= phase < adc_ratio",
             "fantome_propre": "fichier sans bruit : aucun fantome", "pq": "pads du signal : pq >= 0 (a 1 e- pres)"}
    for k in ["longueurs", "pqn", "pnt", "pk0", "fenetre", "ordre", "flag", "q_fantome", "n_ghost", "pid", "phase", "fantome_propre", "pq"]:
        check(n_bad[k] == 0, "H7 %s (%d evenements)" % (texts[k], N), n_bad[k], 0)
    f.close()


# ---------------------------------------------------------------------------
# H8 : trace
# ---------------------------------------------------------------------------
def make_synthetic(path, axis):
    """h5 minimal pour tester trace_xyz : 4 evenements de directions connues, vertex (0.3, 0, -0.2), pas 0,05 cm."""
    f = h5py.File(path, "w")
    md = f.create_group("metadata")
    md.attrs["emission_axis"] = axis
    md.attrs["step_size_eff"] = 0.05
    th = np.array([0.0, np.pi / 2, np.pi / 2, np.pi], dtype=np.float32)
    ph = np.array([0.0, 0.0, np.pi / 2, 0.0], dtype=np.float32)
    vt = h5py.vlen_dtype(np.dtype("float32"))
    for g in "HL":
        grp = f.create_group(g)
        grp.create_dataset("theta", data=th)
        grp.create_dataset("phi", data=ph)
        d = grp.create_dataset("dE", (4,), dtype=vt)
        for i in range(4):
            d[i] = np.ones(10, dtype=np.float32)
    v = f.create_group("vertex")
    v.create_dataset("x", data=np.full(4, 0.3, dtype=np.float32))
    v.create_dataset("y", data=np.zeros(4, dtype=np.float32))
    v.create_dataset("z", data=np.full(4, -0.2, dtype=np.float32))
    f.close()


def test_trace(analysis_dir, pairs):
    print("\nH8 reconstruction de trace")
    sys.path.insert(0, analysis_dir)
    import t2f_io

    # (a) fichier synthetique : directions connues, independantes de la formule
    tmp = tempfile.mkdtemp()
    try:
        for axis in ("Y", "Z"):
            p = os.path.join(tmp, "syn_%s.h5" % axis)
            make_synthetic(p, axis)
            f = h5py.File(p, "r")
            main = 1 if axis == "Y" else 2          # composante principale : y pour Y, z pour Z
            other = 2 if axis == "Y" else 1
            exp = {}
            # theta = 0 : suivant +axe ; theta = pi : suivant -axe ; theta = pi/2, phi = 0 : suivant +x ;
            # theta = pi/2, phi = pi/2 : suivant l'autre axe transverse
            exp[0] = (main, +1.0); exp[3] = (main, -1.0); exp[1] = (0, +1.0); exp[2] = (other, +1.0)
            ok = True
            for i, (comp, sign) in exp.items():
                x, y, z = t2f_io.trace_xyz(f, "H", i)
                pts = np.array([x, y, z])
                s = 0.05 * (np.arange(10) + 0.5)
                want = np.zeros((3, 10))
                want[comp] = sign * s
                want[0] += 0.3
                want[2] += -0.2
                if not np.allclose(pts, want, atol=1e-6):
                    ok = False
                    print("        evt %d axe %s : %s au lieu de la composante %d" % (i, axis, np.abs(pts - want).max(), comp))
            check(ok, "H8 axe %s : theta = 0, pi/2, pi suivent la bonne composante du vecteur directeur" % axis)
            f.close()
    finally:
        shutil.rmtree(tmp)

    # (b) fichiers reels (propres) : normes et barycentre
    for noisy_path, clean_path in pairs:
        f = h5py.File(clean_path, "r")
        label = os.path.basename(clean_path)
        geom = t2f_io.pad_geometry(f)
        pitch = float(attr(f, "pad_size")) + float(attr(f, "inter_pad"))
        step = float(attr(f, "step_size_eff"))
        N = f["/event_id"].shape[0]
        worst_norm = 0.0
        diffs = []
        for i in range(N):
            vx, vy, vz = t2f_io.read_vertex(f, i)
            for g in "HL":
                x, y, z = t2f_io.trace_xyz(f, g, i)
                n = len(x)
                s = step * (np.arange(n) + 0.5)
                d = np.sqrt((x - vx) ** 2 + (y - vy) ** 2 + (z - vz) ** 2)
                worst_norm = max(worst_norm, float(np.abs(d - s).max()) if n else 0.0)
            e = t2f_io.event_pads(f, i, geom)
            if not all(geom[int(p)][3] == 0.0 for p in e["pid"]):
                continue            # evenement qui touche un anneau : la position d'un anneau n'a pas de sens
            q = e["pq"]
            bx = float((q * e["x"]).sum() / q.sum()); by = float((q * e["y"]).sum() / q.sum())
            tx = ty = tw = 0.0
            for g in "HL":
                x, y, z = t2f_io.trace_xyz(f, g, i)
                w = np.array(f["/%s/dE" % g][i], dtype=float)
                tx += float((w * x).sum()); ty += float((w * y).sum()); tw += float(w.sum())
            diffs.append(np.hypot(bx - tx / tw, by - ty / tw))
        check(worst_norm < 1.0e-5, "H8 %s : |point de trace - vertex| = s a 1e-5 cm" % label, "%.2e" % worst_norm, "<1e-5")
        diffs = np.array(diffs)
        tol = 0.25 * pitch
        check(len(diffs) > 20 and diffs.max() < tol,
              "H8 %s : barycentre de la trace = barycentre des pads (%d evenements, max < 0,25 pitch)" % (label, len(diffs)),
              "max %.3f cm" % (diffs.max() if len(diffs) else -1), "<%.3f" % tol)
        f.close()


# ---------------------------------------------------------------------------
# H9 : make_clean_cfg
# ---------------------------------------------------------------------------
def common_pad(noisy_path, clean_path):
    """(evenement, indice dans le fichier propre) du premier pad du signal lu dans les deux fichiers."""
    fn = h5py.File(noisy_path, "r"); fc = h5py.File(clean_path, "r")
    for i in range(fn["/event_id"].shape[0]):
        pn = np.array(fn["/P/pid"][i]); fln = np.array(fn["/P/pflag"][i])
        pc = np.array(fc["/P/pid"][i]); flc = np.array(fc["/P/pflag"][i])
        both = set(pn[fln != 0].tolist()) & set(pc[flc != 0].tolist())
        for k in range(len(pc)):
            if int(pc[k]) in both:
                fn.close(); fc.close()
                return i, k
    fn.close(); fc.close()
    return None, None


def test_make_clean(analysis_dir, pairs):
    print("\nH9 make_clean_cfg.py")
    script = os.path.join(analysis_dir, "make_clean_cfg.py")
    tmp = tempfile.mkdtemp()
    try:
        for noisy_path, clean_path in pairs:
            label = os.path.basename(noisy_path)
            r = subprocess.run([sys.executable, "-I", script, noisy_path, "--check", clean_path],
                               capture_output=True, text=True)
            check(r.returncode == 0, "H9 %s : --check accepte le couple bruite / propre" % label, r.returncode, 0)
            check("rms de (bruite - propre)" in r.stdout, "H9 %s : --check imprime le rms du bruit pur" % label, "", "")
            cfg = os.path.join(tmp, label + ".cfg")
            r2 = subprocess.run([sys.executable, "-I", script, noisy_path, "-o", cfg, "--output-file", os.path.join(tmp, "x.h5")],
                                capture_output=True, text=True)
            txt = open(cfg).read()
            lines = [l.split("#")[0].strip() for l in txt.splitlines()]
            check(r2.returncode == 0 and "noise_enable = 0" in lines, "H9 %s : le cfg propre a noise_enable = 0" % label, "", "")
            check("output_file = %s" % os.path.join(tmp, "x.h5") in lines, "H9 %s : output_file imposee" % label, "", "")

        # --check doit detecter une alteration : on abime une charge vraie dans une copie du fichier propre
        noisy_path, clean_path = pairs[0]
        ev, ip = common_pad(noisy_path, clean_path)      # un pad lu dans les deux fichiers : seuls ceux-la sont compares
        check(ev is not None, "H9 un pad commun aux deux fichiers a ete trouve pour l'alteration", ev, "pas None")
        bad = os.path.join(tmp, "abime.h5")
        shutil.copy(clean_path, bad)
        f = h5py.File(bad, "r+")
        pq = np.array(f["/P/pq"][ev]); pq[ip] = pq[ip] * 1.01 + 5.0
        f["/P/pq"][ev] = pq
        f.close()
        r = subprocess.run([sys.executable, "-I", script, noisy_path, "--check", bad], capture_output=True, text=True)
        check(r.returncode != 0 and "ECHEC" in r.stdout, "H9 --check detecte une charge vraie alteree (code retour != 0)", r.returncode, "!=0")

        # ... et une fenetre alteree
        bad2 = os.path.join(tmp, "abime2.h5")
        shutil.copy(clean_path, bad2)
        f = h5py.File(bad2, "r+")
        pnt = np.array(f["/P/pnt"][ev]); pnt[ip] = pnt[ip] + 1
        f["/P/pnt"][ev] = pnt
        f.close()
        r = subprocess.run([sys.executable, "-I", script, noisy_path, "--check", bad2], capture_output=True, text=True)
        check(r.returncode != 0 and "ECHEC" in r.stdout, "H9 --check detecte une fenetre alteree (code retour != 0)", r.returncode, "!=0")
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# H10 : viewer
# ---------------------------------------------------------------------------
class FakeClick(object):
    def __init__(self, ax, x, y):
        self.inaxes = ax
        self.xdata = x
        self.ydata = y


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_viewer(viewer_path, pairs):
    print("\nH10 viewer (sans affichage)")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **k: None
    V = load_module(viewer_path, "view_pads_under_test")

    for noisy_path, clean_path in pairs:
        label = os.path.basename(noisy_path)
        f = h5py.File(noisy_path, "r")
        a = f["/metadata"].attrs
        has_rings = bool(np.any(np.array(f["/pads/r_out"]) > 0))
        v = V.ViewerType2(noisy_path, start_event=0)
        g = v.pad_geom
        squares = [(k, gg) for k, gg in g.items() if gg[3] == 0.0]
        ringlike = [(k, gg) for k, gg in g.items() if gg[3] > 0.0]

        # carres : le centre de chaque carre redonne son identifiant (echantillon de 200)
        step = max(1, len(squares) // 200)
        bad = [k for k, gg in squares[::step] if v._pad_at(gg[0], gg[1]) != k]
        check(not bad, "H10 %s : clic au centre d'un carre -> son identifiant (%d testes)" % (label, len(squares[::step])), bad[:3], [])
        # hors detecteur
        far = max(float(np.hypot(gg[0], gg[1])) for k, gg in squares) + 2.0
        check(v._pad_at(far + 3.0, 0.0) is None and v._pad_at(0.0, -(far + 3.0)) is None,
              "H10 %s : clic hors du detecteur -> aucun pad" % label, "", None)

        if has_rings:
            # transition et anneaux : un point au milieu du pad, dans son quadrant, redonne l'identifiant
            bad = []
            for k, gg in ringlike:
                rm = 0.5 * (gg[2] + gg[3])
                for ang in (25.0, 45.0, 65.0):
                    x = gg[4] * rm * np.cos(np.radians(ang)); y = gg[5] * rm * np.sin(np.radians(ang))
                    if v._pad_at(x, y) != k:
                        bad.append((k, ang))
            check(not bad, "H10 %s : clic dans la zone de chaque anneau / transition -> son identifiant (%d pads, 3 angles)" % (label, len(ringlike)), bad[:3], [])
            # regle du viewer, identique a celle de la simulation (PadPlane::build_cells) : pour un point de la zone
            # r < r_transition : dans un pad carre -> ce pad ; dans le trou inter-pad d'une cellule -> aucun pad
            # (jamais la transition) ; hors de toutes les cellules -> le pad de transition de son quadrant
            half_pad = float(a["pad_size"]) / 2.0
            half_cell = (float(a["pad_size"]) + float(a["inter_pad"])) / 2.0
            sxy = np.array([[gg[0], gg[1]] for k, gg in squares]); sid = [k for k, gg in squares]
            trans = dict()
            rtr = min(gg[2] for _, gg in ringlike)             # r_quinconce = r_in du pad de transition
            for k, gg in ringlike:
                if abs(gg[2] - rtr) < 1e-9:
                    trans[(int(gg[4]), int(gg[5]))] = k
            r_out_tr = max(gg[3] for _, gg in ringlike if abs(gg[2] - rtr) < 1e-9)
            rng = np.random.RandomState(7)
            n_pad = n_gap = n_virt = 0
            bad = []
            for t in range(3000):
                x, y = rng.uniform(-r_out_tr, r_out_tr, 2)
                if np.hypot(x, y) >= r_out_tr or abs(x) < 1e-6 or abs(y) < 1e-6:
                    continue
                dx = np.abs(sxy[:, 0] - x); dy = np.abs(sxy[:, 1] - y)
                in_pad = np.where((dx <= half_pad) & (dy <= half_pad))[0]
                in_cell = np.where((dx <= half_cell) & (dy <= half_cell))[0]
                got = v._pad_at(x, y)
                if len(in_pad):
                    n_pad += 1
                    if got != sid[in_pad[0]]:
                        bad.append((x, y, got, sid[in_pad[0]]))
                elif len(in_cell):
                    n_gap += 1
                    if got in trans.values():
                        bad.append((x, y, got, "pas la transition"))
                else:
                    n_virt += 1
                    if got != trans[(1 if x > 0 else -1, 1 if y > 0 else -1)]:
                        bad.append((x, y, got, trans[(1 if x > 0 else -1, 1 if y > 0 else -1)]))
            check(not bad and n_pad > 100 and n_gap > 5 and n_virt > 5,
                  "H10 %s : 3000 points tires dans r < r_transition : carre / trou inter-pad / transition (%d, %d, %d)" % (label, n_pad, n_gap, n_virt),
                  bad[:2], [])

        # selection par clic : ajout puis retrait
        k0, g0 = squares[0]
        v.selected = []
        v._on_click(FakeClick(v.ax_pad, g0[0], g0[1]))
        added = list(v.selected)
        v._on_click(FakeClick(v.ax_pad, g0[0], g0[1]))
        check(added == [k0] and v.selected == [], "H10 %s : un clic ajoute le pad, un deuxieme clic le retire" % label, added, [k0])
        v._on_click(FakeClick(v.ax_pad, far + 3.0, 0.0))
        check(v.selected == [], "H10 %s : clic hors detecteur : selection inchangee" % label, v.selected, [])

        # dessin de quelques evenements, y compris un evenement qui touche un anneau
        ev_list = [0, 1, 2]
        if has_rings:
            ringids = set(k for k, gg in ringlike)
            for i in range(f["/event_id"].shape[0]):
                pid = np.array(f["/P/pid"][i]); fl = np.array(f["/P/pflag"][i])
                if any(int(p) in ringids for p, fx in zip(pid, fl) if fx != 0):
                    ev_list.append(i)
                    break
        ok = True
        for i in ev_list:
            try:
                v.selected = []
                v.event = i
                v._draw()
                v.fig.canvas.draw()
            except Exception as e:       # noqa
                ok = False
                print("        evenement %d : %s: %s" % (i, type(e).__name__, e))
        check(ok, "H10 %s : dessin des evenements %s sans erreur" % (label, ev_list))
        plt.close("all")
        f.close()


# ---------------------------------------------------------------------------
# H11 : reproductibilite multi-threads
# ---------------------------------------------------------------------------
def all_datasets(f):
    names = []
    f.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
    return sorted(names)


def test_same(ref_path, other_path):
    print("\nH11 reproductibilite : %s = %s (nombres de threads differents)" % (os.path.basename(ref_path), os.path.basename(other_path)))
    a = h5py.File(ref_path, "r"); b = h5py.File(other_path, "r")
    na, nb = all_datasets(a), all_datasets(b)
    check(na == nb, "H11 memes datasets dans les deux fichiers", len(nb), len(na))
    bad = []
    for d in na:
        if d not in b:
            continue
        x, y = a[d], b[d]
        if x.shape != y.shape:
            bad.append((d, "forme"))
            continue
        if h5py.check_vlen_dtype(x.dtype) is not None:
            n = sum(0 if np.array_equal(np.array(x[i]), np.array(y[i])) else 1 for i in range(x.shape[0]))
        else:
            n = 0 if np.array_equal(np.array(x), np.array(y)) else 1
        if n:
            bad.append((d, n))
    check(not bad, "H11 tous les datasets sont identiques bit a bit (%d datasets)" % len(na), bad[:4], [])
    ignore = ("config_text", "created")           # texte de la config (n_threads, output_file) et date
    diff = [k for k in a["/metadata"].attrs.keys() if k not in ignore and str(a["/metadata"].attrs[k]) != str(b["/metadata"].attrs.get(k))]
    check(not diff, "H11 tous les attributs sont identiques (sauf config_text et created)", diff, [])
    a.close(); b.close()


# ---------------------------------------------------------------------------
# H12 : bruit par pad selon la capacite (noise_model = capacitance)
# ---------------------------------------------------------------------------
def test_capacitance(noisy_path, clean_path):
    label = os.path.basename(noisy_path)
    print("\nH12 bruit par pad selon la capacite : %s" % label)
    f = h5py.File(noisy_path, "r")
    g = h5py.File(clean_path, "r")
    a = f["/metadata"].attrs
    mod = attr(f, "noise_model")
    check(mod == "capacitance", "H12 attribut noise_model = capacitance", mod, "capacitance")
    check("/pads/c_pF" in f and "/pads/enc" in f, "H12 /pads/c_pF et /pads/enc presents", "", "")
    ids = np.array(f["/pads/id"]); c = np.array(f["/pads/c_pF"], dtype=float); enc = np.array(f["/pads/enc"], dtype=float)
    check(len(c) == len(ids) and len(enc) == len(ids), "H12 /pads/c_pF et /pads/enc : un element par pad", len(c), len(ids))
    check(np.all(np.isfinite(c)) and np.all(c >= 0.0), "H12 capacites finies et >= 0", "", "")
    enc0 = float(a["enc0"]); enc_ref = float(a["enc_ref"]); c_ref = float(a["c_ref_pf"]); b = float(a["enc_slope_per_pf"])
    check(abs(b - np.sqrt(enc_ref ** 2 - enc0 ** 2) / c_ref) < 1e-9 * b, "H12 pente b = sqrt(enc_ref^2 - enc0^2) / c_ref", b, "")
    check(np.allclose(enc, np.sqrt(enc0 ** 2 + (b * c) ** 2), rtol=1e-9, atol=0.0), "H12 /pads/enc = sqrt(enc0^2 + (b C)^2)", "", "")
    check(abs(float(a["enc_total"]) - enc_ref) < 1e-9, "H12 attribut enc_total = enc_ref (pad de reference)", float(a["enc_total"]), enc_ref)
    cf = attr(f, "pad_capacitance_file")
    if cf == "":
        cdef = float(a["c_cable_pf"]) + float(a["c_routing_pf"])
        check(np.allclose(c, cdef), "H12 sans fichier : capacite unique = c_cable_pf + c_routing_pf", float(c[0]), cdef)
    if int(attr(f, "ghost_enable")) == 1 and "ghost_p_mean" in a:
        pm, pn, px = float(a["ghost_p_mean"]), float(a["ghost_p_min"]), float(a["ghost_p_max"])
        check(pn * (1 - 1e-9) <= pm <= px * (1 + 1e-9) and pn > 0.0, "H12 p fantome : min <= moyenne <= max, > 0", pm, "")
        check(np.all(np.diff(enc[np.argsort(c)]) >= -1e-9), "H12 ENC croissante avec la capacite", "", "")

    # rms du bruit pur (bruite - propre) par classe de capacite, sur les pads a fort signal (pas de biais du seuil)
    cap = dict(zip([int(i) for i in ids], c))
    acc = {}
    N = len(f["/P/pid"])
    for ev in range(N):
        pa = np.array(f["/P/pid"][ev]); ka = np.array(f["/P/pk0"][ev]); na = np.array(f["/P/pnt"][ev]); qa = np.array(f["/P/pqn"][ev])
        pq = np.array(f["/P/pq"][ev])
        pb = np.array(g["/P/pid"][ev]); kb = np.array(g["/P/pk0"][ev]); nb = np.array(g["/P/pnt"][ev]); qb = np.array(g["/P/pqn"][ev])
        oa = np.concatenate([[0], np.cumsum(na)]); ob = np.concatenate([[0], np.cumsum(nb)])
        ib = dict((int(p), i) for i, p in enumerate(pb))
        for i, p in enumerate(pa):
            j = ib.get(int(p))
            if j is None or pq[i] < 1.0e5:
                continue
            lo = max(ka[i], kb[j]); hi = min(ka[i] + na[i], kb[j] + nb[j])
            if hi <= lo:
                continue
            d = qa[oa[i]:oa[i + 1]][lo - ka[i]:hi - ka[i]] - qb[ob[j]:ob[j + 1]][lo - kb[j]:hi - kb[j]]
            acc.setdefault(round(cap[int(p)], 6), []).append(d)
    n_cl = 0
    for cl in sorted(acc):
        d = np.concatenate(acc[cl])
        if len(d) < 20000:
            continue
        n_cl += 1
        ex = float(np.sqrt(enc0 ** 2 + (b * cl) ** 2))
        rms = float(np.sqrt(np.mean(d ** 2)))
        check(abs(rms / ex - 1.0) < 0.03, "H12 rms du bruit pur a %.0f pF (%d echantillons) = ENC(C) a 3 %%" % (cl, len(d)), "%.1f" % rms, "%.1f" % ex)
    check(n_cl >= 1, "H12 au moins une classe de capacite assez peuplee pour mesurer le bruit", n_cl, ">=1")
    f.close(); g.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, action="append", metavar=("BRUITE", "PROPRE"), required=True)
    ap.add_argument("--cap", nargs=2, action="append", metavar=("BRUITE", "PROPRE"), default=[],
                    help="couple de runs en modele capacitance (H12)")
    ap.add_argument("--same", nargs=2, action="append", metavar=("REF", "AUTRE"), default=[])
    ap.add_argument("--analysis", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
    ap.add_argument("--viewer", default=None)
    args = ap.parse_args()
    analysis = os.path.abspath(args.analysis)
    pairs = [(os.path.abspath(a), os.path.abspath(b)) for a, b in args.pair]

    for noisy_path, clean_path in pairs:
        test_h5_content(noisy_path, "avec bruit", True)
        test_h5_content(clean_path, "sans bruit", False)
    for ref_path, other_path in args.same:
        test_same(os.path.abspath(ref_path), os.path.abspath(other_path))
    for noisy_path, clean_path in args.cap:
        test_capacitance(os.path.abspath(noisy_path), os.path.abspath(clean_path))
    test_trace(analysis, pairs)
    test_make_clean(analysis, pairs)
    if args.viewer:
        test_viewer(os.path.abspath(args.viewer), pairs)
    else:
        print("\nH10 viewer : saute (pas de --viewer)")

    print("\n%d verifications reussies, %d echecs" % (n_ok, n_fail))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
