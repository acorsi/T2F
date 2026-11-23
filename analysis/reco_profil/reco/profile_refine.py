#!/usr/bin/env python3
"""
profile_refine.py : profilage detaille de refine_geom.py, SANS modifier aucun de vos fichiers.

Principe : on importe refine_geom, on enveloppe (a l'execution, en memoire) ses fonctions et celles de
fit_profile / reco_split / rings / numpy.linalg par des chronometres, puis on appelle refine_geom.main().
Les resultats sont identiques a ceux de refine_geom.py (aucune valeur n'est changee).

Usage : exactement les arguments de refine_geom.py, par exemple
  python3 profile_refine.py ../drift_output/drift_pad5mm_new.h5 --exe ../build/profile_basis --cfg ../drift.cfg --n 200 ...
A placer dans le meme dossier que refine_geom.py (imports).

Phases :
  setup : lecture du h5, estimation de depart (build_state), construction de la table par le serveur C++
  geom  : affinage de la geometrie (refine_chunk : Levenberg-Marquardt)
  ext   : test de confinement et rallonge de la grille (extend_chunk)
  final : profils finaux (3 geometries x lam), diagnostics, tableaux, figures (tout ce qui suit)

Temps "inclusif" = fonction et ce qu'elle appelle ; "propre" = inclusif moins les appelees chronometrees.
Les np.linalg.* sont nommes "parent>np.linalg.xxx" pour distinguer qui les appelle.
Surcout des chronometres : quelques % (appels np.linalg.solve nombreux dans Whitener.apply).
"""

import collections
import os
import resource
import sys
import time

import numpy as np


class Registry(object):
    def __init__(self):
        self.stack = []                              # frames actives : [cle, temps des appelees]
        self.incl = collections.defaultdict(float)   # (phase, nom) -> temps inclusif
        self.child = collections.defaultdict(float)  # (phase, nom) -> temps des appelees chronometrees
        self.calls = collections.defaultdict(int)
        self.extra = collections.defaultdict(float)  # (phase, compteur) -> valeur
        self.phase = "setup"
        self.top_total = 0.0                         # temps des appels chronometres de plus haut niveau
        self.n_events = 0


R = Registry()
PHASES = ["setup", "geom", "ext", "final"]


def timed(name, fn, nested=False):
    """enveloppe fn ; nested=True : le nom devient parent>nom"""
    def wrapper(*args, **kw):
        parent = R.stack[-1][0][1] if R.stack else "main"
        key = (R.phase, (parent + ">" + name) if nested else name)
        frame = [key, 0.0]
        was_top = not R.stack
        R.stack.append(frame)
        t = time.perf_counter()
        try:
            return fn(*args, **kw)
        finally:
            dt = time.perf_counter() - t
            R.stack.pop()
            R.incl[key] += dt
            R.child[key] += frame[1]
            R.calls[key] += 1
            if R.stack:
                R.stack[-1][1] += dt
            if was_top:
                R.top_total += dt
    wrapper.__name__ = getattr(fn, "__name__", name)
    return wrapper


def phased(phase, name, fn):
    inner = timed(name, fn)

    def wrapper(*args, **kw):
        R.phase = phase
        try:
            return inner(*args, **kw)
        finally:
            R.phase = "final"
    return wrapper


def timed_generator(name, genfn):
    """temps passe dans next() d'un generateur (lecture du fichier de reponse)"""
    def wrapper(*args, **kw):
        gen = genfn(*args, **kw)
        nxt = timed(name, lambda: next(gen))
        while True:
            try:
                item = nxt()
            except StopIteration:
                return
            yield item
    return wrapper


# ---------------------------------------------------------------------------
def install():
    import refine_geom as rg
    import fit_profile as fp
    import reco_split
    import rings

    if os.environ.get("JAC_FAST") == "1":         # jacobien par fragment (jac_fast.py) : installe avant les chronometres
        import jac_fast
        jac_fast.install(rg)
        sys.stderr.write("[profile_refine] jacobien par fragment (jac_fast) ACTIF : serveur profile_basis_fast requis\n")

    # --- fonctions de refine_geom (appelees par leur nom global dans le module) ---
    for nm in ("lm_step", "tail_state", "window_matrix", "window_truth", "truth_nodes", "zero_window",
               "event_pads", "pad_waveforms", "geometry_true", "geometry_est", "half_length",
               "write_request", "solve_profile_r", "end_pos"):
        if not hasattr(rg, nm):
            sys.exit("[profile_refine] refine_geom n'a pas de %s : version differente de celle fournie" % nm)
        setattr(rg, nm, timed(nm, getattr(rg, nm)))
    rg.read_response_iter = timed_generator("read_response_iter", rg.read_response_iter)
    if hasattr(rg, "read_response_ext"):
        rg.read_response_ext = timed_generator("read_response_iter", rg.read_response_ext)

    # build_state : compte aussi les evenements traites
    bs = timed("build_state", rg.build_state)

    def build_state(*a, **k):
        out = bs(*a, **k)
        if out[0] is not None:
            R.n_events += 1
        return out
    rg.build_state = build_state

    rg.refine_chunk = phased("geom", "refine_chunk", rg.refine_chunk)
    rg.extend_chunk = phased("ext", "extend_chunk", rg.extend_chunk)
    rg.Env.__init__ = timed("Env.__init__", rg.Env.__init__)
    rg.Server.__init__ = timed("Server.__init__ (table + @@READY)", rg.Server.__init__)

    # --- evaluate : compteurs de requetes, d'evaluations, de lignes ---
    ev = timed("evaluate", rg.evaluate)

    def evaluate(ctx, items, lam, want_G=False):
        R.extra[(R.phase, "requetes")] += 1
        R.extra[(R.phase, "evaluations")] += len(items)
        R.extra[(R.phase, "lignes")] += sum(len(it[0]["y"]) for it in items)
        return ev(ctx, items, lam, want_G)
    rg.evaluate = evaluate

    # --- serveur : temps de calcul C++ lu sur la sortie du serveur, volumes echanges ---
    def server_run(self, req, rsp):
        self.p.stdin.write("%s %s\n" % (req, rsp))
        self.p.stdin.flush()
        while True:
            line = self.p.stdout.readline()
            if line == "":
                sys.exit("[profile_refine] profile_basis s'est arrete pendant une requete")
            if line.startswith("calcul :"):
                R.extra[(R.phase, "cpp_calcul_s")] += float(line.split(":")[1].split()[0])
            if line.startswith("@@DONE"):
                break
            if line.startswith("@@ERROR"):
                sys.exit("[profile_refine] " + line.strip())
        R.extra[(R.phase, "octets_reponse")] += os.path.getsize(rsp)
        R.extra[(R.phase, "octets_requete")] += os.path.getsize(req)
    rg.Server.run = timed("Server.run", server_run)

    # --- fit_profile : blanchiment, NNLS ---
    if os.environ.get("WHITEN_FAST") == "1":      # blanchiment exact rapide (whiten_fast.py), avant les chronometres
        import whiten_fast
        whiten_fast.install(fp)
        sys.stderr.write("[profile_refine] blanchiment rapide (whiten_fast) ACTIF\n")
    fp.Whitener.apply = timed("Whitener.apply", fp.Whitener.apply)
    fp.Whitener.chol = timed("Whitener.chol", fp.Whitener.chol)
    fp.nnls = timed("scipy.nnls", fp.nnls)

    # --- reco_split, rings ---
    reco_split.estimate = timed("reco_split.estimate", reco_split.estimate)
    rings.PadRings.split = timed("PadRings.split", rings.PadRings.split)

    # --- numpy.linalg (nom = parent>fonction) ---
    for nm in ("solve", "inv", "cholesky", "lstsq", "eigh"):
        if hasattr(np.linalg, nm):
            setattr(np.linalg, nm, timed("np.linalg." + nm, getattr(np.linalg, nm), nested=True))
    return rg


# ---------------------------------------------------------------------------
def report(wall, ru_self, ru_child, out=sys.stdout):
    w = out.write
    ne = max(R.n_events, 1)
    w("\n" + "=" * 100 + "\n")
    w("PROFIL DETAILLE : %d evenements traites, temps reel total %.1f s (%.0f ms par evenement)\n"
      % (R.n_events, wall, 1000.0 * wall / ne))
    w("=" * 100 + "\n")

    # --- CPU du systeme ---
    us, ss = ru_self.ru_utime, ru_self.ru_stime
    uc, sc = ru_child.ru_utime, ru_child.ru_stime
    w("\nCPU (getrusage) :              utilisateur      systeme      total     (par evenement)\n")
    w("  Python (ce processus)      %10.1f s %10.1f s %9.1f s   %.0f ms\n" % (us, ss, us + ss, 1000.0 * (us + ss) / ne))
    w("  serveur C++ (profile_basis)%10.1f s %10.1f s %9.1f s   %.0f ms\n" % (uc, sc, uc + sc, 1000.0 * (uc + sc) / ne))
    tot = us + ss + uc + sc
    w("  total                      %10.1f s %10.1f s %9.1f s   %.2f cur-s par evenement ; CPU moyen %.0f %% du temps reel\n"
      % (us + uc, ss + sc, tot, tot / ne, 100.0 * tot / wall))
    w("  temps systeme / total : %.0f %%   (Python seul : %.0f %%, serveur seul : %.0f %%)\n"
      % (100.0 * (ss + sc) / tot, 100.0 * ss / max(us + ss, 1e-9), 100.0 * sc / max(uc + sc, 1e-9)))

    # --- par phase ---
    w("\nPar phase (somme des temps propres ; reel) :\n")
    w("  %-8s %10s %8s\n" % ("phase", "temps (s)", "% reel"))
    sum_all = 0.0
    for ph in PHASES:
        s = sum(R.incl[k] - R.child[k] for k in R.incl if k[0] == ph)
        sum_all += s
        w("  %-8s %10.1f %7.1f %%\n" % (ph, s, 100.0 * s / wall))
    w("  %-8s %10.1f %7.1f %%   (code non chronometre : diagnostics de main, CSV, figures, lecture h5 par evenement)\n"
      % ("reste", wall - sum_all, 100.0 * (wall - sum_all) / wall))

    # --- compteurs ---
    w("\nRequetes au serveur et volumes :\n")
    w("  %-8s %9s %12s %10s %12s %10s %12s %10s\n"
      % ("phase", "requetes", "evaluations", "eval/evt", "lignes/eval", "calcul C++", "reponses", "requetes"))
    for ph in PHASES:
        nq = R.extra.get((ph, "requetes"), 0)
        if nq == 0:
            continue
        nev = R.extra[(ph, "evaluations")]
        w("  %-8s %9d %12d %10.1f %12.0f %9.1f s %9.0f Mo %7.1f Mo\n"
          % (ph, nq, nev, nev / ne, R.extra[(ph, "lignes")] / max(nev, 1), R.extra[(ph, "cpp_calcul_s")],
             R.extra[(ph, "octets_reponse")] / 1.0e6, R.extra[(ph, "octets_requete")] / 1.0e6))

    # --- tableau principal ---
    rows = []
    for k in R.incl:
        ph, nm = k
        incl, child, nc = R.incl[k], R.child[k], R.calls[k]
        rows.append((incl - child, ph, nm, nc, incl))
    rows.sort(reverse=True)
    w("\nFonctions chronometrees, triees par temps PROPRE decroissant :\n")
    w("  %-8s %-52s %9s %10s %10s %7s %9s\n" % ("phase", "fonction", "appels", "propre (s)", "inclus (s)", "% reel", "ms/evt"))
    for own, ph, nm, nc, incl in rows[:45]:
        w("  %-8s %-52s %9d %10.2f %10.2f %6.1f%% %9.1f\n"
          % (ph, nm[:52], nc, own, incl, 100.0 * own / wall, 1000.0 * own / ne))

    # --- lecture rapide de evaluate ---
    w("\nDecomposition d'UNE evaluation du modele (phase geom, evaluate inclusif) :\n")
    key = ("geom", "evaluate")
    n_ev = R.extra.get(("geom", "evaluations"), 0)
    if key in R.incl and n_ev:
        tot_e = R.incl[key]
        w("  evaluate total : %.1f s pour %d evaluations = %.1f ms par evaluation\n" % (tot_e, n_ev, 1000.0 * tot_e / n_ev))
        for nm in ("write_request", "Server.run", "read_response_iter", "Whitener.apply", "solve_profile_r"):
            kk = ("geom", nm)
            if kk in R.incl:
                w("    %-22s %8.1f s  (%4.1f %% de evaluate)  %7.2f ms par evaluation\n"
                  % (nm, R.incl[kk], 100.0 * R.incl[kk] / tot_e, 1000.0 * R.incl[kk] / n_ev))
        c = R.extra.get(("geom", "cpp_calcul_s"), 0.0)
        sr = R.incl.get(("geom", "Server.run"), 0.0)
        w("    dont calcul C++ (OpenMP, mesure par le serveur) : %.1f s ; reste de Server.run "
          "(lecture requete, ecriture reponse, tube) : %.1f s\n" % (c, sr - c))
        a = R.calls.get(("geom", "Whitener.apply>np.linalg.solve"), 0)
        if a:
            w("    appels np.linalg.solve dans Whitener.apply : %d (%.1f par evaluation)\n" % (a, a / float(n_ev)))
    else:
        w("  (aucune evaluation en phase geom)\n")

    # --- environnement de threads ---
    w("\nThreads : ")
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        w("%s=%s  " % (v, os.environ.get(v, "(non defini)")))
    w("\n")
    try:
        from threadpoolctl import threadpool_info
        for d in threadpool_info():
            w("  BLAS/OpenMP charge par Python : %s %s, %s threads\n"
              % (d.get("internal_api"), d.get("filepath", "")[-40:], d.get("num_threads")))
    except Exception:
        w("  (threadpoolctl absent : pip install threadpoolctl pour voir les threads BLAS de numpy)\n")
    w("=" * 100 + "\n")


def main():
    sys.argv = ["refine_geom.py"] + sys.argv[1:]
    rg = install()
    t0 = time.perf_counter()
    status = None
    try:
        rg.main()
    except SystemExit as e:
        status = e
    wall = time.perf_counter() - t0
    report(wall, resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN))
    if status is not None and status.code not in (None, 0):
        sys.exit(status.code)


if __name__ == "__main__":
    main()
