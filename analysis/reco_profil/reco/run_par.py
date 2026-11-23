"""
run_par.py : lance refine_fast.py en plusieurs processus independants sur des tranches d'evenements
(file de travail : chaque processus libre prend la tranche suivante), puis fusionne les CSV.

Chaque tranche = un processus refine_fast.py (Python 1 thread, serveur C++ profile_basis_fast a 1 thread).
Rien n'est partage entre processus : les resultats sont ceux d'un run sequentiel tranche par tranche.

Usage :
  python3 run_par.py FICHIER.h5 --n_total N [--start 0] [--procs 6] [--job 500] [--out par] [--keep]
         [--script refine_fast.py] -- <arguments de refine_geom.py : --exe ... --cfg ... --lam ... >
Les arguments apres "--" sont transmis tels quels (sauf --start, --n, --out, --plot_n, geres ici).
Mode production : --script prod_refine.py (sorties <out>_prod.h5 et <out>_skipped.csv ; ne pas passer --lam_plot).
Sorties : <out>_geom.csv, _prof.csv, _flags.csv, _fin.csv, _diag.csv, _wsum.csv, _win.csv (fusionnes, ordre des evenements),
          <out>_jobs/ : journal de chaque tranche (job_<debut>.log) ; fichiers de tranche supprimes sauf avec --keep.
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

KINDS = ["geom", "prof", "flags", "fin", "diag", "wsum", "win", "skipped"]


def run_job(a, extra, start, n):
    tag = os.path.join(a.jobdir, "job_%d" % start)
    cmd = [sys.executable, a.script, a.h5file, "--start", str(start), "--n", str(n), "--out", tag, "--plot_n", "0"] + extra
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[k] = "1"
    env["HDF5_USE_FILE_LOCKING"] = "FALSE"   # plusieurs processus lisent le meme .h5 : evite les echecs de verrou au demarrage
    env.setdefault("WHITEN_FAST", "1")
    env.setdefault("JAC_FAST", "1")
    t0 = time.time()
    for essai in (1, 2):      # un seul nouvel essai en cas d'echec
        with open(tag + ".log", "w") as lg:
            rc = subprocess.call(cmd, stdout=lg, stderr=subprocess.STDOUT, env=env)
        if rc == 0:
            break
        print("[run_par] tranche %d : echec essai %d (code %d)" % (start, essai, rc), flush=True)
    for ext in ("_req.txt", "_rsp.bin"):
        try:
            os.remove(tag + ext)
        except OSError:
            pass
    dt = time.time() - t0
    print("[run_par] tranche %d-%d : %s en %.0f s" % (start, start + n - 1, "OK" if rc == 0 else "ECHEC (code %d, voir %s.log)" % (rc, tag), dt), flush=True)
    return start, rc


def merge_h5(a, starts):
    """sorties HDF5 du mode production (prod_refine.py) : job_<debut>_prod.h5 -> <out>_prod.h5"""
    files = [os.path.join(a.jobdir, "job_%d_prod.h5" % s) for s in starts]
    files = [fn for fn in files if os.path.exists(fn)]
    if not files:
        return
    import prod_io
    n = prod_io.merge_files(a.out + "_prod.h5", files)
    print("[run_par] %s_prod.h5 : %d evenements (%d tranches)" % (a.out, n, len(files)))


def merge(a, starts):
    for kind in KINDS:
        header, rows, nfile = None, [], 0
        for s in starts:
            fn = os.path.join(a.jobdir, "job_%d_%s.csv" % (s, kind))
            if not os.path.exists(fn):
                continue
            with open(fn) as f:
                lines = f.read().splitlines()
            if not lines:
                continue
            if header is None:
                header = lines[0]
            elif lines[0] != header:
                sys.exit("[run_par] en-tetes differents pour %s (tranche %d)" % (kind, s))
            rows.extend(lines[1:])
            nfile += 1
        if header is None:
            continue
        out = "%s_%s.csv" % (a.out, kind)
        with open(out, "w") as f:
            f.write(header + "\n")
            f.write("\n".join(rows) + ("\n" if rows else ""))
        print("[run_par] %s : %d lignes (%d tranches)" % (out, len(rows), nfile))


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("h5file")
    p.add_argument("--n_total", type=int, required=True, help="nombre d'evenements a traiter (a partir de --start)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--procs", type=int, default=6)
    p.add_argument("--job", type=int, default=500, help="evenements par tranche")
    p.add_argument("--out", type=str, default="par")
    p.add_argument("--script", type=str, default="refine_fast.py")
    p.add_argument("--keep", action="store_true", help="garde les fichiers de chaque tranche")
    a = p.parse_args(argv)
    for opt in ("--start", "--n", "--out", "--plot_n"):
        if opt in extra:
            sys.exit("[run_par] ne pas passer %s apres '--' : gere par run_par" % opt)
    if "--exe_args" not in extra:
        extra += ["--exe_args", "n_threads=1"]
    else:
        print("[run_par] ATTENTION : --exe_args fourni, verifiez qu'il contient n_threads=1")
    a.jobdir = a.out + "_jobs"
    if not os.path.isdir(a.jobdir):
        os.makedirs(a.jobdir)
    jobs = []
    s = a.start
    while s < a.start + a.n_total:
        n = min(a.job, a.start + a.n_total - s)
        jobs.append((s, n))
        s += n
    print("[run_par] %d evenements, %d tranches de %d, %d processus" % (a.n_total, len(jobs), a.job, a.procs), flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.procs) as ex:
        futs = [ex.submit(run_job, a, extra, s, n) for (s, n) in jobs]
        res = [f.result() for f in futs]
    bad = [s for s, rc in res if rc != 0]
    print("[run_par] temps reel total %.0f s" % (time.time() - t0))
    if bad:
        sys.exit("[run_par] tranches en echec : %s (pas de fusion ; relancer ces tranches)" % bad)
    starts = sorted(s for s, n in jobs)
    merge(a, starts)
    merge_h5(a, starts)
    if not a.keep:
        for s in starts:
            for kind in KINDS + ["charge", "prod"]:
                for ext in (".csv", ".png", ".h5"):
                    try:
                        os.remove(os.path.join(a.jobdir, "job_%d_%s%s" % (s, kind, ext)))
                    except OSError:
                        pass


if __name__ == "__main__":
    main()
