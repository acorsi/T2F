"""
view_list.py : figures de refine_geom.py (avec la verite) pour une liste d'evenements, et galerie HTML.
Usage : python3 view_list.py FICHIER.h5 liste.txt --out DOSSIER [--procs 6] [--script refine_fast.py] -- <arguments de refine_geom.py>
La liste vient de select_events.py (indice groupe n_iter chi2_start a0_deg). Chaque evenement est traite seul
(--start i --n 1 --plot_n 1), en parallele ; les fichiers volumineux d'echange sont supprimes apres chaque evenement.
Sortie : DOSSIER/<groupe>/<indice>_prof.png (profils : depart / affine / vrai), <indice>_win.png (profils fenetres),
DOSSIER/index.html (galerie, a ouvrir dans un navigateur), DOSSIER/work/ (csv de refine_geom, journaux).
"""
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

argv = sys.argv[1:]
if "--" in argv:
    k = argv.index("--"); extra = argv[k + 1:]; argv = argv[:k]
else:
    extra = []
def opt(name, default):
    if name in argv:
        i = argv.index(name); v = argv[i + 1]; del argv[i:i + 2]; return v
    return default
out = opt("--out", None)
procs = int(opt("--procs", "6"))
script = opt("--script", "refine_fast.py")
if len(argv) < 2 or out is None:
    sys.exit(__doc__)
h5file, lst = argv[0], argv[1]
if "--exe_args" not in extra:
    extra += ["--exe_args", "n_threads=1"]

rows = []
with open(lst) as f:
    for line in f:
        if line.strip() and not line.startswith("#"):
            t = line.split()
            rows.append({"i": int(t[0]), "group": t[1], "nit": t[2], "c0": t[3], "a0": t[4]})
os.makedirs(os.path.join(out, "work"), exist_ok=True)
env = dict(os.environ)
for k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    env[k] = "1"
env["HDF5_USE_FILE_LOCKING"] = "FALSE"
env.setdefault("WHITEN_FAST", "1")
env.setdefault("JAC_FAST", "1")

def one(r):
    i = r["i"]
    prefix = os.path.join(out, "work", "e%d" % i)
    cmd = [sys.executable, script, h5file, "--start", str(i), "--n", "1", "--out", prefix, "--plot_n", "1"] + extra
    rc = -1
    for essai in range(2):
        with open(prefix + ".log", "w") as lg:
            rc = subprocess.call(cmd, stdout=lg, stderr=subprocess.STDOUT, env=env)
        if rc == 0:
            break
    for ext in ("_req.txt", "_rsp.bin"):
        try:
            os.remove(prefix + ext)
        except OSError:
            pass
    gd = os.path.join(out, r["group"])
    os.makedirs(gd, exist_ok=True)
    r["rc"] = rc
    for src, dst in (("_evt%d.png" % i, "%d_prof.png" % i), ("_wevt%d.png" % i, "%d_win.png" % i)):
        s = prefix + src
        r[dst[-8:-4]] = os.path.isfile(s)
        if os.path.isfile(s):
            shutil.move(s, os.path.join(gd, dst))
    print("[view_list] evenement %d (%s) : %s" % (i, r["group"], "OK" if rc == 0 and (r.get("prof") or r.get("_win")) else "ECHEC code %d" % rc if rc else "pas de figure (evenement ecarte ?)"), flush=True)
    return r

t0 = time.time()
with ThreadPoolExecutor(max_workers=procs) as ex:
    res = list(ex.map(one, rows))

order = []
for r in rows:
    if r["group"] not in order:
        order.append(r["group"])
with open(os.path.join(out, "index.html"), "w") as g:
    g.write("<html><head><meta charset='utf-8'><title>evenements a iterations maximales</title>"
            "<style>body{font-family:sans-serif} img{width:48%} h3{background:#eee;padding:4px}</style></head><body>\n")
    g.write("<p>Source : %s ; liste : %s</p>\n" % (h5file, lst))
    for gr in order:
        g.write("<h2>%s</h2>\n" % gr)
        for r in res:
            if r["group"] != gr:
                continue
            g.write("<h3>evenement %d : n_iter %s, chi2_start %s, a0 %s deg</h3>\n" % (r["i"], r["nit"], r["c0"], r["a0"]))
            if r.get("prof") or r.get("_win"):
                for dst in ("%d_prof.png", "%d_win.png"):
                    if os.path.isfile(os.path.join(out, gr, dst % r["i"])):
                        g.write("<img src='%s/%s'>\n" % (gr, dst % r["i"]))
            else:
                g.write("<p>pas de figure (code %d, voir work/e%d.log)</p>\n" % (r["rc"], r["i"]))
    g.write("</body></html>\n")
nok = sum(1 for r in res if r.get("prof") or r.get("_win"))
print("[view_list] %d evenements, %d avec figures, %.0f s ; galerie : %s" % (len(res), nok, time.time() - t0, os.path.join(out, "index.html")))
