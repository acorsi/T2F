#!/usr/bin/env python3
# Balayage : 20 configurations (gaz x geometrie) x m_cells, R fixe (pour isoler l'effet de m_cells).
# usage : python3 sweep_all.py [R [liste de configurations, ex. 1,5,7 [liste de m_cells, ex. 8,16]]]
# Ecrit out/sweep_R<R>.csv. Chaque etape est un processus independant (pas de multiprocessing : conflit avec numpy).
import os, subprocess, sys, re, csv
from concurrent.futures import ThreadPoolExecutor

G1 = "DT_kg=0.02 DT_ga=0.04 DL_kg=0.012 DL_ga=0.016 ve_kg=0.010 ve_ga=0.003"     # Ar+10%CF4 (drift.cfg)
G2 = "DT_kg=0.06 DT_ga=0.05 DL_kg=0.035 DL_ga=0.02 ve_kg=0.0055 ve_ga=0.002"     # P10
G3 = "DT_kg=0.015 DT_ga=0.03 DL_kg=0.01 DL_ga=0.012 ve_kg=0.008 ve_ga=0.004"     # peu diffusif (fictif)
G4 = "DT_kg=0.08 DT_ga=0.09 DL_kg=0.05 DL_ga=0.04 ve_kg=0.004 ve_ga=0.0025"      # tres diffusif (fictif)
CONF = [(G1, .3, .5), (G1, .2, .3), (G1, .5, .5), (G1, .6, .2), (G2, .3, .3), (G2, .5, .5), (G2, .2, .6),
        (G2, .6, .2), (G3, .3, .5), (G3, .2, .2), (G3, .5, .3), (G3, .6, .6), (G4, .3, .5), (G4, .2, .3),
        (G4, .5, .6), (G4, .6, .2), (G1, .4, .4), (G2, .4, .4), (G3, .4, .25), (G4, .45, .35)]
MC = [8, 12, 16, 24, 32]
ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")


def job(args):
    idx, gas, g, pad, m, R = args
    name = "c%02d" % idx
    base = "out/sw_%s_mc%d" % (name, m)
    cmd = "./sweep %s %s dga=%g pad=%g m_cells=%d R=%d" % (name, gas, g, pad, m, R)
    out = subprocess.run(cmd.split(), capture_output=True, text=True, env=ENV).stdout.strip()
    if "SKIP" in out or not os.path.exists(base + ".csv"):
        return [idx, g, pad, m, "", "", ""]
    sh = subprocess.run(["python3", "sweep_shape.py", base], capture_output=True, text=True, env=ENV).stdout
    err = float(re.search(r"erreur max ([\d.e+-]+)", sh).group(1))
    kv = dict(re.findall(r"(\w+)=([-\d.e+]+)", out))
    return [idx, g, pad, m, kv["a"], kv["sigma_gap"], err]


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    sel = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else list(range(1, len(CONF) + 1))
    mcs = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else MC
    jobs = [(i, CONF[i - 1][0], CONF[i - 1][1], CONF[i - 1][2], m, R) for i in sel for m in mcs]
    with ThreadPoolExecutor(2) as ex:
        res = list(ex.map(job, jobs))
    with open("out/sweep_R%d.csv" % R, "w") as fh:
        w = csv.writer(fh)
        w.writerow(["conf", "g", "pad", "m_cells", "a", "sigma_gap", "shape_max"])
        w.writerows(res)
    print("fini : out/sweep_R%d.csv" % R)
