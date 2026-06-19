"""
generate_templates_full.py
==========================
Génère les templates (μ, σ) par pad pour tous les triplets (Z, A, KE_bin)
en utilisant les ranges extraits depuis za_ranges.npz.
Couvre tous les Z de la fission du 252Cf (Z_L ∈ [29,70]).
"""

import sys
import os
import numpy as np
import time
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_ref import (load_dedx_tables, build_dedx_bin_lookup,
                       get_trace_ref_fast, DENSITY, BIN_MM)

# ── Paramètres ────────────────────────────────────────────────────────────────
DEDX_DIR    = '../../dedx_from_data/G4'
RANGES_FILE = 'za_ranges.npz'
OUT_FILE    = 'templates_full.npz'

KE_BIN_WIDTH = 0.5    # MeV
N_SIM        = 2000   # traces simulées par bin KE
N_JOBS       = 14

SIGMA_COEFF = 0.027
BIN_MGCM2   = DENSITY * 1e3 * BIN_MM * 0.1

# ── Chargement ranges ─────────────────────────────────────────────────────────
print("Chargement za_ranges.npz...")
ranges   = np.load(RANGES_FILE)
Z_L_arr  = ranges['Z_L'].astype(int)
AL_min   = ranges['AL_min'].astype(int)
AL_max   = ranges['AL_max'].astype(int)
KEL_min  = ranges['KEL_min'].astype(float)
KEL_max  = ranges['KEL_max'].astype(float)
AH_min   = ranges['AH_min'].astype(int)
AH_max   = ranges['AH_max'].astype(int)
KEH_min  = ranges['KEH_min'].astype(float)
KEH_max  = ranges['KEH_max'].astype(float)
Z_SUM    = int(ranges['Z_SUM'][0])

n_zl = len(Z_L_arr)
print(f"  {n_zl} Z_L couverts : {Z_L_arr.tolist()}")

# ── Construction liste (Z, A, ke_min, ke_max) ─────────────────────────────────
print("\nConstruction liste des jobs...")
jobs = []   # (Z, A, ke_min, ke_max, sigma_pad)

for i, zl in enumerate(Z_L_arr):
    zh = Z_SUM - zl
    # Côté léger
    for al in range(AL_min[i], AL_max[i] + 1):
        sigma_pad = SIGMA_COEFF * zl * BIN_MGCM2
        jobs.append((zl, al, KEL_min[i], KEL_max[i], sigma_pad))
    # Côté lourd
    for ah in range(AH_min[i], AH_max[i] + 1):
        sigma_pad = SIGMA_COEFF * zh * BIN_MGCM2
        jobs.append((zh, ah, KEH_min[i], KEH_max[i], sigma_pad))

# Dédoublonner par (Z, A) en gardant le range KE le plus large
za_jobs = {}
for Z, A, ke_min, ke_max, sigma_pad in jobs:
    key = (Z, A)
    if key not in za_jobs:
        za_jobs[key] = (ke_min, ke_max, sigma_pad)
    else:
        old_min, old_max, _ = za_jobs[key]
        za_jobs[key] = (min(old_min, ke_min), max(old_max, ke_max), sigma_pad)

jobs_dedup = [(Z, A, v[0], v[1], v[2]) for (Z, A), v in za_jobs.items()]
n_za = len(jobs_dedup)
n_triplets = sum(int((ke_max - ke_min) / KE_BIN_WIDTH) + 1
                 for _, _, ke_min, ke_max, _ in jobs_dedup)
print(f"  {n_za} couples (Z,A)  ~{n_triplets} triplets (Z,A,KE_bin)")

# ── Chargement tables dE/dx + lookup ─────────────────────────────────────────
print("\nChargement tables dE/dx...")
tables = load_dedx_tables(DEDX_DIR)

za_list = [(Z, A) for Z, A, _, _, _ in jobs_dedup]
print("Précalcul lookup...")
lookup = build_dedx_bin_lookup(tables, za_list=za_list)
print(f"  {len(lookup)} couples précalculés")

# ── Fonction de génération pour un (Z, A) ─────────────────────────────────────
def generate_za(Z, A, ke_min, ke_max, sigma_pad, seed):
    rng_local = np.random.default_rng(seed)
    ke_edges  = np.arange(ke_min, ke_max + KE_BIN_WIDTH * 0.5, KE_BIN_WIDTH)
    ke_centers = 0.5 * (ke_edges[:-1] + ke_edges[1:])
    results = []

    for ke_c in ke_centers:
        traces = []
        ke_samples = rng_local.uniform(ke_c - KE_BIN_WIDTH/2,
                                        ke_c + KE_BIN_WIDTH/2, N_SIM)
        for ke in ke_samples:
            ref = get_trace_ref_fast(Z, A, ke, lookup)
            if ref is None or len(ref) < 2:
                continue
            ref   = ref[:-1].astype(np.float32)
            noise = rng_local.normal(0, sigma_pad, len(ref)).astype(np.float32)
            traces.append(ref + noise)

        if len(traces) < 10:
            continue

        n_pads = min(len(t) for t in traces)
        arr    = np.array([t[:n_pads] for t in traces], dtype=np.float32)

        results.append(dict(
            ke_center = float(ke_c),
            mu        = arr.mean(axis=0).astype(np.float32),
            sigma     = arr.std(axis=0).astype(np.float32),
            n_pads    = n_pads,
            n_sim     = len(traces),
        ))

    return (Z, A, results)

# ── Génération parallèle ──────────────────────────────────────────────────────
print(f"\nGénération parallèle ({N_JOBS} workers)...")
t0 = time.time()

raw_results = Parallel(n_jobs=N_JOBS, backend='loky', verbose=5)(
    delayed(generate_za)(Z, A, ke_min, ke_max, sigma_pad, seed)
    for seed, (Z, A, ke_min, ke_max, sigma_pad) in enumerate(jobs_dedup)
)

dt = time.time() - t0
print(f"\nGénération terminée en {dt:.1f}s ({dt/60:.1f} min)")

# ── Assemblage ────────────────────────────────────────────────────────────────
print("Assemblage...")
Z_list         = []
A_list         = []
bin_idx_list   = []
ke_center_list = []
n_pads_list    = []
mu_list        = []
sigma_list     = []

n_templates = 0
for Z, A, results in raw_results:
    for i, r in enumerate(results):
        Z_list.append(Z)
        A_list.append(A)
        bin_idx_list.append(i)
        ke_center_list.append(r['ke_center'])
        n_pads_list.append(r['n_pads'])
        mu_list.append(r['mu'])
        sigma_list.append(r['sigma'])
        n_templates += 1

print(f"  {n_templates} templates générés")

max_pads  = max(len(m) for m in mu_list)
mu_arr    = np.full((n_templates, max_pads), np.nan, dtype=np.float32)
sigma_arr = np.full((n_templates, max_pads), np.nan, dtype=np.float32)
for i, (mu, sig) in enumerate(zip(mu_list, sigma_list)):
    mu_arr[i, :len(mu)]    = mu
    sigma_arr[i, :len(sig)] = sig

np.savez_compressed(
    OUT_FILE,
    Z           = np.array(Z_list,         dtype=np.int16),
    A           = np.array(A_list,         dtype=np.int16),
    bin_idx     = np.array(bin_idx_list,   dtype=np.int16),
    ke_center   = np.array(ke_center_list, dtype=np.float32),
    n_pads      = np.array(n_pads_list,    dtype=np.int16),
    mu          = mu_arr,
    sigma       = sigma_arr,
    meta_ke_bin_width = np.array([KE_BIN_WIDTH]),
    meta_n_sim        = np.array([N_SIM]),
    meta_sigma_coeff  = np.array([SIGMA_COEFF]),
    meta_bin_mgcm2    = np.array([BIN_MGCM2]),
)

fsize = os.path.getsize(OUT_FILE) / 1e6
print(f"\nTemplates sauvegardés : {OUT_FILE}  ({fsize:.1f} MB)")
print(f"  {n_templates} templates  max_pads={max_pads}")

# ── Vérification ──────────────────────────────────────────────────────────────
print("\nVérification...")
d = np.load(OUT_FILE)
print(f"  Shape mu    : {d['mu'].shape}")
print(f"  Z uniques   : {np.unique(d['Z']).tolist()}")
print(f"  KE range    : [{d['ke_center'].min():.1f}, {d['ke_center'].max():.1f}] MeV")
print("Done.")
