"""
trace_ref2.py
Précalcul de tables dedx_bin[Z, A, KE] pour génération rapide de traces de référence.
Pour chaque (Z, A) : array 1D où l'index = int(KE / KE_STEP) donne l'énergie
déposée dans le premier bin de 2mm quand le fragment arrive avec cette KE.
"""

import os
import numpy as np

# ── Paramètres physiques ──────────────────────────────────────────────────────
DENSITY       = 0.0004835        # g/cm³
STEP_UM       = 100.0            # µm
BIN_MM        = 2.0              # mm
STEPS_PER_BIN = int(BIN_MM * 1e3 / STEP_UM)   # 20
STEP_MGCM2    = DENSITY * 1e3 * STEP_UM * 1e-4 # mg/cm² par pas fin

KE_STEP       = 0.1              # MeV, pas grille KE
A_OFFSET      = 5                # +/- autour de A_mean


def load_dedx_tables(dedx_dir, z_min=29, z_max=70, verbose=True):
    """Charge les tables dE/dx. Retourne dict Z -> (kea_array, dedx_array)."""
    tables = {}
    for Z in range(z_min, z_max + 1):
        fpath = os.path.join(dedx_dir, f"Z{Z:03d}_dedx_svd.txt")
        if not os.path.exists(fpath):
            continue
        kea, dedx = [], []
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        kea.append(float(parts[0]))
                        dedx.append(float(parts[1]))
                    except ValueError:
                        continue
        if len(kea) > 2:
            kea_a  = np.array(kea)
            dedx_a = np.array(dedx)
            valid  = (kea_a > 0) & (dedx_a > 0)
            if valid.sum() > 2:
                tables[Z] = (kea_a[valid], dedx_a[valid])
    if verbose:
        print(f"Tables dE/dx chargées : {len(tables)} Z ({min(tables)}-{max(tables)})")
    return tables


def interpolate_dedx(kea_val, kea_arr, dedx_arr):
    """Interpolation log-log de dE/dx à kea_val [MeV/u]."""
    if kea_val <= 0 or kea_val < kea_arr[0]:
        return float(dedx_arr[0])
    if kea_val >= kea_arr[-1]:
        return float(dedx_arr[-1])
    return float(np.exp(np.interp(np.log(kea_val),
                                   np.log(kea_arr),
                                   np.log(dedx_arr))))


def dedx_one_bin(KE, A, kea_arr, dedx_arr):
    """
    Calcule l'énergie déposée dans un bin de 2mm (STEPS_PER_BIN pas)
    pour un fragment arrivant avec énergie cinétique KE [MeV].
    Retourne (dE_bin, KE_out) où KE_out est l'énergie résiduelle après le bin.
    Si le fragment s'arrête dans le bin, retourne (KE, 0.0).
    """
    E = float(KE)
    bin_energy = 0.0
    for _ in range(STEPS_PER_BIN):
        dE = interpolate_dedx(E / A, kea_arr, dedx_arr) * STEP_MGCM2
        if dE >= E:
            bin_energy += E
            return bin_energy, 0.0
        E          -= dE
        bin_energy += dE
    return bin_energy, E


def _compute_one_za(Z, A, tables):
    """Calcule l'array dedx_bin pour un couple (Z,A). Utilisé en parallèle."""
    if Z not in tables:
        return None
    kea_arr, dedx_arr = tables[Z]
    ke_max = kea_arr[-1] * A
    n_ke   = int(ke_max / KE_STEP) + 1
    arr    = np.zeros(n_ke, dtype=np.float32)
    for ke_idx in range(1, n_ke):
        KE = ke_idx * KE_STEP
        dE, _ = dedx_one_bin(KE, A, kea_arr, dedx_arr)
        arr[ke_idx] = dE
    return (Z, A, arr)


def build_dedx_bin_lookup(tables, za_list=None, verbose=True):
    """
    Précalcule pour chaque (Z, A) un array 1D :
      dedx_bin_arr[ke_idx] = énergie déposée dans le 1er bin de 2mm
                             quand le fragment arrive avec KE = ke_idx * KE_STEP MeV.

    Si za_list est fourni (liste de tuples (Z,A)), seuls ces couples sont calculés.
    Sinon, balaye la grille A_mean ± A_OFFSET pour chaque Z.

    Retourne dict (Z, A) -> np.array float32.
    """
    from joblib import Parallel, delayed

    if za_list is None:
        za_list = []
        for Z, (kea_arr, _) in tables.items():
            A_mean = int(round(2.57 * Z))
            for A in range(A_mean - A_OFFSET, A_mean + A_OFFSET + 1):
                za_list.append((Z, A))

    # Dédoublonner
    za_list = list(dict.fromkeys(za_list))

    results = Parallel(n_jobs=-1)(
        delayed(_compute_one_za)(Z, A, tables) for Z, A in za_list
    )

    lookup = {}
    for r in results:
        if r is not None:
            Z, A, arr = r
            lookup[(Z, A)] = arr

    if verbose:
        print(f"Lookup précalculé : {len(lookup)} couples (Z,A)")
    return lookup


def get_trace_ref_fast(Z, A, KE_MeV, lookup):
    """
    Génère la trace de référence pour (Z, A, KE) depuis le lookup précalculé.
    Retourne np.array float32 (longueur variable, dernier bin incomplet ignoré).
    """
    key = (Z, A)
    if key not in lookup:
        return None

    arr = lookup[key]
    n_max = len(arr)
    bins = []
    KE = float(KE_MeV)

    while KE > 0:
        ke_idx = int(KE / KE_STEP)
        if ke_idx <= 0 or ke_idx >= n_max:
            break
        dE = float(arr[ke_idx])
        if dE <= 0 or dE >= KE:
            break
        bins.append(dE)
        KE -= dE

    return np.array(bins, dtype=np.float32)
