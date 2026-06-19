#!/usr/bin/env python3
"""
accumulate_dedx_v7.py
=====================
Pipeline complet d'extraction des tables dE/dx depuis données HDF5.

Sections :
  1. Accumulation + lissage spline (sans extrapolation polynomiale)
  2. Fonctions communes + a_predicted
  3. Remplacement des fits pathologiques
  4. Tables finales 2 colonnes (_dedx_corr.txt)
  5. Correction SVD rang-K des tables lisses -> _dedx_svd.txt
  6. Homothetie a(Z) : extrapolation Z manquants depuis _dedx_svd.txt
  7. Extrapolation log-log haute energie
  8. PDF de comparaison final
"""

import h5py
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import make_smoothing_spline, CubicSpline

# ══════════════════════════════════════════════════════════════════════════════
# Paramètres globaux
# ══════════════════════════════════════════════════════════════════════════════

INPUT_FILE    = "../../simulation_output/T2F_G4.h5"
OUTPUT_DIR    = "../../dedx_from_data/G4"
DIAG_DIR      = "diag"
PDF_OUT       = "results.pdf"

KEEP_PROB      = 0.30
RANDOM_SEED    = 42
MAX_BINS       = 50
KEA_BIN_WIDTH  = 0.005
KEA_MAX_ALLOC  = 3.0

MIN_COUNT_SIGMA  = 5
MIN_COUNT_SPLINE = 5
KEA_TRANSITION   = 0.2
W_REDUCE         = 100.0
N_REF            = 100.0
REDUCE_FACTOR    = 2.0
RESIDU_MAX_PAR_BIN = 1.0
KEA_EXTRAP_WIDTH = 0.1
K_EXTRAP         = 2.0
REPLACE_RATIO    = 2.0

RHO_GAS     = 0.00048348
BIN_SIZE_CM = 0.2
DEDX_CONV   = 1.0 / (RHO_GAS * BIN_SIZE_CM * 1000.0)

Z_EXTRAP_LOW  = list(range(27, 32))
Z_EXTRAP_HIGH = list(range(67, 72))

# SVD
SVD_RANK      = 2
MIN_STAT_SVD  = 50
MIN_BINS_SVD  = 5
Z_TRAIN_MIN   = 34
Z_TRAIN_MAX   = 65
Z_PRED_MIN    = 29
Z_PRED_MAX    = 70
N_EXTRAP_SPL  = 5

# Extrapolation log-log
SLOPE_FRAC  = 0.7
EXTRAP_FRAC = 1.1

PANELS_PER_PAGE = 11

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DIAG_DIR,   exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 : Accumulation + lissage
# ══════════════════════════════════════════════════════════════════════════════

print("="*70)
print("Section 1 : Accumulation et lissage")
print("="*70)

print("Chargement des donnees...")
with h5py.File(INPUT_FILE, "r") as f:
    L_Z = f["/L/Z"][:]; L_KE = f["/L/KE"][:]; L_A = f["/L/A"][:]
    L_traces = f["/L/raw_trace"][:]
    H_Z = f["/H/Z"][:]; H_KE = f["/H/KE"][:]; H_A = f["/H/A"][:]
    H_traces = f["/H/raw_trace"][:]

n_events  = len(L_Z)
rng       = np.random.default_rng(RANDOM_SEED)
keep_mask = rng.random(n_events) < KEEP_PROB
idx_kept  = np.where(keep_mask)[0]
print(f"  {n_events} evenements, {len(idx_kept)} conserves ({100*len(idx_kept)/n_events:.2f}%)")

all_Z    = np.concatenate([L_Z, H_Z]).astype(int)
Z_values = np.arange(int(all_Z.min()), int(all_Z.max()) + 1)
all_A    = np.concatenate([L_A, H_A])
A_mean   = {Z: float(np.mean(all_A[all_Z == Z])) if np.any(all_Z == Z) else 1.0
            for Z in Z_values}

n_kea_bins  = int(np.ceil(KEA_MAX_ALLOC / KEA_BIN_WIDTH))
kea_centers = (np.arange(n_kea_bins) + 0.5) * KEA_BIN_WIDTH

bin_sum   = {Z: np.zeros(n_kea_bins, dtype=np.float64) for Z in Z_values}
bin_sum2  = {Z: np.zeros(n_kea_bins, dtype=np.float64) for Z in Z_values}
bin_count = {Z: np.zeros(n_kea_bins, dtype=np.int64)   for Z in Z_values}

def _accum(Za, KEa, Aa, traces, label):
    n = 0
    for idx in idx_kept:
        Z = int(Za[idx]); A = float(Aa[idx])
        if A <= 0 or Z not in bin_sum: continue
        t = np.zeros(MAX_BINS, dtype=np.float64)
        t[:min(len(traces[idx]), MAX_BINS)] = traces[idx][:MAX_BINS]
        nz = np.nonzero(t)[0]
        if len(nz) == 0: continue
        real_len = int(nz[-1]) + 1
        if real_len < 2: continue  # pas assez de bins valides
        KE = float(KEa[idx])
        for i in range(real_len - 1):  # exclure le dernier bin (fragment arrete)
            v = t[i]; b = int((KE - v/2) / A / KEA_BIN_WIDTH)
            if 0 <= b < n_kea_bins:
                bin_sum[Z][b] += v; bin_sum2[Z][b] += v*v; bin_count[Z][b] += 1
            KE -= v
            if KE <= 0: break
        n += 1
    print(f"  {label} : {n} fragments traites")

print("Accumulation L+H...")
_accum(L_Z, L_KE, L_A, L_traces, "L")
_accum(H_Z, H_KE, H_A, H_traces, "H")

# Mean et sigma
mean_raw = {}; sigma_raw = {}
for Z in Z_values:
    c = bin_count[Z]; s1 = bin_sum[Z]; s2 = bin_sum2[Z]
    m = np.where(c > 0, s1/np.maximum(c,1), np.nan)
    v = np.where(c >= 2, np.maximum(s2/np.maximum(c,1)-m**2, 0.), np.nan)
    mean_raw[Z] = m; sigma_raw[Z] = np.sqrt(v)

# Fit sigma/Z = alpha + beta*KE/A
print("\nFit sigma/Z = alpha + beta*KE/A...")
kea_f, soz_f, w_f = [], [], []
for Z in Z_values:
    c = bin_count[Z]; sig = sigma_raw[Z]; has = c > 0
    if not np.any(has): continue
    last = int(np.where(has)[0][-1]) + 1
    kea  = kea_centers[:last]; km = kea[has[:last]][-1]
    excl = np.zeros(last, dtype=bool); excl[np.where(has[:last])[0][:3]] = True
    v = has[:last] & ~excl & (kea<=0.9*km) & (c[:last]>=MIN_COUNT_SIGMA) & np.isfinite(sig[:last]) & (sig[:last]>0)
    if not np.any(v): continue
    kea_f.extend(kea[v]); soz_f.extend(sig[:last][v]/Z); w_f.extend(c[:last][v].astype(float))

kea_f = np.array(kea_f); soz_f = np.array(soz_f); w_f = np.array(w_f)
X = np.column_stack([np.ones_like(kea_f), kea_f])
XtW = (X * w_f[:,None]).T
alpha, beta = np.linalg.solve(XtW@X, XtW@soz_f)
print(f"  alpha={alpha*DEDX_CONV:.6f}  beta={beta*DEDX_CONV:+.6f} MeV/(mg/cm2)")

# Lissage
print("\nLissage...")
corr_tables = {}  # {Z: (kea_arr, dedx_arr)} pour section 5

for Z in Z_values:
    has_data = bin_count[Z] > 0
    if not np.any(has_data):
        print(f"  Z={Z:3d} : aucune donnee, ignore"); continue

    last     = int(np.where(has_data)[0][-1]) + 1
    cnt      = bin_count[Z][:last]
    kea_out  = kea_centers[:last]
    mean_c   = mean_raw[Z][:last] * DEDX_CONV
    sig_resc = np.maximum(Z*(alpha+beta*kea_out), 1e-10)
    sig_mean = sig_resc / np.sqrt(np.maximum(cnt.astype(float),1)) * DEDX_CONV

    mask_spl = has_data[:last] & (cnt>=MIN_COUNT_SPLINE) & np.isfinite(mean_c) & (sig_mean>0)
    n_spl    = int(np.sum(mask_spl))
    dedx_sm  = np.full(last, np.nan)
    lam_used = None; lam_tag = "N/A"; n_extrap = 0

    if n_spl >= 4:
        xs = kea_out[mask_spl]; ys = mean_c[mask_spl]; ws = 1./sig_mean[mask_spl]
        n_mean = float(cnt[mask_spl].mean())
        Weff   = W_REDUCE * max(N_REF/n_mean, 1.)
        try:
            Wc = Weff
            while True:
                wm  = np.where(xs>KEA_TRANSITION, ws/Wc, ws)
                spl = make_smoothing_spline(xs, ys, w=wm, lam=None)
                lam_used = float(np.sum(wm*(ys-spl(xs))**2))
                if lam_used/n_spl <= RESIDU_MAX_PAR_BIN or Wc <= 1.: break
                Wc = max(Wc/REDUCE_FACTOR, 1.)
            lam_tag = f"GCV(wr={Wc:.1f},n={n_mean:.0f})"

            xi = kea_out[has_data[:last]]
            xi = xi[(xi>=xs[0]) & (xi<=xs[-1])]
            for xv, yv in zip(xi, spl(xi)):
                b = int(round((xv-kea_centers[0])/KEA_BIN_WIDTH))
                if 0 <= b < last: dedx_sm[b] = float(yv)

            # Extrapolation polynomiale intra-données (entre spline et dernier bin)
            klv = xs[-1]
            mex = has_data[:last] & np.isnan(dedx_sm) & (kea_out>klv)
            if np.any(mex):
                x0 = klv; c0 = float(spl(x0)); b0 = max(float(spl(x0,nu=1)),0.)
                mfe = has_data[:last] & np.isfinite(mean_c) & (cnt[:last]>=MIN_COUNT_SIGMA) & (kea_out>x0)
                xfe = kea_out[mfe]; yfe = mean_c[mfe]; wfe = 1./sig_mean[mfe] if np.any(mfe) else np.array([])
                def fit_a(xe,ye,we):
                    if len(xe)<1: return 0.
                    dx=xe-x0; rhs=ye-b0*dx-c0
                    num=float(np.sum(we*rhs*dx**2)); den=float(np.sum(we*dx**4))
                    return num/(den+den/K_EXTRAP**2) if (den+den/K_EXTRAP**2)>0 else 0.
                ac = fit_a(xfe,yfe,wfe); xe = kea_out[mex]
                def ee(a): return a*(xe-x0)**2+b0*(xe-x0)+c0
                kl = x0-KEA_EXTRAP_WIDTH
                while True:
                    ye2 = ee(ac)
                    if len(ye2)<=1 or np.all(np.diff(ye2)>=0): break
                    kl += 0.01
                    mr = xfe>=kl; xfe=xfe[mr]; yfe=yfe[mr]; wfe=wfe[mr]
                    ac = fit_a(xfe,yfe,wfe)
                    if kl>=x0: ac=0.; ye2=ee(ac); break
                dedx_sm[mex] = np.maximum(ye2, 0.); n_extrap = int(np.sum(mex))

        except Exception as e:
            lam_tag = "ERR"; print(f"  Z={Z:3d} : spline echoue ({e})")
    else:
        print(f"  Z={Z:3d} : stat insuffisante ({n_spl} bins)")

    n_bins  = int(np.sum(has_data[:last]))
    outfile = os.path.join(OUTPUT_DIR, f"dedx_Z{Z:02d}_raw.dat")
    with open(outfile, "w") as fout:
        fout.write(f"# Table dE/dx pour Z = {Z}\n# A_mean = {A_mean[Z]:.4f}\n")
        fout.write(f"# KEEP_PROB = {KEEP_PROB:.4f}\n")
        fout.write(f"# sigma(Z,KE/A) = Z*(alpha+beta*KE/A), "
                   f"alpha={alpha*DEDX_CONV:.6f}, beta={beta*DEDX_CONV:+.6f} MeV/(mg/cm2)\n")
        fout.write(f"# Lissage : {lam_tag}\n")
        if lam_used is not None: fout.write(f"# Residu spline = {lam_used:.6e}\n")
        fout.write(f"# Binning KE/A : pas = {KEA_BIN_WIDTH} MeV/u\n")
        fout.write(f"# Bins donnees : {n_bins}  Bins spline : {n_spl}\n#\n")
        fout.write(f"# {'KE/A':>14}   {'dE/dx_mean':>22}   {'dE/dx_smooth':>22}   {'n_events':>10}\n")
        fout.write(f"# {'-'*14}   {'-'*22}   {'-'*22}   {'-'*10}\n")
        for bi in range(last):
            if not has_data[bi] or np.isnan(mean_c[bi]): continue
            ss = f"{dedx_sm[bi]:22.6f}" if np.isfinite(dedx_sm[bi]) else f"{'nan':>22}"
            fout.write(f"  {kea_out[bi]:14.6f}   {mean_c[bi]:22.6f}   {ss}   {cnt[bi]:10d}\n")

    print(f"  Z={Z:3d} : {n_bins} bins, {n_spl} spline, {lam_tag}")

print("\nSection 1 terminee.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 : Fonctions utilitaires communes
# ══════════════════════════════════════════════════════════════════════════════

def load_raw_smooth(Z):
    """Lit col 3 (smooth) de _dedx_raw."""
    fp = os.path.join(OUTPUT_DIR, f"dedx_Z{Z:02d}_raw.dat")
    if not os.path.exists(fp): return None
    k, s = [], []
    with open(fp) as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            p = line.split()
            if len(p) < 3: continue
            try:
                kv = float(p[0]); sv = float(p[2])
                if np.isfinite(sv): k.append(kv); s.append(sv)
            except ValueError: continue
    return (np.array(k), np.array(s)) if len(k) >= 4 else None

def load_raw_mean(Z):
    """Lit col 1 (mean) et col 4 (count) de _dedx_raw."""
    fp = os.path.join(OUTPUT_DIR, f"dedx_Z{Z:02d}_raw.dat")
    if not os.path.exists(fp): return None
    k, m = [], []
    with open(fp) as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            p = line.split()
            if len(p) < 4: continue
            try:
                kv = float(p[0]); mv = float(p[1]); cv = int(p[3])
                if np.isfinite(mv) and cv > 0: k.append(kv); m.append(mv)
            except ValueError: continue
    return (np.array(k), np.array(m)) if len(k) >= 4 else None

def load_svd(Z):
    """Lit _dedx_svd (2 colonnes)."""
    fp = os.path.join(OUTPUT_DIR, f"dedx_Z{Z:02d}_svd.dat")
    if not os.path.exists(fp): return None
    k, s = [], []
    with open(fp) as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            p = line.split()
            if len(p) < 2: continue
            try:
                kv = float(p[0]); sv = float(p[1])
                if np.isfinite(sv) and sv > 0: k.append(kv); s.append(sv)
            except ValueError: continue
    return (np.array(k), np.array(s)) if len(k) >= 4 else None

def load_corr(Z):
    """Lit _dedx_corr (2 colonnes)."""
    fp = os.path.join(OUTPUT_DIR, f"dedx_Z{Z:02d}_corr.dat")
    if not os.path.exists(fp): return None
    k, s = [], []
    with open(fp) as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            p = line.split()
            if len(p) < 2: continue
            try:
                kv = float(p[0]); sv = float(p[1])
                if np.isfinite(sv) and sv > 0: k.append(kv); s.append(sv)
            except ValueError: continue
    return (np.array(k), np.array(s)) if len(k) >= 4 else None

# a_predicted depuis les tables raw (section 1 déjà terminée)
_Zr, _ar = [], []
for _Z in sorted([Z for Z in Z_values
                  if os.path.exists(os.path.join(OUTPUT_DIR, f"dedx_Z{Z:02d}_raw.dat"))]):
    rZ = load_raw_smooth(_Z); rZm1 = load_raw_smooth(_Z-1)
    if rZ is None or rZm1 is None: continue
    kZ, sZ = rZ; kZm1, sZm1 = rZm1
    km = max(kZ[0],kZm1[0]); kM = min(kZ[-1],kZm1[-1])
    if kM <= km: continue
    kc = np.arange(km, kM, KEA_BIN_WIDTH)
    if len(kc) < 4: continue
    sZi = np.interp(kc,kZ,sZ); sZm1i = np.interp(kc,kZm1,sZm1)
    den = float(np.sum(sZi**2))
    if den > 0: _Zr.append(_Z); _ar.append(float(np.sum(sZm1i*sZi))/den)

_Zr = np.array(_Zr); _ar = np.array(_ar)
_pl = np.polyfit(_Zr[_Zr<50],  _ar[_Zr<50],  1) if np.any(_Zr<50)  else None
_ph = np.polyfit(_Zr[_Zr>=50], _ar[_Zr>=50], 1) if np.any(_Zr>=50) else None

def a_predicted(Z):
    if Z < 50  and _pl is not None: return float(np.polyval(_pl, Z))
    if Z >= 50 and _ph is not None: return float(np.polyval(_ph, Z))
    return None

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 : Remplacement des fits pathologiques
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("Section 3 : Remplacement des fits pathologiques")
print("="*70)

Z_avail = sorted([Z for Z in range(int(all_Z.min()), 72)
                  if os.path.exists(os.path.join(OUTPUT_DIR, f"dedx_Z{Z:02d}_raw.dat"))])

def _residu(kref, sref, km, mv, Z):
    si = np.interp(km, kref, sref, left=np.nan, right=np.nan)
    ok = np.isfinite(si) & np.isfinite(mv)
    if not np.any(ok): return np.inf
    s2 = np.maximum((Z*(alpha+beta*km[ok])*DEDX_CONV)**2, 1e-10)
    return float(np.sum((si[ok]-mv[ok])**2/s2))

n_rep = 0
for Z in Z_avail:
    Zv = Z+1 if Z<50 else Z-1
    as_ = a_predicted(Z+1) if Z<50 else a_predicted(Z)
    if as_ is None or as_ <= 0: continue
    rv = load_raw_smooth(Zv); rm = load_raw_mean(Z); rf = load_raw_smooth(Z)
    if rv is None or rm is None or rf is None: continue
    kv,sv = rv; km,mm = rm; kf,sf = rf
    ssc = sv*as_ if Z<50 else sv/as_
    rf_ = _residu(kf,sf,km,mm,Z); rs_ = _residu(kv,ssc,km,mm,Z)
    if rs_ < rf_/REPLACE_RATIO:
        fp = os.path.join(OUTPUT_DIR, f"dedx_Z{Z:02d}_raw.dat")
        with open(fp) as f: lines = f.readlines()
        si = np.interp(km, kv, ssc, left=np.nan, right=np.nan)
        with open(fp, "w") as f:
            for line in lines:
                if line.startswith("#"): f.write(line); continue
                p = line.split()
                if len(p) < 4: f.write(line); continue
                try:
                    k=float(p[0]); m=float(p[1]); c=int(p[3])
                    idx2=np.argmin(np.abs(km-k)); sn=si[idx2]
                    ss=f"{sn:22.6f}" if np.isfinite(sn) else f"{'nan':>22}"
                    f.write(f"  {k:14.6f}   {m:22.6f}   {ss}   {c:10d}\n")
                except (ValueError,IndexError): f.write(line)
        print(f"  Z={Z:3d} : remplace depuis Z={Zv}")
        n_rep += 1

print(f"  {n_rep} Z remplaces.")
print("Section 3 terminee.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 : Tables finales 2 colonnes (_dedx_corr.txt)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("Section 4 : Tables finales 2 colonnes (_dedx_corr)")
print("="*70)

Z_avail4 = sorted([Z for Z in range(int(all_Z.min()), 72)
                   if os.path.exists(os.path.join(OUTPUT_DIR, f"dedx_Z{Z:02d}_raw.dat"))])
corr_tables = {}; n4 = 0

for Z in Z_avail4:
    fp = os.path.join(OUTPUT_DIR, f"dedx_Z{Z:02d}_raw.dat")
    meta = []; kl = []; sl = []
    with open(fp) as f:
        for line in f:
            if line.startswith("#"): meta.append(line); continue
            if not line.strip(): continue
            p = line.split()
            if len(p) < 3: continue
            try:
                k=float(p[0]); s=float(p[2])
                if np.isfinite(s) and s>0: kl.append(k); sl.append(s)
            except ValueError: continue
    if len(kl) < 2: continue
    ka = np.array(kl); sa = np.array(sl)
    corr_tables[Z] = (ka, sa)
    out = os.path.join(OUTPUT_DIR, f"dedx_Z{Z:02d}_corr.dat")
    with open(out, "w") as f:
        for ml in meta: f.write(ml)
        f.write(f"#\n# {'KE/A [MeV/u]':>14}   {'dE/dx [MeV/(mg/cm2)]':>22}\n")
        f.write(f"# {'-'*14}   {'-'*22}\n")
        for k,s in zip(ka,sa): f.write(f"  {k:14.6f}   {s:22.6f}\n")
    n4 += 1; print(f"  Z={Z:3d} : {len(kl)} pts -> {out}")

print(f"  {n4} tables (_dedx_corr).")
print("Section 4 terminee.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 : Correction SVD -> _dedx_svd.txt
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("Section 5 : Correction SVD des tables lisses")
print("="*70)

# 5a. Résidus
print("\n5a. Residus delta = mean/S_smooth - 1...")
residuals = {}
for Z in range(Z_TRAIN_MIN, Z_TRAIN_MAX+1):
    if Z not in corr_tables: continue
    ks, ds = corr_tables[Z]
    cZ = bin_count[Z]; hZ = cZ>0
    if not np.any(hZ): continue
    last = int(np.where(hZ)[0][-1])+1
    kr = kea_centers[:last]; mr = mean_raw[Z][:last]*DEDX_CONV; cr = cZ[:last]
    sm = Z*np.maximum(alpha+beta*kr,1e-8)/np.sqrt(np.maximum(cr.astype(float),1))*DEDX_CONV
    v = (cr>=MIN_STAT_SVD) & np.isfinite(mr) & (mr>0)
    if v.sum() < MIN_BINS_SVD:
        print(f"  Z={Z:3d} : stat insuffisante ({v.sum()} bins), ignore"); continue
    si = np.interp(kr[v], ks, ds, left=np.nan, right=np.nan)
    ok = np.isfinite(si) & (si>0)
    if ok.sum() < MIN_BINS_SVD: continue
    residuals[Z] = dict(kea=kr[v][ok], delta=mr[v][ok]/si[ok]-1.,
                        weights=1./np.maximum(sm[v][ok],1e-10)**2,
                        mean=mr[v][ok], s_smooth=si[ok])

rms_g = np.std(np.concatenate([r["delta"] for r in residuals.values()]))
print(f"  Z retenus : {sorted(residuals.keys())}")
print(f"  RMS global residus : {100*rms_g:.2f}%")

if len(residuals) < SVD_RANK+1:
    print("Pas assez de Z, SVD abandonnee.")
    # Copier _dedx_corr.txt -> _dedx_svd.txt pour les Z disponibles
    for Z in range(Z_PRED_MIN, Z_PRED_MAX+1):
        rc = load_corr(Z)
        if rc is None: continue
        kc, dc = rc
        fp = os.path.join(OUTPUT_DIR, f"dedx_Z{Z:02d}_svd.dat")
        with open(fp, "w") as f:
            f.write(f"# Table dE/dx (copie corr, SVD non disponible) Z={Z}\n")
            for k,s in zip(kc,dc): f.write(f"  {k:14.6f}   {s:22.6f}\n")
    Z_valid = np.array([]); s_svd = None; Vt = None; splines = None; kea_grid = None
else:
    # 5b. SVD
    print(f"\n5b. SVD rang-{SVD_RANK}...")
    Zt = sorted(residuals.keys())
    km5 = min(r["kea"].min() for r in residuals.values())
    kM5 = max(r["kea"].max() for r in residuals.values())
    kea_grid = np.arange(km5, kM5+KEA_BIN_WIDTH/2, KEA_BIN_WIDTH*2)
    print(f"  Grille : [{km5:.3f}, {kM5:.3f}] MeV/u ({len(kea_grid)} pts)")

    Rm = []; Zv = []
    for Z in Zt:
        r = residuals[Z]
        dg = np.full(len(kea_grid), np.nan); wg = np.zeros(len(kea_grid))
        for j in range(len(r["kea"])):
            ki = np.argmin(np.abs(kea_grid-r["kea"][j]))
            dg[ki] = (dg[ki]*wg[ki]+r["delta"][j]*r["weights"][j])/(wg[ki]+r["weights"][j]) \
                     if np.isfinite(dg[ki]) else r["delta"][j]
            wg[ki] += r["weights"][j]
        fin = np.isfinite(dg)
        if fin.mean() < 0.3: continue
        idx_f = np.where(fin)[0]
        dg = np.interp(np.arange(len(kea_grid)), idx_f, dg[idx_f])
        Rm.append(dg); Zv.append(Z)

    Rm = np.array(Rm); Z_valid = np.array(Zv)
    U, s_svd, Vt = np.linalg.svd(Rm, full_matrices=False)
    var = s_svd**2/np.sum(s_svd**2)
    print(f"  Matrice : {Rm.shape}")
    for i in range(min(6,len(s_svd))):
        print(f"  mode {i+1} : sigma={s_svd[i]:.4f}  var={100*var[i]:.2f}%  cum={100*np.cumsum(var)[i]:.2f}%")

    # 5c. Splines hybrides
    print("\n5c. Splines hybrides U_i(Z)...")
    def make_spl(Zk, uv):
        us = np.convolve(uv, np.ones(3)/3, mode='same')
        us[0]=uv[0]; us[-1]=uv[-1]
        cs = CubicSpline(Zk, us)
        sl = np.polyfit(Zk[:N_EXTRAP_SPL], us[:N_EXTRAP_SPL], 1)[0]
        sr = np.polyfit(Zk[-N_EXTRAP_SPL:], us[-N_EXTRAP_SPL:], 1)[0]
        z0l=Zk[0]; v0l=us[0]; z0r=Zk[-1]; v0r=us[-1]
        def ev(Za):
            Za=np.atleast_1d(np.asarray(Za,float)); out=np.empty_like(Za)
            l=Za<z0l; ri=Za>z0r; inn=~l&~ri
            out[inn]=cs(Za[inn]); out[l]=v0l+sl*(Za[l]-z0l); out[ri]=v0r+sr*(Za[ri]-z0r)
            return out
        return ev
    splines = [make_spl(Z_valid, U[:,i]) for i in range(SVD_RANK)]

    # 5d. Figures diagnostic
    print("\n5d. Figures de diagnostic...")
    cmap = plt.cm.viridis; cols = cmap(np.linspace(0,1,len(Z_valid)))

    fig,ax = plt.subplots(figsize=(10,5))
    for i,Z in enumerate(Z_valid):
        r=residuals[Z]; ax.plot(r["kea"],100*r["delta"],lw=0.7,alpha=0.6,color=cmap(i/len(Z_valid)))
    ax.axhline(0,color="k",lw=0.8); ax.axhline(5,color="gray",lw=0.5,ls=":")
    ax.axhline(-5,color="gray",lw=0.5,ls=":"); ax.set_xscale("log"); ax.set_ylim(-25,25)
    ax.set_xlabel("KE/A (MeV/u)"); ax.set_ylabel("delta (%)"); ax.set_title(f"Residus bruts (RMS={100*rms_g:.1f}%)")
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(Z_valid[0],Z_valid[-1])),ax=ax,label="Z")
    plt.tight_layout(); plt.savefig(os.path.join(DIAG_DIR,"diag5_residus_bruts.png"),dpi=150); plt.close()
    print("  diag5_residus_bruts.png")

    ns = min(10,len(s_svd)); rks = np.arange(1,ns+1)
    fig,axes = plt.subplots(1,3,figsize=(15,4)); fig.suptitle("SVD residus delta",fontsize=11)
    axes[0].bar(rks,100*var[:ns],color="steelblue",alpha=0.8)
    axes[0].plot(rks,100*np.cumsum(var[:ns]),"ro-",ms=4)
    axes[0].set_xlabel("Rang"); axes[0].set_ylabel("Variance (%)"); axes[0].set_title("Spectre SVD")
    for i in range(min(4,len(s_svd))): axes[1].plot(kea_grid,Vt[i],lw=1.2,label=f"mode {i+1} ({100*var[i]:.1f}%)")
    axes[1].axhline(0,color="k",lw=0.5); axes[1].set_xscale("log")
    axes[1].set_xlabel("KE/A"); axes[1].set_title("Modes V_i"); axes[1].legend(fontsize=7)
    Ze5=np.arange(Z_PRED_MIN,Z_PRED_MAX+1)
    for i in range(min(4,len(s_svd))):
        axes[2].plot(Z_valid,U[:,i]*s_svd[i],"o-",ms=3,lw=1,label=f"mode {i+1}")
        if i<len(splines): axes[2].plot(Ze5,splines[i](Ze5)*s_svd[i],"--",lw=1,alpha=0.6)
    axes[2].axhline(0,color="k",lw=0.5); axes[2].set_xlabel("Z")
    axes[2].set_title("Modes U·σ + splines"); axes[2].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(DIAG_DIR,"diag5_svd_spectrum.png"),dpi=150); plt.close()
    print("  diag5_svd_spectrum.png")

    fig,axes=plt.subplots(1,3,figsize=(15,4)); fig.suptitle("Residus delta vs SVD",fontsize=11)
    for rk,ax in zip([0,1,2],axes):
        Rs = Rm if rk==0 else Rm-sum(s_svd[i]*np.outer(U[:,i],Vt[i]) for i in range(rk))
        tt = "Brut" if rk==0 else f"Apres rang-{rk}"
        for i in range(len(Z_valid)): ax.plot(kea_grid,100*Rs[i],color=cols[i],lw=0.6,alpha=0.6)
        ax.axhline(0,color="k",lw=0.8); ax.axhline(5,color="gray",lw=0.5,ls=":"); ax.axhline(-5,color="gray",lw=0.5,ls=":")
        ax.set_xscale("log"); ax.set_ylim(-20,20); ax.set_xlabel("KE/A"); ax.set_ylabel("delta (%)")
        ax.set_title(f"{tt}  RMS={100*np.std(Rs):.1f}%")
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(Z_valid[0],Z_valid[-1])),ax=axes[-1],label="Z")
    plt.tight_layout(); plt.savefig(os.path.join(DIAG_DIR,"diag5_svd_residuals.png"),dpi=150); plt.close()
    print("  diag5_svd_residuals.png")

    fig,axes=plt.subplots(1,SVD_RANK,figsize=(6*SVD_RANK,4))
    if SVD_RANK==1: axes=[axes]
    for i in range(SVD_RANK):
        axes[i].plot(Z_valid,U[:,i],"o",ms=4,color="steelblue",label="U data")
        axes[i].plot(Ze5,splines[i](Ze5),"-",lw=1.5,color="darkorange",label="spline hybride")
        axes[i].axvline(Z_TRAIN_MIN,color="gray",ls="--",lw=0.8); axes[i].axvline(Z_TRAIN_MAX,color="gray",ls="--",lw=0.8)
        axes[i].set_title(f"mode {i+1}"); axes[i].set_xlabel("Z"); axes[i].legend(fontsize=7); axes[i].grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(DIAG_DIR,"diag5_U_splines.png"),dpi=150); plt.close()
    print("  diag5_U_splines.png")

    # 5e. Export _dedx_svd.txt
    print("\n5e. Export tables corrigees (_dedx_svd)...")
    n5 = 0
    for Z in range(Z_PRED_MIN, Z_PRED_MAX+1):
        if Z not in corr_tables: continue
        ks, ds = corr_tables[Z]
        corr = np.ones(len(ks))
        for i in range(SVD_RANK):
            vi = np.interp(ks, kea_grid, Vt[i], left=Vt[i][0], right=Vt[i][-1])
            corr += float(splines[i](float(Z))) * float(s_svd[i]) * vi
        dc = ds * corr
        fp = os.path.join(OUTPUT_DIR, f"dedx_Z{Z:02d}_svd.dat")
        with open(fp, "w") as f:
            f.write(f"# Table dE/dx corrigee SVD rang-{SVD_RANK} Z={Z}\n")
            f.write(f"# KE/A [MeV/u]        dE/dx [MeV/(mg/cm2)]\n")
            for k,s in zip(ks,dc): f.write(f"  {k:14.6f}   {s:22.6f}\n")
        n5 += 1
    print(f"  {n5} tables exportees (_dedx_svd.txt).")

print("Section 5 terminee.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 : Homothetie a(Z) depuis _dedx_svd.txt -> Z manquants
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("Section 6 : Homothetie a(Z) depuis _dedx_svd.txt")
print("="*70)

Z_avail6 = sorted([Z for Z in Z_values
                   if os.path.exists(os.path.join(OUTPUT_DIR, f"dedx_Z{Z:02d}_svd.dat"))])

Zr6, ar6 = [], []
for Z in Z_avail6:
    rZ = load_svd(Z); rZm1 = load_svd(Z-1)
    if rZ is None or rZm1 is None: continue
    kZ,sZ = rZ; kZm1,sZm1 = rZm1
    km=max(kZ[0],kZm1[0]); kM=min(kZ[-1],kZm1[-1])
    if kM<=km: continue
    kc=np.arange(km,kM,KEA_BIN_WIDTH)
    if len(kc)<4: continue
    si=np.interp(kc,kZ,sZ); sm=np.interp(kc,kZm1,sZm1)
    den=float(np.sum(si**2))
    if den>0: Zr6.append(Z); ar6.append(float(np.sum(sm*si))/den)

Zr6=np.array(Zr6); ar6=np.array(ar6)
pl6 = np.polyfit(Zr6[Zr6<50],  ar6[Zr6<50],  1) if np.any(Zr6<50)  else None
ph6 = np.polyfit(Zr6[Zr6>=50], ar6[Zr6>=50], 1) if np.any(Zr6>=50) else None
if pl6 is not None: print(f"  Fit Z<50  : a(Z) = {pl6[0]:+.6f}*Z + {pl6[1]:.6f}")
if ph6 is not None: print(f"  Fit Z>=50 : a(Z) = {ph6[0]:+.6f}*Z + {ph6[1]:.6f}")

def a_pred6(Z):
    if Z<50  and pl6 is not None: return float(np.polyval(pl6, Z))
    if Z>=50 and ph6 is not None: return float(np.polyval(ph6, Z))
    return None

def write_extrap(Z, kea_arr, s_arr, src, direction):
    # _dedx_raw.txt
    fp_raw = os.path.join(OUTPUT_DIR, f"dedx_Z{Z:02d}_raw.dat")
    with open(fp_raw, "w") as f:
        f.write(f"# Table dE/dx EXTRAPOLEE Z={Z} depuis Z={src} ({direction})\n")
        f.write(f"# sigma alpha={alpha*DEDX_CONV:.6f} beta={beta*DEDX_CONV:+.6f}\n#\n")
        f.write(f"# {'KE/A':>14}   {'mean':>22}   {'smooth':>22}   {'n':>10}\n")
        for k,s in zip(kea_arr, s_arr):
            if np.isfinite(s): f.write(f"  {k:14.6f}   {'nan':>22}   {s:22.6f}   {0:10d}\n")
    # _dedx_svd.txt
    fp_svd = os.path.join(OUTPUT_DIR, f"dedx_Z{Z:02d}_svd.dat")
    with open(fp_svd, "w") as f:
        f.write(f"# Table dE/dx EXTRAPOLEE Z={Z} depuis Z={src} ({direction})\n")
        f.write(f"# KE/A [MeV/u]        dE/dx [MeV/(mg/cm2)]\n")
        for k,s in zip(kea_arr, s_arr):
            if np.isfinite(s) and s>0: f.write(f"  {k:14.6f}   {s:22.6f}\n")
    print(f"  Z={Z:3d} : extrapolee depuis Z={src} ({direction})")

# Cote legers
Z_src_low = min([Z for Z in Z_avail6 if Z > max(Z_EXTRAP_LOW)], default=None)
if Z_src_low is not None:
    res = load_svd(Z_src_low)
    if res is not None:
        ks, ss = res
        for Zt in sorted(Z_EXTRAP_LOW, reverse=True):
            factor = 1.0
            for k in range(Zt+1, Z_src_low+1):
                a = a_pred6(k)
                if a is None or a<=0: factor=None; break
                factor *= a
            if factor is not None: write_extrap(Zt, ks, ss*factor, Z_src_low, "low")

# Cote lourds
Z_src_high = max([Z for Z in Z_avail6 if Z < min(Z_EXTRAP_HIGH)], default=None)
if Z_src_high is not None:
    res = load_svd(Z_src_high)
    if res is not None:
        ks, ss = res
        for Zt in sorted(Z_EXTRAP_HIGH):
            factor = 1.0
            for k in range(Z_src_high+1, Zt+1):
                a = a_pred6(k)
                if a is None or a<=0: factor=None; break
                factor *= a
            if factor is not None: write_extrap(Zt, ks, ss/factor, Z_src_high, "high")

print("Section 6 terminee.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 : Extrapolation log-log haute energie
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("Section 7 : Extrapolation log-log haute energie")
print("="*70)

kea_max_per_Z = {Z: kea_centers[bin_count[Z]>0][-1]
                 for Z in bin_count if np.any(bin_count[Z]>0)}
print(f"  kea_max par Z disponible pour {len(kea_max_per_Z)} Z")

n7 = 0
for Z in range(Z_PRED_MIN, Z_PRED_MAX+1):
    fp = os.path.join(OUTPUT_DIR, f"dedx_Z{Z:02d}_svd.dat")
    if not os.path.exists(fp): continue

    if Z in kea_max_per_Z:
        kea_data = kea_max_per_Z[Z]
    else:
        Zn = min(kea_max_per_Z.keys(), key=lambda z: abs(z-Z))
        kea_data = kea_max_per_Z[Zn]
    kea_target = kea_data * EXTRAP_FRAC

    kea_t, dedx_t, hdrs = [], [], []
    with open(fp) as f:
        for line in f:
            if line.startswith("#"): hdrs.append(line); continue
            if not line.strip(): continue
            p = line.split()
            if len(p) < 2: continue
            try:
                k=float(p[0]); s=float(p[1])
                if np.isfinite(s) and s>0: kea_t.append(k); dedx_t.append(s)
            except ValueError: continue
    if len(kea_t) < 4: continue

    kt = np.array(kea_t); dt = np.array(dedx_t)
    kl = kt[-1]

    kea_new = np.arange(kl+KEA_BIN_WIDTH, kea_target+KEA_BIN_WIDTH*0.5, KEA_BIN_WIDTH)
    if len(kea_new) == 0: continue

    mf = kt >= kl*SLOPE_FRAC
    if mf.sum() < 2: mf = np.zeros(len(kt),dtype=bool); mf[-4:]=True
    slope, intercept = np.polyfit(np.log(kt[mf]), np.log(dt[mf]), 1)
    dedx_new = np.exp(intercept + slope*np.log(kea_new))

    with open(fp, "a") as f:
        for k,s in zip(kea_new, dedx_new): f.write(f"  {k:14.6f}   {s:22.6f}\n")
    n7 += 1

print(f"  {n7} tables prolongees jusqu'a kea_data*{EXTRAP_FRAC}")
print("Section 7 terminee.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 : PDF de comparaison final
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("Section 8 : PDF de comparaison final")
print("="*70)

Z_pred  = list(range(Z_PRED_MIN, Z_PRED_MAX+1))
n_pages = int(np.ceil(len(Z_pred)/PANELS_PER_PAGE))
ncols   = 3
nrows   = int(np.ceil(PANELS_PER_PAGE/ncols))

with PdfPages(PDF_OUT) as pdf:
    for page in range(n_pages):
        zp  = Z_pred[page*PANELS_PER_PAGE:(page+1)*PANELS_PER_PAGE]
        fig = plt.figure(figsize=(ncols*4.5, nrows*4.2))
        fig.suptitle(f"Correction SVD rang-{SVD_RANK} — page {page+1}/{n_pages}", fontsize=12)

        for ki, Z in enumerate(zp):
            col      = ki % ncols
            row_base = (ki // ncols) * 2
            in_train = len(Z_valid) > 0 and Z in Z_valid
            in_data  = Z in bin_count and np.any(bin_count[Z] > 0)
            label    = " [train]" if in_train else (" [data]" if in_data else " [extrap]")

            ax_top = fig.add_subplot(nrows*2, ncols, row_base*ncols+col+1)
            ax_bot = fig.add_subplot(nrows*2, ncols, (row_base+1)*ncols+col+1, sharex=ax_top)
            ax_top.set_title(f"Z={Z}{label}", fontsize=8)

            # Tables depuis disque
            res_c = load_corr(Z)
            res_s = load_svd(Z)

            if res_c is not None:
                kc, dc = res_c
                ax_top.plot(kc, dc, color="steelblue", lw=1.2, label="lisse (v6)")
            if res_s is not None:
                ks2, ds2 = res_s
                ax_top.plot(ks2, ds2, color="darkorange", lw=1.4, ls="--", label="corrigee SVD")

            # Points bruts
            if Z in bin_count and np.any(bin_count[Z] > 0):
                hZ2 = bin_count[Z] > 0
                last2 = int(np.where(hZ2)[0][-1])+1
                kr2 = kea_centers[:last2]; mr2 = mean_raw[Z][:last2]*DEDX_CONV; cr2 = bin_count[Z][:last2]
                hv = (cr2>=1) & np.isfinite(mr2)
                if np.any(hv):
                    ax_top.plot(kr2[hv], mr2[hv], ".", color="green", ms=2, alpha=0.5,
                                label="points bruts", zorder=5)

            ax_top.set_xscale("log"); ax_top.set_yscale("log")
            ax_top.set_ylabel("Stot (MeV/mg/cm²)", fontsize=6)
            ax_top.tick_params(labelsize=6, labelbottom=False)
            if ax_top.get_legend_handles_labels()[0]:
                ax_top.legend(fontsize=5, loc="upper right")

            # Résidu %
            if Z in bin_count and np.any(bin_count[Z] > 0):
                hZ2 = bin_count[Z] > 0
                last2 = int(np.where(hZ2)[0][-1])+1
                kr2 = kea_centers[:last2]; mr2 = mean_raw[Z][:last2]*DEDX_CONV; cr2 = bin_count[Z][:last2]
                hv = (cr2>=1) & np.isfinite(mr2) & (mr2>0)
                if np.any(hv):
                    if res_c is not None:
                        kc, dc = res_c
                        sj = np.interp(kr2[hv], kc, dc)
                        ax_bot.plot(kr2[hv], 100*(sj/mr2[hv]-1), ".", color="steelblue",
                                    ms=2, alpha=0.5, label="lisse")
                    if res_s is not None:
                        ks2, ds2 = res_s
                        cj = np.interp(kr2[hv], ks2, ds2)
                        ax_bot.plot(kr2[hv], 100*(cj/mr2[hv]-1), ".", color="darkorange",
                                    ms=2, alpha=0.7, label="corrigee")
                    ax_bot.axhline(0, color="k", lw=0.7)
                    ax_bot.axhline( 5, color="gray", lw=0.5, ls=":")
                    ax_bot.axhline(-5, color="gray", lw=0.5, ls=":")
                    ax_bot.set_ylim(-20, 20)
                    ax_bot.legend(fontsize=5)

            ax_bot.set_xscale("log")
            ax_bot.set_xlabel("KE/A (MeV/u)", fontsize=6)
            ax_bot.set_ylabel("Residu (%)", fontsize=6)
            ax_bot.tick_params(labelsize=6)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

print(f"  PDF : {PDF_OUT}")

print("\n" + "="*70)
print("Tout termine.")
print("="*70)
