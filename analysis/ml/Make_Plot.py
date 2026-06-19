"""
generate_report_figures.py
===========================
Génère les figures pour le rapport (PDF vectoriels séparés dans ./figures/) :

Z   : matrice de confusion, distribution Z_pred (espérance, binning=0.05),
      accuracy exacte vs niveau de bruit sur l'énergie des pads
A   : matrices de confusion A_L/A_H (2 panneaux),
      accuracy exacte A_L vs niveau de bruit
nu  : matrices de confusion nu_L/nu_H (2 panneaux),
      accuracy exacte nu_L et nu_H vs niveau de bruit

Dataset  : ../simulation_output/T2F_G4.h5  (10% labelisé, val seed=42)
Modèles  : ../dann_ae_final_model.pt        (Z)
           ../dann_ae_mass_kin_model_G4.pt  (A)
           nu_heads_model.pt                (nu, têtes sur latent A figé)

Bruit : sigma_noise = (level/100) * GLOBAL_MAX, additif sur traces brutes,
        clip >= 0. Niveaux : 0, 1, 2, 5, 10, 15, 20 (%).
"""

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
H5_TARGET    = '../simulation_output/T2F_G4.h5'
H5_SOURCE    = '../simulation_output/T2F_G4_Corr.h5'
MODEL_Z_PATH = 'dann_ae_final_model.pt'
MODEL_A_PATH = './dann_ae_mass_kin_model_G4.pt'
MODEL_NU_PATH = 'nu_heads_model.pt'
FIG_DIR      = './figures'

PAD_LEN    = 50
VAL_FRAC   = 0.1
BATCH_SIZE = 2048
NOISE_LEVELS = [0, 1, 2, 5, 10, 15, 20]   # % ajouté en plus du bruit de simulation
BASE_NOISE_KEV = 60.0    # bruit déjà présent dans la simulation
REF_MAX_KEV    = 4500.0  # GLOBAL_MAX correspond à 4.5 MeV

DEVICE = ('mps' if torch.backends.mps.is_available() else
          'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Architecture (encodeurs + têtes, structure exacte des checkpoints) ───────
def make_encoder(hidden, latent):
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            dims = [PAD_LEN] + hidden
            layers = []
            for i in range(len(dims) - 1):
                layers += [nn.Linear(dims[i], dims[i+1]),
                           nn.LayerNorm(dims[i+1]), nn.GELU()]
            self.net    = nn.Sequential(*layers)
            self.z_head = nn.Linear(hidden[-1], latent)
        def forward(self, x): return self.z_head(self.net(x))
    return Encoder()

def make_clf(clf_in, hidden, n_cls):
    dims = [clf_in] + hidden
    layers = []
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i+1]), nn.LayerNorm(dims[i+1]), nn.GELU()]
    layers.append(nn.Linear(hidden[-1], n_cls))
    return nn.Sequential(*layers)

def load_filtered(module, state, prefix):
    sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    module.load_state_dict(sub)

# ── Chargement modèle Z ───────────────────────────────────────────────────────
print("Chargement modèle Z...")
ckpt_Z = torch.load(MODEL_Z_PATH, map_location=DEVICE, weights_only=False)
state_Z = ckpt_Z['model_state']
Z_L_unique  = ckpt_Z['Z_L_unique']
trace_max_Z = ckpt_Z['trace_max']

LATENT = 8
enc_L_Z = make_encoder([64,64,64], LATENT); enc_H_Z = make_encoder([64,64,64], LATENT)
load_filtered(enc_L_Z, state_Z, 'encoder_L.')
load_filtered(enc_H_Z, state_Z, 'encoder_H.')
clf_Z = make_clf(LATENT*2, [32,32], len(Z_L_unique))
load_filtered(clf_Z, state_Z, 'clf.')

for m in [enc_L_Z, enc_H_Z, clf_Z]:
    m.to(DEVICE).eval()
    for p in m.parameters(): p.requires_grad = False

z_vals = torch.tensor(Z_L_unique, dtype=torch.float32)

# ── Chargement modèle A ───────────────────────────────────────────────────────
print("Chargement modèle A...")
ckpt_A = torch.load(MODEL_A_PATH, map_location=DEVICE, weights_only=False)
state_A = ckpt_A['model_state']
A_L_unique  = ckpt_A['A_L_unique']
A_H_unique  = ckpt_A['A_H_unique']
trace_max_A = ckpt_A['trace_max']

enc_L_A = make_encoder([256,128,64], LATENT); enc_H_A = make_encoder([256,128,64], LATENT)
load_filtered(enc_L_A, state_A, 'encoder_L.')
load_filtered(enc_H_A, state_A, 'encoder_H.')
clf_AL = make_clf(LATENT*2+1, [64,64], len(A_L_unique))
clf_AH = make_clf(LATENT*2+1, [64,64], len(A_H_unique))
load_filtered(clf_AL, state_A, 'clf_AL.')
load_filtered(clf_AH, state_A, 'clf_AH.')

for m in [enc_L_A, enc_H_A, clf_AL, clf_AH]:
    m.to(DEVICE).eval()
    for p in m.parameters(): p.requires_grad = False

a_vals_L = torch.tensor(A_L_unique, dtype=torch.float32)
a_vals_H = torch.tensor(A_H_unique, dtype=torch.float32)

# ── Chargement têtes nu (sur encodeur A figé) ─────────────────────────────────
print("Chargement têtes nu...")
ckpt_nu = torch.load(MODEL_NU_PATH, map_location=DEVICE, weights_only=False)
N_CLASSES = ckpt_nu['N_CLASSES']
HIDDEN_CLF_NU = ckpt_nu['HIDDEN_CLF']
clf_nuL = make_clf(LATENT*2, HIDDEN_CLF_NU, N_CLASSES)
clf_nuH = make_clf(LATENT*2, HIDDEN_CLF_NU, N_CLASSES)
clf_nuL.load_state_dict(ckpt_nu['clf_nuL_state'])
clf_nuH.load_state_dict(ckpt_nu['clf_nuH_state'])
for m in [clf_nuL, clf_nuH]:
    m.to(DEVICE).eval()
    for p in m.parameters(): p.requires_grad = False

print(f"  Z_L : {Z_L_unique.min()}..{Z_L_unique.max()}")
print(f"  A_L : {A_L_unique.min()}..{A_L_unique.max()}  A_H : {A_H_unique.min()}..{A_H_unique.max()}")
print(f"  nu  : 0..{N_CLASSES-1}")

# ── Chargement données ────────────────────────────────────────────────────────
print(f"\nChargement {H5_TARGET}...")
with h5py.File(H5_TARGET, 'r') as f:
    total     = f['L/Z'].shape[0]
    n_labeled = int(total * 0.1)
    sl        = slice(0, n_labeled)
    raw_L = f['L/raw_trace'][sl]
    raw_H = f['H/raw_trace'][sl]
    true_Z_L = f['L/Z'][sl].astype(int)
    true_A_L = f['L/A'][sl].astype(int)
    true_A_H = f['H/A'][sl].astype(int)
    true_nu_L = np.clip(f['L/n'][sl].astype(int), 0, N_CLASSES-1)
    true_nu_H = np.clip(f['H/n'][sl].astype(int), 0, N_CLASSES-1)

def pad_traces(raw):
    N = len(raw)
    t = np.zeros((N, PAD_LEN), dtype=np.float32)
    for i, tr in enumerate(raw):
        k = min(len(tr), PAD_LEN)
        t[i, :k] = tr[:k]
    return t

tL_raw = pad_traces(raw_L)
tH_raw = pad_traces(raw_H)

rng     = np.random.default_rng(42)
idx_all = rng.permutation(n_labeled)
n_val   = int(n_labeled * VAL_FRAC)
idx_val = idx_all[-n_val:]
N_VAL   = len(idx_val)
print(f"  n_labeled={n_labeled}  N_VAL={N_VAL}")

GLOBAL_MAX = max(trace_max_Z, trace_max_A)
print(f"  trace_max_Z={trace_max_Z}  trace_max_A={trace_max_A}  GLOBAL_MAX={GLOBAL_MAX}")

# Niveau de bruit total (%) = sqrt(60keV^2 + sigma_add^2) / 4.5MeV * 100
NOISE_TOTAL_PCT = [
    100.0 * np.sqrt(BASE_NOISE_KEV**2 + (lvl/100.0 * REF_MAX_KEV)**2) / REF_MAX_KEV
    for lvl in NOISE_LEVELS
]
print(f"  Niveaux bruit ajouté (%) : {NOISE_LEVELS}")
print(f"  Niveaux bruit total  (%) : {[f'{x:.2f}' for x in NOISE_TOTAL_PCT]}")

def to_gpu(arr): return torch.from_numpy(arr).to(DEVICE)

# ── Fonctions d'inférence ──────────────────────────────────────────────────────
def infer_Z(tL_n, tH_n):
    """tL_n, tH_n normalisés par trace_max_Z. Retourne Z_pred (espérance) et argmax."""
    all_logits = []
    with torch.no_grad():
        for s in range(0, len(tL_n), BATCH_SIZE):
            sl = slice(s, s+BATCH_SIZE)
            zL = enc_L_Z(to_gpu(tL_n[sl])); zH = enc_H_Z(to_gpu(tH_n[sl]))
            all_logits.append(clf_Z(torch.cat([zL, zH], dim=1)).cpu())
    logits = torch.cat(all_logits)
    proba  = torch.softmax(logits, dim=1)
    Z_reg  = (proba * z_vals).sum(dim=1).numpy()
    Z_arg  = Z_L_unique[proba.argmax(dim=1).numpy()]
    return Z_reg, Z_arg

def infer_A(tL_n, tH_n, akin_L, akin_H):
    """tL_n, tH_n normalisés par trace_max_A.
    Retourne AL_arg, AH_arg (argmax) et AL_reg, AH_reg (espérance softmax)."""
    all_AL, all_AH = [], []
    with torch.no_grad():
        for s in range(0, len(tL_n), BATCH_SIZE):
            sl = slice(s, s+BATCH_SIZE)
            zL = enc_L_A(to_gpu(tL_n[sl])); zH = enc_H_A(to_gpu(tH_n[sl]))
            z  = torch.cat([zL, zH], dim=1)
            akL = to_gpu(akin_L[sl]); akH = to_gpu(akin_H[sl])
            lAL = clf_AL(torch.cat([z, akL/252.], dim=1))
            lAH = clf_AH(torch.cat([z, akH/252.], dim=1))
            all_AL.append(lAL.cpu()); all_AH.append(lAH.cpu())
    logits_AL = torch.cat(all_AL); logits_AH = torch.cat(all_AH)
    proba_AL  = torch.softmax(logits_AL, dim=1)
    proba_AH  = torch.softmax(logits_AH, dim=1)
    AL_arg = A_L_unique[proba_AL.argmax(dim=1).numpy()]
    AH_arg = A_H_unique[proba_AH.argmax(dim=1).numpy()]
    AL_reg = (proba_AL * a_vals_L).sum(dim=1).numpy()
    AH_reg = (proba_AH * a_vals_H).sum(dim=1).numpy()
    return AL_arg, AH_arg, AL_reg, AH_reg

def infer_nu(tL_n, tH_n):
    """tL_n, tH_n normalisés par trace_max_A (encodeur A). Retourne nuL_arg, nuH_arg."""
    all_L, all_H = [], []
    with torch.no_grad():
        for s in range(0, len(tL_n), BATCH_SIZE):
            sl = slice(s, s+BATCH_SIZE)
            zL = enc_L_A(to_gpu(tL_n[sl])); zH = enc_H_A(to_gpu(tH_n[sl]))
            z  = torch.cat([zL, zH], dim=1)
            all_L.append(clf_nuL(z).argmax(dim=1).cpu())
            all_H.append(clf_nuH(z).argmax(dim=1).cpu())
    return torch.cat(all_L).numpy(), torch.cat(all_H).numpy()

def compute_akin(tL, tH):
    KE_L = tL.sum(axis=1); KE_H = tH.sum(axis=1)
    denom = np.clip(KE_L + KE_H, 1e-6, None)
    akin_L = (252.0 * KE_H / denom).astype(np.float32)[:, None]
    akin_H = (252.0 * KE_L / denom).astype(np.float32)[:, None]
    return akin_L, akin_H

def add_noise(trace_raw, level_pct, ref_max, rng_local):
    if level_pct == 0:
        return trace_raw.copy()
    sigma = (level_pct / 100.0) * ref_max
    noisy = trace_raw + rng_local.normal(0, sigma, trace_raw.shape).astype(np.float32)
    return np.clip(noisy, 0, None)

# ── Baseline (sans bruit) — confusion matrices et distribution ──────────────
print("\nInférence baseline (sans bruit)...")
tL_val_raw = tL_raw[idx_val]; tH_val_raw = tH_raw[idx_val]
true_ZL = true_Z_L[idx_val]; true_AL = true_A_L[idx_val]; true_AH = true_A_H[idx_val]
true_nuL = true_nu_L[idx_val]; true_nuH = true_nu_H[idx_val]

tLZ = tL_val_raw / trace_max_Z; tHZ = tH_val_raw / trace_max_Z
tLA = tL_val_raw / trace_max_A; tHA = tH_val_raw / trace_max_A
akin_L, akin_H = compute_akin(tL_val_raw, tH_val_raw)

Z_reg, Z_arg = infer_Z(tLZ, tHZ)

# Extraction latent z_L (2 premières coordonnées) pour scatter coloré par Z
with torch.no_grad():
    zL_latent = enc_L_Z(to_gpu(tLZ)).cpu().numpy()   # (N_VAL, LATENT_DIM)
AL_arg, AH_arg, AL_reg, AH_reg = infer_A(tLA, tHA, akin_L, akin_H)
nuL_arg, nuH_arg = infer_nu(tLA, tHA)

# Distribution Z sur toute la stat labelisée (n_labeled)
print("Inférence distribution Z (toute la stat labelisée)...")
tLZ_full = tL_raw[:n_labeled] / trace_max_Z
tHZ_full = tH_raw[:n_labeled] / trace_max_Z
Z_reg_full, _ = infer_Z(tLZ_full, tHZ_full)

# ── FIGURES Z ─────────────────────────────────────────────────────────────────
print("\nGénération figures Z...")

# Fig Z : matrice de confusion
z_range  = np.arange(true_ZL.min(), true_ZL.max()+1)
z_to_row = {z: i for i, z in enumerate(z_range)}
cm = np.zeros((len(z_range), len(z_range)), dtype=int)
for t, p in zip(true_ZL, Z_arg):
    if t in z_to_row and p in z_to_row:
        cm[z_to_row[t], z_to_row[p]] += 1
cm_n = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
acc_Z = (Z_arg == true_ZL).mean()

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm_n, cmap='Blues', vmin=0, vmax=1, aspect='auto')
step = max(1, len(z_range)//20)
ax.set_xticks(range(0, len(z_range), step)); ax.set_xticklabels(z_range[::step], rotation=90, fontsize=8)
ax.set_yticks(range(0, len(z_range), step)); ax.set_yticklabels(z_range[::step], fontsize=8)
ax.set_xlabel('Z prédit'); ax.set_ylabel('Z vrai')
ax.set_title(f'Matrice de confusion Z_L — acc={acc_Z:.3f}', fontsize=11)
plt.colorbar(im, ax=ax)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_Z_confusion.pdf')
plt.close(fig)

# Fig Z : distribution Z_pred (espérance, binning=0.05) — baseline vs bruit ajouté 5%
print("Inférence distribution Z avec bruit ajouté 5%...")
rng_d = np.random.default_rng(9999)
tL_full_raw = tL_raw[:n_labeled]; tH_full_raw = tH_raw[:n_labeled]
ADD_LVL = 5
tL_n2 = add_noise(tL_full_raw, ADD_LVL, GLOBAL_MAX, rng_d) / trace_max_Z
tH_n2 = add_noise(tH_full_raw, ADD_LVL, GLOBAL_MAX, rng_d) / trace_max_Z
Z_reg_full_n2, _ = infer_Z(tL_n2, tH_n2)

total_pct_0   = 100.0 * BASE_NOISE_KEV / REF_MAX_KEV
total_pct_add = 100.0 * np.sqrt(BASE_NOISE_KEV**2 + (ADD_LVL/100.0*REF_MAX_KEV)**2) / REF_MAX_KEV

fig, ax = plt.subplots(figsize=(10, 5))
bins = np.arange(Z_L_unique.min()-0.5, Z_L_unique.max()+0.55, 0.05)
ax.hist(Z_reg_full,    bins=bins, color='coral',     alpha=0.6,
        label=f'Bruit simu seul ({total_pct_0:.1f}%)')
ax.hist(Z_reg_full_n2, bins=bins, color='steelblue', alpha=0.6,
        label=f'Bruit simu + {ADD_LVL}% ajouté ({total_pct_add:.1f}% total)')
ax.set_xlabel('Z_L prédit (espérance softmax)')
ax.set_ylabel('N events')
ax.set_title(f'Distribution de Z_L reconstruit (N={n_labeled:,})', fontsize=11)
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_Z_distribution.pdf')
plt.close(fig)

# Fig Z : scatter latent (2 premières coordonnées) coloré par Z vrai
fig, ax = plt.subplots(figsize=(8, 7))
sc = ax.scatter(zL_latent[:, 0], zL_latent[:, 1], c=true_ZL, cmap='viridis',
                 s=4, alpha=0.5)
ax.set_xlabel('z_L[0]'); ax.set_ylabel('z_L[1]')
ax.set_title('Espace latent (encodeur Z, fragment L)\ncoloré par Z_L vrai', fontsize=11)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Z_L vrai')
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_Z_latent.pdf')
plt.close(fig)

# Fig Z : domain adaptation — distribution z_L[0] et z_L[1] source vs cible, Z=32 et 44
print("\nChargement source pour domain adaptation...")
with h5py.File(H5_SOURCE, 'r') as f:
    raw_L_src = f['L/raw_trace'][:]
    raw_H_src = f['H/raw_trace'][:]
    Z_L_src   = f['L/Z'][:].astype(int)

tL_src_raw = pad_traces(raw_L_src)
tL_src = tL_src_raw / trace_max_Z

Z_TARGETS = [32, 44]
with torch.no_grad():
    zL_latent_src_all = enc_L_Z(to_gpu(tL_src)).cpu().numpy()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for row, z_val in enumerate(Z_TARGETS):
    mask_src = Z_L_src == z_val
    mask_tgt = true_ZL == z_val
    for col in range(2):
        ax = axes[row, col]
        data_src = zL_latent_src_all[mask_src, col]
        data_tgt = zL_latent[mask_tgt, col]
        bins = np.linspace(
            min(data_src.min(), data_tgt.min()),
            max(data_src.max(), data_tgt.max()), 40)
        ax.hist(data_src, bins=bins, alpha=0.5, color='steelblue', density=True,
                label=f'Source (N={mask_src.sum()})')
        ax.hist(data_tgt, bins=bins, alpha=0.5, color='coral', density=True,
                label=f'Cible (N={mask_tgt.sum()})')
        ax.set_xlabel(f'z_L[{col}]'); ax.set_ylabel('Densité')
        ax.set_title(f'Z_L = {z_val}  —  z_L[{col}]', fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
fig.suptitle('Adaptation de domaine — distributions latentes source vs cible', fontsize=12)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_Z_domain_adapt.pdf')
plt.close(fig)

# Fig Z : accuracy exacte vs bruit
print("Scan bruit Z...")
acc_Z_noise = []
for lvl in NOISE_LEVELS:
    rng_n = np.random.default_rng(1000 + lvl)
    tL_n = add_noise(tL_val_raw, lvl, GLOBAL_MAX, rng_n) / trace_max_Z
    tH_n = add_noise(tH_val_raw, lvl, GLOBAL_MAX, rng_n) / trace_max_Z
    _, Z_arg_n = infer_Z(tL_n, tH_n)
    acc_Z_noise.append((Z_arg_n == true_ZL).mean())
    print(f"  bruit={lvl:2d}%  acc_Z={acc_Z_noise[-1]:.3f}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(NOISE_TOTAL_PCT, acc_Z_noise, 'b-o')
ax.set_xlabel('Bruit total sur énergie des pads (%)')
ax.set_ylabel('Accuracy exacte Z_L')
ax.set_title('Résolution Z_L vs bruit total sur énergie des pads', fontsize=11)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_Z_noise.pdf')
plt.close(fig)

# Fig Z : pureté (fenêtre ±0.3, espérance) vs bruit total
print("Scan pureté Z vs bruit...")
HALF_WIN_Z = 0.3
true_ZL_full = true_Z_L[:n_labeled]
purity_Z_noise = []
for lvl in NOISE_LEVELS:
    rng_n = np.random.default_rng(4000 + lvl)
    tL_n = add_noise(tL_full_raw, lvl, GLOBAL_MAX, rng_n) / trace_max_Z
    tH_n = add_noise(tH_full_raw, lvl, GLOBAL_MAX, rng_n) / trace_max_Z
    Z_reg_n, _ = infer_Z(tL_n, tH_n)
    Z_round = np.round(Z_reg_n)
    captured = np.abs(Z_reg_n - Z_round) <= HALF_WIN_Z
    purity = (Z_round[captured] == true_ZL_full[captured]).mean()
    purity_Z_noise.append(purity)
    print(f"  bruit={lvl:2d}%  pureté_Z={purity:.3f}  "
          f"(coverage={captured.mean():.3f})")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(NOISE_TOTAL_PCT, purity_Z_noise, 'g-o')
ax.set_xlabel('Bruit total sur énergie des pads (%)')
ax.set_ylabel(f'Pureté (fenêtre ±{HALF_WIN_Z})')
ax.set_title(f'Pureté Z_L (Z_pred∈[Z±{HALF_WIN_Z}]) vs bruit total', fontsize=11)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_Z_purity_noise.pdf')
plt.close(fig)

# ── FIGURES A ─────────────────────────────────────────────────────────────────
print("\nGénération figures A...")

# Fig A : matrices de confusion A_L et A_H (2 panneaux)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, true_A, pred_A, A_unique, title in [
    (axes[0], true_AL, AL_arg, A_L_unique, 'A_L'),
    (axes[1], true_AH, AH_arg, A_H_unique, 'A_H'),
]:
    a_range  = np.arange(true_A.min(), true_A.max()+1)
    a_to_row = {a: i for i, a in enumerate(a_range)}
    cm = np.zeros((len(a_range), len(a_range)), dtype=int)
    for t, p in zip(true_A, pred_A):
        if t in a_to_row and p in a_to_row:
            cm[a_to_row[t], a_to_row[p]] += 1
    cm_n = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    acc  = (pred_A == true_A).mean()
    im = ax.imshow(cm_n, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    step = max(1, len(a_range)//20)
    ax.set_xticks(range(0, len(a_range), step)); ax.set_xticklabels(a_range[::step], rotation=90, fontsize=7)
    ax.set_yticks(range(0, len(a_range), step)); ax.set_yticklabels(a_range[::step], fontsize=7)
    ax.set_xlabel(f'{title} prédit'); ax.set_ylabel(f'{title} vrai')
    ax.set_title(f'{title} — acc={acc:.3f}', fontsize=11)
    plt.colorbar(im, ax=ax)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_A_confusion.pdf')
plt.close(fig)

# Fig A : sigma régression (espérance) A_L vs bruit
print("Scan bruit A...")
sigma_AL_noise = []
for lvl in NOISE_LEVELS:
    rng_n = np.random.default_rng(2000 + lvl)
    tL_n_raw = add_noise(tL_val_raw, lvl, GLOBAL_MAX, rng_n)
    tH_n_raw = add_noise(tH_val_raw, lvl, GLOBAL_MAX, rng_n)
    akL_n, akH_n = compute_akin(tL_n_raw, tH_n_raw)
    tL_n = tL_n_raw / trace_max_A; tH_n = tH_n_raw / trace_max_A
    _, _, AL_reg_n, _ = infer_A(tL_n, tH_n, akL_n, akH_n)
    sigma_AL_noise.append((AL_reg_n - true_AL).std())
    print(f"  bruit={lvl:2d}%  sigma_A_L={sigma_AL_noise[-1]:.3f}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(NOISE_TOTAL_PCT, sigma_AL_noise, 'r-o')
ax.set_xlabel('Bruit total sur énergie des pads (%)')
ax.set_ylabel('σ(A_L) — régression espérance')
ax.set_title('Résolution A_L (régression) vs bruit total sur énergie des pads', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_A_noise.pdf')
plt.close(fig)

# ── FIGURES NU ────────────────────────────────────────────────────────────────
print("\nGénération figures nu...")

# Fig nu : matrices de confusion nu_L et nu_H (2 panneaux)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, true_nu, pred_nu, title in [
    (axes[0], true_nuL, nuL_arg, 'nu_L'),
    (axes[1], true_nuH, nuH_arg, 'nu_H'),
]:
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for t, p in zip(true_nu, pred_nu):
        cm[t, p] += 1
    cm_n = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    acc  = (pred_nu == true_nu).mean()
    pm1  = (np.abs(pred_nu - true_nu) <= 1).mean()
    im = ax.imshow(cm_n, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
    ax.set_xlabel(f'{title} prédit'); ax.set_ylabel(f'{title} vrai')
    ax.set_title(f'{title} — acc={acc:.3f}  ±1={pm1:.3f}', fontsize=11)
    plt.colorbar(im, ax=ax)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_nu_confusion.pdf')
plt.close(fig)

# Fig nu : accuracy exacte nu_L et nu_H vs bruit
print("Scan bruit nu...")
acc_nuL_noise, acc_nuH_noise = [], []
for lvl in NOISE_LEVELS:
    rng_n = np.random.default_rng(3000 + lvl)
    tL_n = add_noise(tL_val_raw, lvl, GLOBAL_MAX, rng_n) / trace_max_A
    tH_n = add_noise(tH_val_raw, lvl, GLOBAL_MAX, rng_n) / trace_max_A
    nuL_arg_n, nuH_arg_n = infer_nu(tL_n, tH_n)
    acc_nuL_noise.append((nuL_arg_n == true_nuL).mean())
    acc_nuH_noise.append((nuH_arg_n == true_nuH).mean())
    print(f"  bruit={lvl:2d}%  acc_nuL={acc_nuL_noise[-1]:.3f}  acc_nuH={acc_nuH_noise[-1]:.3f}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(NOISE_TOTAL_PCT, acc_nuL_noise, 'b-o', label='nu_L')
ax.plot(NOISE_TOTAL_PCT, acc_nuH_noise, 'r-o', label='nu_H')
ax.set_xlabel('Bruit total sur énergie des pads (%)')
ax.set_ylabel('Accuracy exacte')
ax.set_title('Résolution nu_L / nu_H vs bruit total sur énergie des pads', fontsize=11)
ax.set_ylim(0, 1)
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_nu_noise.pdf')
plt.close(fig)

print(f"\nFigures générées dans {FIG_DIR}/ :")
for fn in sorted(os.listdir(FIG_DIR)):
    print(f"  {fn}")
