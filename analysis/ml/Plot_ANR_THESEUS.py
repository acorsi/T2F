"""
generate_Z_nuL_figure.py
=========================
Génère UNE figure à deux panneaux (a, b) pour le dossier :
  (a) distribution de Z_L reconstruit (prédiction ML) vs Z_L vrai
  (b) distribution de nu_L reconstruit (prédiction ML) vs nu_L vrai

Sobre, sans titre au-dessus des graphiques (labels (a)/(b) intégrés
dans les axes). Sortie : PDF vectoriel unique dans ./figures/.

Réutilise les chemins / architectures de generate_report_figures.py.
"""

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

# ── Config ────────────────────────────────────────────────────────────────
H5_TARGET     = '../../simulation_output/T2F_G4_native.h5'
MODEL_Z_PATH  = 'dann_ae_final_model.pt'
MODEL_A_PATH  = './dann_ae_mass_kin_model_G4.pt'
MODEL_NU_PATH = 'nu_heads_model.pt'
FIG_DIR       = './figures'
FIG_NAME      = 'fig_Z_nuL_panels.png'

PAD_LEN    = 50
VAL_FRAC   = 0.1
BATCH_SIZE = 2048

DEVICE = ('mps' if torch.backends.mps.is_available() else
          'cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Style sobre ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 14.3,             # 11 * 1.3 (axes, labels)
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'legend.frameon': False,
})
LEGEND_FONTSIZE = 9 * 1.5

# ── Architecture (identique au script source) ───────────────────────────
def make_encoder(hidden, latent):
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            dims = [PAD_LEN] + hidden
            layers = []
            for i in range(len(dims) - 1):
                layers += [nn.Linear(dims[i], dims[i + 1]),
                           nn.LayerNorm(dims[i + 1]), nn.GELU()]
            self.net = nn.Sequential(*layers)
            self.z_head = nn.Linear(hidden[-1], latent)
        def forward(self, x): return self.z_head(self.net(x))
    return Encoder()

def make_clf(clf_in, hidden, n_cls):
    dims = [clf_in] + hidden
    layers = []
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.LayerNorm(dims[i + 1]), nn.GELU()]
    layers.append(nn.Linear(hidden[-1], n_cls))
    return nn.Sequential(*layers)

def load_filtered(module, state, prefix):
    sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    module.load_state_dict(sub)

def to_gpu(arr): return torch.from_numpy(arr).to(DEVICE)

def pad_traces(raw):
    N = len(raw)
    t = np.zeros((N, PAD_LEN), dtype=np.float32)
    for i, tr in enumerate(raw):
        k = min(len(tr), PAD_LEN)
        t[i, :k] = tr[:k]
    return t

LATENT = 8

# ── Modèle Z ──────────────────────────────────────────────────────────────
print("Chargement modèle Z...")
ckpt_Z = torch.load(MODEL_Z_PATH, map_location=DEVICE, weights_only=False)
state_Z = ckpt_Z['model_state']
Z_L_unique  = ckpt_Z['Z_L_unique']
trace_max_Z = ckpt_Z['trace_max']

enc_L_Z = make_encoder([64, 64, 64], LATENT)
enc_H_Z = make_encoder([64, 64, 64], LATENT)
load_filtered(enc_L_Z, state_Z, 'encoder_L.')
load_filtered(enc_H_Z, state_Z, 'encoder_H.')
clf_Z = make_clf(LATENT * 2, [32, 32], len(Z_L_unique))
load_filtered(clf_Z, state_Z, 'clf.')

for m in [enc_L_Z, enc_H_Z, clf_Z]:
    m.to(DEVICE).eval()
    for p in m.parameters():
        p.requires_grad = False

z_vals = torch.tensor(Z_L_unique, dtype=torch.float32)

# ── Modèle A (nécessaire à l'encodeur utilisé par les têtes nu) ─────────
print("Chargement modèle A...")
ckpt_A = torch.load(MODEL_A_PATH, map_location=DEVICE, weights_only=False)
state_A = ckpt_A['model_state']
trace_max_A = ckpt_A['trace_max']

enc_L_A = make_encoder([256, 128, 64], LATENT)
enc_H_A = make_encoder([256, 128, 64], LATENT)
load_filtered(enc_L_A, state_A, 'encoder_L.')
load_filtered(enc_H_A, state_A, 'encoder_H.')

for m in [enc_L_A, enc_H_A]:
    m.to(DEVICE).eval()
    for p in m.parameters():
        p.requires_grad = False

# ── Têtes nu (sur encodeur A figé) ───────────────────────────────────────
print("Chargement têtes nu...")
ckpt_nu = torch.load(MODEL_NU_PATH, map_location=DEVICE, weights_only=False)
N_CLASSES = ckpt_nu['N_CLASSES']
HIDDEN_CLF_NU = ckpt_nu['HIDDEN_CLF']
clf_nuL = make_clf(LATENT * 2, HIDDEN_CLF_NU, N_CLASSES)
clf_nuL.load_state_dict(ckpt_nu['clf_nuL_state'])
clf_nuL.to(DEVICE).eval()
for p in clf_nuL.parameters():
    p.requires_grad = False

# ── Données ───────────────────────────────────────────────────────────────
print(f"Chargement {H5_TARGET}...")
with h5py.File(H5_TARGET, 'r') as f:
    total     = f['L/Z'].shape[0]
    n_labeled = int(total * 0.1)
    sl        = slice(0, n_labeled)
    raw_L = f['L/raw_trace'][sl]
    raw_H = f['H/raw_trace'][sl]
    true_Z_L  = f['L/Z'][sl].astype(int)
    true_nu_L = np.clip(f['L/n'][sl].astype(int), 0, N_CLASSES - 1)

tL_raw = pad_traces(raw_L)
tH_raw = pad_traces(raw_H)

rng     = np.random.default_rng(42)
idx_all = rng.permutation(n_labeled)
n_val   = int(n_labeled * VAL_FRAC)
idx_val = idx_all[-n_val:]

tL_val_raw = tL_raw[:n_labeled]
tH_val_raw = tH_raw[:n_labeled]
true_ZL  = true_Z_L[:n_labeled]
true_nuL = true_nu_L[:n_labeled]

# ── Inférence Z (espérance softmax) ──────────────────────────────────────
def infer_Z(tL_n, tH_n):
    all_logits = []
    with torch.no_grad():
        for s in range(0, len(tL_n), BATCH_SIZE):
            b = slice(s, s + BATCH_SIZE)
            zL = enc_L_Z(to_gpu(tL_n[b]))
            zH = enc_H_Z(to_gpu(tH_n[b]))
            all_logits.append(clf_Z(torch.cat([zL, zH], dim=1)).cpu())
    logits = torch.cat(all_logits)
    proba  = torch.softmax(logits, dim=1)
    return (proba * z_vals).sum(dim=1).numpy()

# ── Inférence nu_L (argmax) ───────────────────────────────────────────────
def infer_nuL(tL_n, tH_n):
    all_L = []
    with torch.no_grad():
        for s in range(0, len(tL_n), BATCH_SIZE):
            b = slice(s, s + BATCH_SIZE)
            zL = enc_L_A(to_gpu(tL_n[b]))
            zH = enc_H_A(to_gpu(tH_n[b]))
            z  = torch.cat([zL, zH], dim=1)
            all_L.append(clf_nuL(z).argmax(dim=1).cpu())
    return torch.cat(all_L).numpy()

print("Inférence...")
Z_pred  = infer_Z(tL_val_raw / trace_max_Z, tH_val_raw / trace_max_Z)
nuL_pred = infer_nuL(tL_val_raw / trace_max_A, tH_val_raw / trace_max_A)

# ── Figure à deux panneaux ────────────────────────────────────────────────
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.5))

# (a) Distribution de Z_L prédit (ML) — probabilité (%), ticks Z pairs uniquement
bins_z = np.arange(true_ZL.min() - 0.5, true_ZL.max() + 1.5, 0.1)
ax_a.hist(Z_pred, bins=bins_z, color='steelblue', alpha=0.85,
          label='ML prediction', histtype='stepfilled',
          weights=100 * np.ones_like(Z_pred) / len(Z_pred))
ax_a.set_xlabel(r'$Z_L$')
ax_a.set_ylabel('Probability (%)')
ax_a.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
ax_a.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
ax_a.set_xlim(33, 50)
ax_a.legend(loc='upper right', fontsize=LEGEND_FONTSIZE)
ax_a.text(0.03, 0.93, '(a)', transform=ax_a.transAxes,
          fontsize=13, fontweight='bold', va='top')

# (b) Distribution de nu_L : vrai vs prédit (prédiction en transparence) — probabilité (%)
bins_nu = np.arange(-0.5, N_CLASSES + 0.5, 1)
ax_b.hist(true_nuL, bins=bins_nu, color='0.75', label='True', histtype='stepfilled',
          weights=100 * np.ones_like(true_nuL, dtype=float) / len(true_nuL))
ax_b.hist(nuL_pred, bins=bins_nu, color='indianred', alpha=0.4,
          label='ML prediction', histtype='stepfilled',
          weights=100 * np.ones_like(nuL_pred, dtype=float) / len(nuL_pred))
ax_b.set_xlabel(r'$\nu_L$')
ax_b.set_ylabel('Probability (%)')
ax_b.set_xlim(0, 6.5)
ax_b.legend(loc='upper right', fontsize=LEGEND_FONTSIZE)
ax_b.text(0.03, 0.93, '(b)', transform=ax_b.transAxes,
          fontsize=13, fontweight='bold', va='top')

plt.tight_layout()
fig.savefig(f'{FIG_DIR}/{FIG_NAME}', dpi=300)
plt.close(fig)

print(f"Figure sauvegardée : {FIG_DIR}/{FIG_NAME}")
