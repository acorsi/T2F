"""
diagnose_traces.py
==================
Affichage interactif des traces H5 vs templates pour diagnostic.
Clic ou touche droite/gauche pour passer à l'event suivant.
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # backend interactif
import time
import sys

# ── Config ────────────────────────────────────────────────────────────────────
H5_TARGET      = '../../simulation_output/T2F_G4.h5'
MODEL_Z_PATH   = '../dann_ae_final_model.pt'
MODEL_A_PATH   = '../dann_ae_mass_kin_model_G4.pt'
TEMPLATES_PATH = './templates_full.npz'

PAD_LEN      = 50
N_SHOW       = 200    # nombre d'events à parcourir
VAL_FRAC     = 0.1
BATCH_SIZE   = 2048
Z_SUM        = 98
N_A_CANDS    = 5
A_OFFSET     = 2

DEVICE = ('mps' if torch.backends.mps.is_available() else
          'cuda' if torch.cuda.is_available() else 'cpu')

# ── Architectures (identiques à bayesian_mass_vec.py) ─────────────────────────
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

def make_decoder(hidden, latent):
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            dims = [latent] + list(reversed(hidden))
            layers = []
            for i in range(len(dims) - 1):
                layers += [nn.Linear(dims[i], dims[i+1]),
                           nn.LayerNorm(dims[i+1]), nn.GELU()]
            layers += [nn.Linear(hidden[0], PAD_LEN), nn.Softplus()]
            self.net = nn.Sequential(*layers)
        def forward(self, z): return self.net(z)
    return Decoder()

def make_disc(hidden, latent):
    class Disc(nn.Module):
        def __init__(self):
            super().__init__()
            dims = [latent * 2] + hidden
            layers = []
            for i in range(len(dims) - 1):
                layers += [nn.Linear(dims[i], dims[i+1]),
                           nn.LayerNorm(dims[i+1]), nn.GELU()]
            layers.append(nn.Linear(hidden[-1], 2))
            self.net = nn.Sequential(*layers)
        def forward(self, x): return self.net(x)
    return Disc()

def make_clf(clf_in, hidden, n_cls):
    dims = [clf_in] + hidden
    layers = []
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i+1]), nn.LayerNorm(dims[i+1]), nn.GELU()]
    layers.append(nn.Linear(hidden[-1], n_cls))
    return nn.Sequential(*layers)

class ModelZ(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        L, HAE, HCLF, HD = 8, [64,64,64], [32,32], [32,32]
        self.encoder_L     = make_encoder(HAE, L)
        self.encoder_H     = make_encoder(HAE, L)
        self.decoder_L     = make_decoder(HAE, L)
        self.decoder_H     = make_decoder(HAE, L)
        self.discriminator = make_disc(HD, L)
        self.clf           = make_clf(L*2, HCLF, n_cls)
    def forward(self, xL, xH):
        return self.clf(torch.cat([self.encoder_L(xL), self.encoder_H(xH)], dim=1))

class ModelA(nn.Module):
    def __init__(self, n_AL, n_AH):
        super().__init__()
        L, HAE, HCLF, HD = 8, [256,128,64], [64,64], [32,32]
        self.encoder_L     = make_encoder(HAE, L)
        self.encoder_H     = make_encoder(HAE, L)
        self.decoder_L     = make_decoder(HAE, L)
        self.decoder_H     = make_decoder(HAE, L)
        self.discriminator = make_disc(HD, L)
        self.clf_AL        = make_clf(L*2+1, HCLF, n_AL)
        self.clf_AH        = make_clf(L*2+1, HCLF, n_AH)
    def forward(self, xL, xH, akL, akH):
        z = torch.cat([self.encoder_L(xL), self.encoder_H(xH)], dim=1)
        return (self.clf_AL(torch.cat([z, akL/252.], dim=1)),
                self.clf_AH(torch.cat([z, akH/252.], dim=1)))

# ── Chargement ────────────────────────────────────────────────────────────────
print("Chargement modèles...")
ckpt_Z = torch.load(MODEL_Z_PATH, map_location=DEVICE, weights_only=False)
ckpt_A = torch.load(MODEL_A_PATH, map_location=DEVICE, weights_only=False)
Z_L_unique = ckpt_Z['Z_L_unique']
A_L_unique = ckpt_A['A_L_unique']
A_H_unique = ckpt_A['A_H_unique']
trace_max_Z = ckpt_Z['trace_max']
trace_max_A = ckpt_A['trace_max']

model_Z = ModelZ(len(Z_L_unique)).to(DEVICE)
model_Z.load_state_dict(ckpt_Z['model_state']); model_Z.eval()
model_A = ModelA(len(A_L_unique), len(A_H_unique)).to(DEVICE)
model_A.load_state_dict(ckpt_A['model_state']); model_A.eval()

z_vals   = torch.tensor(Z_L_unique, dtype=torch.float32)
a_vals_L = torch.tensor(A_L_unique, dtype=torch.float32)
a_vals_H = torch.tensor(A_H_unique, dtype=torch.float32)

print("Chargement templates...")
tpl          = np.load(TEMPLATES_PATH)
tpl_Z        = tpl['Z'].astype(int)
tpl_A        = tpl['A'].astype(int)
tpl_ke       = tpl['ke_center'].astype(np.float32)
tpl_mu       = tpl['mu'].astype(np.float32)
tpl_sigma    = tpl['sigma'].astype(np.float32)
tpl_n_pads   = tpl['n_pads'].astype(int)
KE_BIN_WIDTH = float(tpl['meta_ke_bin_width'])

# Lookup
from collections import defaultdict
ke_bin_arr = np.round(tpl_ke / KE_BIN_WIDTH).astype(int)
lookup = {}
for i in range(len(tpl_Z)):
    key = (tpl_Z[i], tpl_A[i], ke_bin_arr[i])
    lookup[key] = i

print(f"Chargement {H5_TARGET}...")
with h5py.File(H5_TARGET, 'r') as f:
    total     = f['L/Z'].shape[0]
    n_labeled = int(total * 0.1)
    sl        = slice(0, n_labeled)
    raw_L    = f['L/raw_trace'][sl]
    raw_H    = f['H/raw_trace'][sl]
    true_Z_L = f['L/Z'][sl].astype(int)
    true_Z_H = f['H/Z'][sl].astype(int)
    true_A_L = f['L/A'][sl].astype(int)
    true_A_H = f['H/A'][sl].astype(int)

def pad_traces(raw):
    N = len(raw)
    t = np.zeros((N, PAD_LEN), dtype=np.float32)
    for i, tr in enumerate(raw):
        k = min(len(tr), PAD_LEN)
        t[i, :k] = tr[:k]
    return t

tL_raw = pad_traces(raw_L)
tH_raw = pad_traces(raw_H)
KE_L_raw = tL_raw.sum(axis=1)
KE_H_raw = tH_raw.sum(axis=1)
denom    = np.clip(KE_L_raw + KE_H_raw, 1e-6, None)
akin_L   = (252.0 * KE_H_raw / denom).astype(np.float32)[:, None]
akin_H   = (252.0 * KE_L_raw / denom).astype(np.float32)[:, None]

tL_Z = tL_raw / trace_max_Z; tH_Z = tH_raw / trace_max_Z
tL_A = tL_raw / trace_max_A; tH_A = tH_raw / trace_max_A

# Val set
rng     = np.random.default_rng(42)
idx_all = rng.permutation(n_labeled)
n_val   = int(n_labeled * VAL_FRAC)
idx_val = idx_all[-n_val:]

# Inférence sur les N_SHOW premiers events du val set
idx_show = idx_val[:N_SHOW]
N = len(idx_show)

def to_gpu(arr): return torch.from_numpy(arr).to(DEVICE)

print("Inférence réseaux...")
with torch.no_grad():
    lZ = model_Z(to_gpu(tL_Z[idx_show]), to_gpu(tH_Z[idx_show])).cpu()
    lAL, lAH = model_A(to_gpu(tL_A[idx_show]), to_gpu(tH_A[idx_show]),
                       to_gpu(akin_L[idx_show]), to_gpu(akin_H[idx_show]))
    lAL = lAL.cpu(); lAH = lAH.cpu()

Z_pred  = (torch.softmax(lZ,   dim=1) * z_vals  ).sum(dim=1).numpy()
AL_pred = (torch.softmax(lAL,  dim=1) * a_vals_L).sum(dim=1).numpy()
AH_pred = (torch.softmax(lAH,  dim=1) * a_vals_H).sum(dim=1).numpy()

Z_int_L = np.round(Z_pred).astype(int)
Z_int_H = Z_SUM - Z_int_L
AL_int  = np.round(AL_pred).astype(int)
AH_int  = np.round(AH_pred).astype(int)

def get_template(Z, A, KE):
    """Retourne (mu, sigma, ke_used) ou None."""
    keb = int(round(KE / KE_BIN_WIDTH))
    for dk in range(0, 5):
        for sign in ([0] if dk == 0 else [-1, 1]):
            key = (int(Z), int(A), keb + sign*dk)
            if key in lookup:
                idx = lookup[key]
                n_p = tpl_n_pads[idx]
                return (tpl_mu[idx, :n_p],
                        tpl_sigma[idx, :n_p],
                        tpl_ke[idx])
    return None

# ── Affichage interactif ───────────────────────────────────────────────────────
current = [0]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle('', fontsize=11)
plt.subplots_adjust(hspace=0.4, wspace=0.3)

def draw(i):
    ev = idx_show[i]
    for ax in axes.flat:
        ax.cla()

    tr_L = tL_raw[ev]
    tr_H = tH_raw[ev]
    ke_l = KE_L_raw[ev]
    ke_h = KE_H_raw[ev]
    z_l  = Z_int_L[i]; z_h = Z_int_H[i]
    al   = AL_int[i];  ah  = AH_int[i]
    t_zl = true_Z_L[ev]; t_zh = true_Z_H[ev]
    t_al = true_A_L[ev]; t_ah = true_A_H[ev]

    pads_L = np.arange(len(tr_L))
    pads_H = np.arange(len(tr_H))

    # Fragment L
    ax = axes[0, 0]
    ax.step(pads_L, tr_L, where='mid', color='black', lw=1.5, label='Trace H5 (brut)')
    tpl_res = get_template(z_l, al, ke_l)
    if tpl_res is not None:
        mu_t, sig_t, ke_used = tpl_res
        pads_t = np.arange(len(mu_t))
        ax.plot(pads_t, mu_t, 'r-', lw=1.5,
                label=f'Template (Z={z_l},A={al},KE={ke_used:.1f})')
        ax.fill_between(pads_t, mu_t - sig_t, mu_t + sig_t,
                        alpha=0.2, color='red', label='±1σ template')
    else:
        ax.text(0.5, 0.5, 'Template absent', transform=ax.transAxes,
                ha='center', color='red', fontsize=10)
    ax.set_title(f'Fragment L — Z_true={t_zl} Z_pred={z_l}  '
                 f'A_true={t_al} A_pred={al}  KE={ke_l:.1f} MeV',
                 fontsize=8)
    ax.set_xlabel('Pad'); ax.set_ylabel('ADC brut')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Fragment H
    ax = axes[0, 1]
    ax.step(pads_H, tr_H, where='mid', color='black', lw=1.5, label='Trace H5 (brut)')
    tpl_res = get_template(z_h, ah, ke_h)
    if tpl_res is not None:
        mu_t, sig_t, ke_used = tpl_res
        pads_t = np.arange(len(mu_t))
        ax.plot(pads_t, mu_t, 'b-', lw=1.5,
                label=f'Template (Z={z_h},A={ah},KE={ke_used:.1f})')
        ax.fill_between(pads_t, mu_t - sig_t, mu_t + sig_t,
                        alpha=0.2, color='blue', label='±1σ template')
    else:
        ax.text(0.5, 0.5, 'Template absent', transform=ax.transAxes,
                ha='center', color='blue', fontsize=10)
    ax.set_title(f'Fragment H — Z_true={t_zh} Z_pred={z_h}  '
                 f'A_true={t_ah} A_pred={ah}  KE={ke_h:.1f} MeV',
                 fontsize=8)
    ax.set_xlabel('Pad'); ax.set_ylabel('ADC brut')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Chi2 sur les 5 candidats A_L
    ax = axes[1, 0]
    offsets = np.arange(-A_OFFSET, -A_OFFSET + N_A_CANDS)
    chi2_L = []
    a_cands = AL_int[i] + offsets
    for a_c in a_cands:
        res = get_template(z_l, a_c, ke_l)
        if res is None:
            chi2_L.append(np.nan)
            continue
        mu_t, sig_t, _ = res
        n = min(len(mu_t), PAD_LEN)
        sig_t = np.clip(sig_t[:n], 1e-3, None)
        chi2_L.append(np.sum((tr_L[:n] - mu_t[:n])**2 / sig_t**2))
    colors = ['red' if a_c == t_al else 'steelblue' for a_c in a_cands]
    ax.bar(a_cands, [c if not np.isnan(c) else 0 for c in chi2_L],
           color=colors, alpha=0.8)
    ax.set_xlabel('A candidat'); ax.set_ylabel('χ²')
    ax.set_title(f'χ² candidats A_L (rouge=vrai A={t_al})', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Chi2 sur les 5 candidats A_H
    ax = axes[1, 1]
    a_cands_h = AH_int[i] + offsets
    chi2_H = []
    for a_c in a_cands_h:
        res = get_template(z_h, a_c, ke_h)
        if res is None:
            chi2_H.append(np.nan)
            continue
        mu_t, sig_t, _ = res
        n = min(len(mu_t), PAD_LEN)
        sig_t = np.clip(sig_t[:n], 1e-3, None)
        chi2_H.append(np.sum((tr_H[:n] - mu_t[:n])**2 / sig_t**2))
    colors = ['red' if a_c == t_ah else 'steelblue' for a_c in a_cands_h]
    ax.bar(a_cands_h, [c if not np.isnan(c) else 0 for c in chi2_H],
           color=colors, alpha=0.8)
    ax.set_xlabel('A candidat'); ax.set_ylabel('χ²')
    ax.set_title(f'χ² candidats A_H (rouge=vrai A={t_ah})', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Event {i+1}/{N}  (idx={ev})  '
                 f'← / → pour naviguer  |  q pour quitter', fontsize=10)
    fig.canvas.draw()

def on_key(event):
    if event.key in ('right', 'n', ' '):
        current[0] = min(current[0] + 1, N - 1)
        draw(current[0])
    elif event.key in ('left', 'p'):
        current[0] = max(current[0] - 1, 0)
        draw(current[0])
    elif event.key == 'q':
        plt.close('all')
        sys.exit(0)

fig.canvas.mpl_connect('key_press_event', on_key)
draw(0)
print("Navigation : → ou espace = suivant  |  ← = précédent  |  q = quitter")
plt.show()
