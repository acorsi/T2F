"""
bayesian_mass_vec.py
====================
Version vectorisée de bayesian_mass.py.
  - Z_final = round(Z_pred)  — un seul Z par event
  - 5 candidats A : [round(A_pred)-2 .. round(A_pred)+2]
  - Chi2 calculé en une seule opération numpy (N_events × 5 × N_pads)
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import time

# ── Config ────────────────────────────────────────────────────────────────────
H5_TARGET      = '../../simulation_output/T2F_G4.h5'
MODEL_Z_PATH   = '../dann_ae_final_model.pt'
MODEL_A_PATH   = '../dann_ae_mass_kin_model_G4.pt'
TEMPLATES_PATH = './templates_full.npz'
OUT_PDF        = 'bayesian_mass_vec.pdf'

PAD_LEN    = 50
VAL_FRAC   = 0.1
BATCH_SIZE = 2048
N_A_CANDS  = 5       # [A_pred-2 .. A_pred+2]
A_OFFSET   = 2       # centré sur round(A_pred)
Z_SUM      = 98

DEVICE = ('mps' if torch.backends.mps.is_available() else
          'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")

# ── Architectures ─────────────────────────────────────────────────────────────
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)

def make_encoder(hidden, latent):
    """Structure exacte : self.net + self.z_head"""
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
        def forward(self, x):
            return self.z_head(self.net(x))
    return Encoder()

def make_decoder(hidden, latent):
    """Structure exacte : self.net (inclut la couche de sortie)"""
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
        def forward(self, z):
            return self.net(z)
    return Decoder()

def make_disc(hidden, latent):
    """Structure exacte : self.net"""
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
        def forward(self, x):
            return self.net(x)
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

# ── Chargement modèles ────────────────────────────────────────────────────────
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
print(f"  Z_L : {Z_L_unique.min()}..{Z_L_unique.max()}")
print(f"  A_L : {A_L_unique.min()}..{A_L_unique.max()}")
print(f"  A_H : {A_H_unique.min()}..{A_H_unique.max()}")

# ── Chargement templates ──────────────────────────────────────────────────────
print("Chargement templates...")
t0 = time.time()
tpl          = np.load(TEMPLATES_PATH)
tpl_Z        = tpl['Z'].astype(int)
tpl_A        = tpl['A'].astype(int)
tpl_ke       = tpl['ke_center'].astype(np.float32)
tpl_mu       = tpl['mu'].astype(np.float32)    # (N_tpl, max_pads)
tpl_sigma    = tpl['sigma'].astype(np.float32)
tpl_n_pads   = tpl['n_pads'].astype(int)
KE_BIN_WIDTH = float(tpl['meta_ke_bin_width'])
N_TPL        = len(tpl_Z)
MAX_PADS     = tpl_mu.shape[1]
print(f"  {N_TPL} templates  max_pads={MAX_PADS}  en {time.time()-t0:.1f}s")

# ── Lookup vectorisé : (Z, A, ke_bin) → index template ───────────────────────
# Pour chaque (Z, A), on indexe par ke_bin = round(ke_center / KE_BIN_WIDTH)
print("Construction lookup vectorisé...")
# Dictionnaire (Z, A, ke_bin) → row index dans tpl_mu/sigma
lookup = {}
ke_bin_arr = np.round(tpl_ke / KE_BIN_WIDTH).astype(int)
for i in range(N_TPL):
    key = (tpl_Z[i], tpl_A[i], ke_bin_arr[i])
    lookup[key] = i
print(f"  {len(lookup)} entrées dans le lookup")

def get_templates_batch(Z_arr, A_matrix, KE_arr):
    """
    Extrait les templates pour un batch d'events.
    Z_arr    : (N,)        Z entier par event
    A_matrix : (N, N_CANDS) A candidats entiers
    KE_arr   : (N,)        KE mesuré (MeV)
    Retourne :
      mu_batch    : (N, N_CANDS, PAD_LEN)
      sigma_batch : (N, N_CANDS, PAD_LEN)
      valid_mask  : (N, N_CANDS) bool — template trouvé
    """
    N = len(Z_arr)
    mu_batch    = np.zeros((N, N_A_CANDS, PAD_LEN), dtype=np.float32)
    sigma_batch = np.ones ((N, N_A_CANDS, PAD_LEN), dtype=np.float32) * 1e6
    valid_mask  = np.zeros((N, N_A_CANDS), dtype=bool)

    ke_bins = np.round(KE_arr / KE_BIN_WIDTH).astype(int)

    for i in range(N):
        z   = int(Z_arr[i])
        keb = int(ke_bins[i])
        for j in range(N_A_CANDS):
            a   = int(A_matrix[i, j])
            key = (z, a, keb)
            # Si bin exact absent, chercher le plus proche
            if key not in lookup:
                # Chercher bins voisins ±3
                best_idx = None
                best_dist = 999
                for dk in range(1, 4):
                    for sign in (-1, 1):
                        k2 = (z, a, keb + sign*dk)
                        if k2 in lookup:
                            if dk < best_dist:
                                best_dist = dk
                                best_idx = lookup[k2]
                            break
                if best_idx is None:
                    continue
                idx = best_idx
            else:
                idx = lookup[key]

            n_p = min(tpl_n_pads[idx], PAD_LEN)
            mu_batch[i, j, :n_p]    = tpl_mu[idx, :n_p]
            sigma_batch[i, j, :n_p] = np.clip(tpl_sigma[idx, :n_p], 1e-3, None)
            valid_mask[i, j]        = True

    return mu_batch, sigma_batch, valid_mask

def chi2_vectorized(traces, mu_batch, sigma_batch, valid_mask):
    """
    traces      : (N, PAD_LEN)
    mu_batch    : (N, N_CANDS, PAD_LEN)
    sigma_batch : (N, N_CANDS, PAD_LEN)
    valid_mask  : (N, N_CANDS)
    Retourne chi2 : (N, N_CANDS)
    """
    tr = traces[:, np.newaxis, :]                      # (N, 1, PAD_LEN)
    chi2 = ((tr - mu_batch)**2 / sigma_batch**2).sum(axis=2)  # (N, N_CANDS)
    chi2[~valid_mask] = np.inf
    return chi2

# ── Chargement données ────────────────────────────────────────────────────────
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

denom  = np.clip(KE_L_raw + KE_H_raw, 1e-6, None)
akin_L = (252.0 * KE_H_raw / denom).astype(np.float32)[:, None]
akin_H = (252.0 * KE_L_raw / denom).astype(np.float32)[:, None]

tL_Z = tL_raw / trace_max_Z;  tH_Z = tH_raw / trace_max_Z
tL_A = tL_raw / trace_max_A;  tH_A = tH_raw / trace_max_A

rng     = np.random.default_rng(42)
idx_all = rng.permutation(n_labeled)
n_val   = int(n_labeled * VAL_FRAC)
idx_val = idx_all[-n_val:]
N_VAL   = len(idx_val)

def to_gpu(arr): return torch.from_numpy(arr).to(DEVICE)

# ── Inférence réseaux ─────────────────────────────────────────────────────────
print("Inférence réseau Z...")
tLZ_gpu = to_gpu(tL_Z[idx_val]); tHZ_gpu = to_gpu(tH_Z[idx_val])
all_lZ = []
with torch.no_grad():
    for s in range(0, N_VAL, BATCH_SIZE):
        sl = slice(s, s + BATCH_SIZE)
        all_lZ.append(model_Z(tLZ_gpu[sl], tHZ_gpu[sl]).cpu())
logits_Z = torch.cat(all_lZ)
proba_Z  = torch.softmax(logits_Z, dim=1)
Z_pred   = (proba_Z * z_vals).sum(dim=1).numpy()

print("Inférence réseau A...")
tLA_gpu  = to_gpu(tL_A[idx_val]); tHA_gpu  = to_gpu(tH_A[idx_val])
akL_gpu  = to_gpu(akin_L[idx_val]); akH_gpu = to_gpu(akin_H[idx_val])
all_lAL, all_lAH = [], []
with torch.no_grad():
    for s in range(0, N_VAL, BATCH_SIZE):
        sl = slice(s, s + BATCH_SIZE)
        lAL, lAH = model_A(tLA_gpu[sl], tHA_gpu[sl], akL_gpu[sl], akH_gpu[sl])
        all_lAL.append(lAL.cpu()); all_lAH.append(lAH.cpu())
logits_AL = torch.cat(all_lAL); logits_AH = torch.cat(all_lAH)
proba_AL  = torch.softmax(logits_AL, dim=1)
proba_AH  = torch.softmax(logits_AH, dim=1)
AL_pred   = (proba_AL * a_vals_L).sum(dim=1).numpy()
AH_pred   = (proba_AH * a_vals_H).sum(dim=1).numpy()

# Z entier
Z_int_L = np.round(Z_pred).astype(int)
Z_int_H = Z_SUM - Z_int_L

# Matrices de candidats A : (N_VAL, N_A_CANDS)
AL_center = np.round(AL_pred).astype(int)
AH_center = np.round(AH_pred).astype(int)
offsets   = np.arange(-A_OFFSET, -A_OFFSET + N_A_CANDS)   # [-2,-1,0,1,2]
A_cands_L = AL_center[:, None] + offsets[None, :]   # (N_VAL, 5)
A_cands_H = AH_center[:, None] + offsets[None, :]

KE_L_val = KE_L_raw[idx_val]
KE_H_val = KE_H_raw[idx_val]
trace_L_val = tL_raw[idx_val]
trace_H_val = tH_raw[idx_val]

# ── Template matching vectorisé ───────────────────────────────────────────────
print("Template matching vectorisé...")
t0 = time.time()

# Côté léger
print("  Fragment L...")
mu_L, sig_L, valid_L = get_templates_batch(Z_int_L, A_cands_L, KE_L_val)
chi2_L = chi2_vectorized(trace_L_val, mu_L, sig_L, valid_L)  # (N, 5)
best_idx_L = np.argmin(chi2_L, axis=1)                        # (N,)
A_final_L  = A_cands_L[np.arange(N_VAL), best_idx_L]
chi2_best_L = chi2_L[np.arange(N_VAL), best_idx_L]

# Côté lourd
print("  Fragment H...")
mu_H, sig_H, valid_H = get_templates_batch(Z_int_H, A_cands_H, KE_H_val)
chi2_H = chi2_vectorized(trace_H_val, mu_H, sig_H, valid_H)
best_idx_H = np.argmin(chi2_H, axis=1)
A_final_H  = A_cands_H[np.arange(N_VAL), best_idx_H]
chi2_best_H = chi2_H[np.arange(N_VAL), best_idx_H]

print(f"  Terminé en {time.time()-t0:.1f}s")

# ── Métriques ─────────────────────────────────────────────────────────────────
true_ZL = true_Z_L[idx_val]; true_ZH = true_Z_H[idx_val]
true_AL = true_A_L[idx_val]; true_AH = true_A_H[idx_val]

AL_net = np.round(AL_pred).astype(int)
AH_net = np.round(AH_pred).astype(int)

def metrics(pred, true, name):
    acc   = (pred == true).mean()
    pm1   = (np.abs(pred - true) <= 1).mean()
    pm2   = (np.abs(pred - true) <= 2).mean()
    sigma = (pred.astype(float) - true.astype(float)).std()
    print(f"  {name:20s} acc={acc:.3f}  ±1={pm1:.3f}  ±2={pm2:.3f}  σ={sigma:.3f}")
    return acc, pm1, pm2, sigma

print(f"\n{'─'*60}")
print("Réseau seul :")
metrics(Z_int_L,   true_ZL, 'Z_L')
metrics(AL_net,    true_AL, 'A_L')
metrics(AH_net,    true_AH, 'A_H')
print("Template matching :")
metrics(Z_int_L,   true_ZL, 'Z_L (inchangé)')
acc_AL_tpl,pm1_AL,pm2_AL,sig_AL = metrics(A_final_L, true_AL, 'A_L')
acc_AH_tpl,pm1_AH,pm2_AH,sig_AH = metrics(A_final_H, true_AH, 'A_H')

# ── Figures ───────────────────────────────────────────────────────────────────
print(f"\nGénération {OUT_PDF}...")
with PdfPages(OUT_PDF) as pdf:

    # Fig 1 : résidus réseau vs template
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Résidus — réseau vs template matching', fontsize=11)
    for ax, res_net, res_tpl, title in [
        (axes[0], AL_net - true_AL, A_final_L - true_AL, 'A_L'),
        (axes[1], AH_net - true_AH, A_final_H - true_AH, 'A_H'),
    ]:
        bins = np.arange(-10.5, 11.5, 1)
        ax.hist(res_net, bins=bins, alpha=0.6, color='steelblue', density=True,
                label=f'Réseau  σ={res_net.std():.3f}')
        ax.hist(res_tpl, bins=bins, alpha=0.6, color='coral',     density=True,
                label=f'Template  σ={res_tpl.astype(float).std():.3f}')
        ax.set_xlabel('A_pred - A_true'); ax.set_ylabel('Densité')
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Fig 2 : matrices de confusion template matching
    for true_A, pred_A, title in [
        (true_AL, A_final_L, f'A_L template — acc={acc_AL_tpl:.3f}  σ={sig_AL:.3f}'),
        (true_AH, A_final_H, f'A_H template — acc={acc_AH_tpl:.3f}  σ={sig_AH:.3f}'),
    ]:
        a_min, a_max = true_A.min(), true_A.max()
        a_range  = np.arange(a_min, a_max + 1)
        a_to_row = {a: i for i, a in enumerate(a_range)}
        cm = np.zeros((len(a_range), len(a_range)), dtype=int)
        for t, p in zip(true_A, pred_A):
            if t in a_to_row and p in a_to_row:
                cm[a_to_row[t], a_to_row[p]] += 1
        cm_n = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(cm_n, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        step = max(1, len(a_range) // 20)
        ax.set_xticks(range(0, len(a_range), step))
        ax.set_xticklabels(a_range[::step], fontsize=6, rotation=90)
        ax.set_yticks(range(0, len(a_range), step))
        ax.set_yticklabels(a_range[::step], fontsize=6)
        ax.set_xlabel('A prédit'); ax.set_ylabel('A vrai')
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Fig 3 : distribution chi2
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, chi2, title in [
        (axes[0], chi2_best_L, 'χ² best — fragment L'),
        (axes[1], chi2_best_H, 'χ² best — fragment H'),
    ]:
        finite = chi2[np.isfinite(chi2)]
        ax.hist(np.clip(finite, 0, np.percentile(finite, 99)),
                bins=60, color='steelblue', alpha=0.8)
        ax.set_xlabel('χ²'); ax.set_ylabel('N events')
        ax.set_title(title, fontsize=10); ax.grid(True, alpha=0.3)
    fig.suptitle('Distribution des χ² minimaux', fontsize=11)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Fig 4 : sigma résidus par A — réseau vs template
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, true_A, res_net, res_tpl, title in [
        (axes[0], true_AL, AL_net - true_AL,
         A_final_L.astype(float) - true_AL, 'A_L'),
        (axes[1], true_AH, AH_net - true_AH,
         A_final_H.astype(float) - true_AH, 'A_H'),
    ]:
        a_uniq = np.unique(true_A)
        s_net, s_tpl, a_valid = [], [], []
        for a in a_uniq:
            mask = true_A == a
            if mask.sum() < 5: continue
            s_net.append(res_net[mask].std())
            s_tpl.append(res_tpl[mask].std())
            a_valid.append(a)
        x = np.arange(len(a_valid))
        step = max(1, len(a_valid) // 20)
        ax.plot(x, s_net, 'b-o', markersize=3,
                label=f'Réseau   σ moy={np.mean(s_net):.3f}')
        ax.plot(x, s_tpl, 'r-o', markersize=3,
                label=f'Template σ moy={np.mean(s_tpl):.3f}')
        ax.set_xticks(x[::step])
        ax.set_xticklabels(np.array(a_valid)[::step], fontsize=7, rotation=90)
        ax.set_xlabel(f'{title} vrai'); ax.set_ylabel('σ résidus')
        ax.set_title(f'Sigma résidus par {title}', fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle('Réseau vs template matching — sigma par A', fontsize=11)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Fig 5 : fraction de templates valides par event
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, valid, title in [
        (axes[0], valid_L.sum(axis=1), 'Fragment L'),
        (axes[1], valid_H.sum(axis=1), 'Fragment H'),
    ]:
        vals, counts = np.unique(valid, return_counts=True)
        ax.bar(vals, counts / counts.sum(), color='steelblue', alpha=0.8)
        ax.set_xlabel('N templates valides / event (sur 5)')
        ax.set_ylabel('Fraction')
        ax.set_title(title, fontsize=10); ax.grid(True, alpha=0.3, axis='y')
    fig.suptitle('Couverture des templates par event', fontsize=11)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

print(f"\nPDF généré : {OUT_PDF}")
print(f"A_L template : acc={acc_AL_tpl:.3f}  ±1={pm1_AL:.3f}  ±2={pm2_AL:.3f}  σ={sig_AL:.3f}")
print(f"A_H template : acc={acc_AH_tpl:.3f}  ±1={pm1_AH:.3f}  ±2={pm2_AH:.3f}  σ={sig_AH:.3f}")
