"""
explore_regression.py
=====================
Diagnostic de régression sur Z_L à partir du modèle dann_ae_final.
Deux approches comparées :
  1. Postprocessing : Z_pred = E[Z] = sum(softmax * Z_L_unique)
  2. Référence      : argmax (classification standard)

Métriques :
  - sigma(Z_pred - Z_true)
  - MAE
  - biais
  - distribution des résidus
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ── Config ────────────────────────────────────────────────────────────────────
H5_TARGET  = '../simulation_output/T2F_G4.h5'
MODEL_PATH = 'dann_ae_final_model.pt'
OUT_PDF    = 'explore_regression.pdf'

PAD_LEN    = 50
VAL_FRAC   = 0.1
BATCH_SIZE = 2048

DEVICE = ('mps' if torch.backends.mps.is_available() else
          'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")

# ── Architecture (identique à train_dann_ae_final.py) ─────────────────────────
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

def make_mlp(dims, activation=nn.GELU, final_layer=None):
    layers = []
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i+1]), nn.LayerNorm(dims[i+1]), activation()]
    if final_layer is not None:
        layers.append(final_layer)
    return nn.Sequential(*layers)

class Encoder(nn.Module):
    def __init__(self, hidden, latent_dim):
        super().__init__()
        self.net    = make_mlp([PAD_LEN] + hidden)
        self.z_head = nn.Linear(hidden[-1], latent_dim)
    def forward(self, x):
        return self.z_head(self.net(x))

class Decoder(nn.Module):
    def __init__(self, hidden, latent_dim):
        super().__init__()
        dims = [latent_dim] + list(reversed(hidden))
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.LayerNorm(dims[i+1]), nn.GELU()]
        layers += [nn.Linear(hidden[0], PAD_LEN), nn.Softplus()]
        self.net = nn.Sequential(*layers)
    def forward(self, z):
        return self.net(z)

class DomainDiscriminator(nn.Module):
    def __init__(self, hidden, latent_dim):
        super().__init__()
        self.net = make_mlp([latent_dim * 2] + hidden,
                            final_layer=nn.Linear(hidden[-1], 2))
    def forward(self, z_L, z_H, alpha=1.0):
        x = grad_reverse(torch.cat([z_L, z_H], dim=1), alpha)
        return self.net(x)

class DANN_AE(nn.Module):
    def __init__(self, n_classes, latent_dim, hidden_ae, hidden_clf, hidden_disc):
        super().__init__()
        self.encoder_L     = Encoder(hidden_ae, latent_dim)
        self.decoder_L     = Decoder(hidden_ae, latent_dim)
        self.encoder_H     = Encoder(hidden_ae, latent_dim)
        self.decoder_H     = Decoder(hidden_ae, latent_dim)
        self.discriminator = DomainDiscriminator(hidden_disc, latent_dim)
        clf_dims = [latent_dim * 2] + hidden_clf
        self.clf = make_mlp(clf_dims, final_layer=nn.Linear(hidden_clf[-1], n_classes))
    def forward(self, xL, xH, alpha=0.0):
        z_L = self.encoder_L(xL); z_H = self.encoder_H(xH)
        logits = self.clf(torch.cat([z_L, z_H], dim=1))
        return logits

# ── Chargement modèle ─────────────────────────────────────────────────────────
print(f"Chargement {MODEL_PATH}...")
ckpt       = torch.load(MODEL_PATH, map_location=DEVICE)
Z_L_unique = ckpt['Z_L_unique']
n_classes  = len(Z_L_unique)
trace_max  = ckpt['trace_max']

# Architecture — hyperparamètres Optuna
latent_dim  = 8
hidden_ae   = [64, 64, 64]
hidden_clf  = [32, 32]
hidden_disc = [32, 32]
print(f"  Architecture : latent={latent_dim}  ae={hidden_ae}  "
      f"clf={hidden_clf}  disc={hidden_disc}")

model = DANN_AE(n_classes, latent_dim, hidden_ae, hidden_clf, hidden_disc).to(DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()

# Vecteur Z pour l'espérance
z_vals = torch.tensor(Z_L_unique, dtype=torch.float32, device=DEVICE)

# ── Chargement données ────────────────────────────────────────────────────────
print(f"Chargement {H5_TARGET}...")
with h5py.File(H5_TARGET, 'r') as f:
    total     = f['L/Z'].shape[0]
    n_labeled = int(total * 0.1)
    sl        = slice(0, n_labeled)
    raw_L  = f['L/raw_trace'][sl]
    raw_H  = f['H/raw_trace'][sl]
    Z_L    = f['L/Z'][sl].astype(int)

def pad_traces(raw):
    N = len(raw)
    traces = np.zeros((N, PAD_LEN), dtype=np.float32)
    masks  = np.zeros((N, PAD_LEN), dtype=np.float32)
    for i, t in enumerate(raw):
        k = min(len(t), PAD_LEN)
        traces[i, :k] = t[:k]; masks[i, :k] = 1.0
    return traces, masks

tL, _ = pad_traces(raw_L); tH, _ = pad_traces(raw_H)
tL /= trace_max; tH /= trace_max

rng     = np.random.default_rng(42)
idx_all = rng.permutation(n_labeled)
n_val   = int(n_labeled * VAL_FRAC)
idx_val = idx_all[-n_val:]
true_Z  = Z_L[idx_val]

def to_gpu(arr): return torch.from_numpy(arr).to(DEVICE)
tL_gpu = to_gpu(tL[idx_val])
tH_gpu = to_gpu(tH[idx_val])

# ── Inférence ─────────────────────────────────────────────────────────────────
print("Inférence...")
all_logits = []
n = len(idx_val)
with torch.no_grad():
    for start in range(0, n, BATCH_SIZE):
        sl = slice(start, start + BATCH_SIZE)
        logits = model(tL_gpu[sl], tH_gpu[sl])
        all_logits.append(logits.cpu())

logits_all = torch.cat(all_logits, dim=0)
proba      = torch.softmax(logits_all, dim=1).numpy()
pred_Z_clf = Z_L_unique[proba.argmax(axis=1)]          # classification argmax
pred_Z_reg = (torch.softmax(logits_all, dim=1) * z_vals.cpu()).sum(dim=1).numpy()  # espérance

# ── Métriques ─────────────────────────────────────────────────────────────────
res_clf = pred_Z_clf.astype(float) - true_Z.astype(float)
res_reg = pred_Z_reg              - true_Z.astype(float)

def metrics(res, name):
    print(f"\n{name}:")
    print(f"  sigma (std résidus) = {res.std():.4f}")
    print(f"  MAE                 = {np.abs(res).mean():.4f}")
    print(f"  biais               = {res.mean():.4f}")
    print(f"  RMSE                = {np.sqrt((res**2).mean()):.4f}")

metrics(res_clf, "Argmax (classification)")
metrics(res_reg, "Espérance softmax (régression)")

# Gain par Z
z_vals_unique = np.unique(true_Z)
sigma_clf_z, sigma_reg_z, z_valid = [], [], []
for z in z_vals_unique:
    mask = true_Z == z
    if mask.sum() < 5: continue
    sigma_clf_z.append(res_clf[mask].std())
    sigma_reg_z.append(res_reg[mask].std())
    z_valid.append(z)

# ── Figures ───────────────────────────────────────────────────────────────────
print(f"\nGénération {OUT_PDF}...")
with PdfPages(OUT_PDF) as pdf:

    # Fig 1 : distribution des résidus
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.arange(-5.5, 6.5, 1)
    bins_reg = np.linspace(-5, 5, 80)

    axes[0].hist(res_clf, bins=bins, color='steelblue', alpha=0.8, density=True)
    axes[0].set_xlabel('Z_pred - Z_true'); axes[0].set_ylabel('Densité')
    axes[0].set_title(f'Argmax — σ={res_clf.std():.3f}  MAE={np.abs(res_clf).mean():.3f}')
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(res_reg, bins=bins_reg, color='coral', alpha=0.8, density=True)
    axes[1].set_xlabel('Z_pred - Z_true'); axes[1].set_ylabel('Densité')
    axes[1].set_title(f'Espérance — σ={res_reg.std():.3f}  MAE={np.abs(res_reg).mean():.3f}')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Distribution des résidus Z_pred - Z_true\nArgmax vs Espérance softmax',
                 fontsize=11)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Fig 2 : sigma par Z
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(z_valid))
    ax.plot(x, sigma_clf_z, 'b-o', markersize=5, label=f'Argmax  (σ moy={np.mean(sigma_clf_z):.3f})')
    ax.plot(x, sigma_reg_z, 'r-o', markersize=5, label=f'Espérance (σ moy={np.mean(sigma_reg_z):.3f})')
    ax.set_xticks(x); ax.set_xticklabels(z_valid, fontsize=7, rotation=90)
    ax.set_xlabel('Z_L vrai'); ax.set_ylabel('σ résidus')
    ax.set_title('Sigma des résidus par Z — argmax vs espérance', fontsize=11)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Fig 3 : Z_pred vs Z_true (régression)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(true_Z, pred_Z_reg, alpha=0.1, s=2, color='coral')
    zmin, zmax = true_Z.min(), true_Z.max()
    ax.plot([zmin, zmax], [zmin, zmax], 'k--', alpha=0.5, label='y=x')
    ax.set_xlabel('Z_L vrai'); ax.set_ylabel('Z_L prédit (espérance)')
    ax.set_title(f'Régression par espérance softmax\nσ={res_reg.std():.3f}  MAE={np.abs(res_reg).mean():.3f}',
                 fontsize=11)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Fig 4 : profil de confiance — std résidus vs confiance
    conf     = proba.max(axis=1)
    n_bins   = 10
    bin_edges = np.percentile(conf, np.linspace(0, 100, n_bins + 1))
    bin_centers, sigma_clf_c, sigma_reg_c = [], [], []
    for i in range(n_bins):
        mask = (conf >= bin_edges[i]) & (conf < bin_edges[i+1])
        if mask.sum() < 5: continue
        bin_centers.append(conf[mask].mean())
        sigma_clf_c.append(res_clf[mask].std())
        sigma_reg_c.append(res_reg[mask].std())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bin_centers, sigma_clf_c, 'b-o', label='Argmax')
    ax.plot(bin_centers, sigma_reg_c, 'r-o', label='Espérance')
    ax.set_xlabel('Confiance (max softmax)')
    ax.set_ylabel('σ résidus')
    ax.set_title('Sigma résidus vs confiance\nLes events incertains bénéficient-ils de la régression ?',
                 fontsize=11)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Fig 5 : histogramme des Z prédits (espérance) binning=0.05 — toute la stat
    # Charger tous les events labelisés (pas seulement le val set)
    with h5py.File(H5_TARGET, 'r') as f:
        total     = f['L/Z'].shape[0]
        n_labeled = int(total * 0.1)
        raw_L_all = f['L/raw_trace'][:n_labeled]
        raw_H_all = f['H/raw_trace'][:n_labeled]
        Z_L_full  = f['L/Z'][:n_labeled].astype(int)

    tL_all, _ = pad_traces(raw_L_all); tH_all, _ = pad_traces(raw_H_all)
    tL_all /= trace_max; tH_all /= trace_max
    tL_all_gpu = to_gpu(tL_all); tH_all_gpu = to_gpu(tH_all)

    all_logits_full = []
    with torch.no_grad():
        for start in range(0, n_labeled, BATCH_SIZE):
            sl = slice(start, start + BATCH_SIZE)
            logits = model(tL_all_gpu[sl], tH_all_gpu[sl])
            all_logits_full.append(logits.cpu())
    logits_full  = torch.cat(all_logits_full, dim=0)
    pred_Z_full  = (torch.softmax(logits_full, dim=1) *
                    z_vals.cpu()).sum(dim=1).numpy()

    fig, ax = plt.subplots(figsize=(14, 5))
    bins_fine = np.arange(Z_L_unique.min() - 0.5, Z_L_unique.max() + 0.55, 0.05)
    ax.hist(pred_Z_full, bins=bins_fine, color='coral', alpha=0.8,
            label=f'N={len(pred_Z_full):,}')

    # Fit gaussien sur le pic Z=42 entre 41.7 et 42.3
    from scipy.optimize import curve_fit
    from scipy.stats import norm as sp_norm

    mask_42   = (pred_Z_full >= 41.7) & (pred_Z_full <= 42.3)
    data_42   = pred_Z_full[mask_42]
    counts_42, edges_42 = np.histogram(data_42, bins=np.arange(41.7, 42.31, 0.05))
    centers_42 = (edges_42[:-1] + edges_42[1:]) / 2

    def gaussian(x, amp, mu, sigma):
        return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)

    try:
        p0   = [counts_42.max(), 42.0, 0.15]
        popt, _ = curve_fit(gaussian, centers_42, counts_42, p0=p0)
        amp_fit, mu_fit, sigma_fit = popt
        x_fit = np.linspace(41.7, 42.3, 200)
        ax.plot(x_fit, gaussian(x_fit, *popt), 'b-', linewidth=2,
                label=f'Fit Z=42 : μ={mu_fit:.3f}  σ={abs(sigma_fit):.3f}')
        ax.axvspan(41.7, 42.3, alpha=0.08, color='blue', label='Zone fit')
    except Exception as e:
        print(f"Fit gaussien échoué : {e}")

    ax.set_xlabel('Z_L prédit (espérance softmax)')
    ax.set_ylabel('N events')
    ax.set_title(f'Distribution des Z_L prédits — binning=0.05  (N={len(pred_Z_full):,})',
                 fontsize=11)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Fig 6 : distribution Z prédits par seuil de confiance — 4 panneaux
    conf_full = torch.softmax(logits_full, dim=1).max(dim=1).values.numpy()
    thresholds_plot = [0.65, 0.75, 0.80, 0.85]
    bins_fine2 = np.arange(Z_L_unique.min() - 0.5, Z_L_unique.max() + 0.55, 0.05)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
    fig.suptitle('Distribution Z_L prédits — séparation par seuil de confiance\n'
                 'Bleu : conf ≥ seuil   |   Orange : conf < seuil', fontsize=12)

    for ax, thr in zip(axes.flat, thresholds_plot):
        mask_high = conf_full >= thr
        mask_low  = ~mask_high
        frac_high = mask_high.mean()
        frac_low  = mask_low.mean()

        ax.hist(pred_Z_full[mask_low],  bins=bins_fine2, color='coral',
                alpha=0.8, label=f'conf < {thr}  ({frac_low*100:.1f}%)')
        ax.hist(pred_Z_full[mask_high], bins=bins_fine2, color='steelblue',
                alpha=0.8, label=f'conf ≥ {thr}  ({frac_high*100:.1f}%)')

        ax.set_xlabel('Z_L prédit (espérance softmax)')
        ax.set_ylabel('N events')
        ax.set_title(f'Seuil = {thr}', fontsize=11)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Fig 7 : table de pureté — pour chaque seuil et chaque Z
    # Pureté = P(Z_vrai=Z | Z_pred ∈ [Z-0.4, Z+0.4] & conf≥seuil)
    HALF_WIN  = 0.4
    thresholds_pur = [0.55, 0.65, 0.75, 0.80, 0.85]
    z_list = sorted(np.unique(Z_L_full))

    # Tableau : lignes=Z, colonnes=seuil
    purity_table  = np.full((len(z_list), len(thresholds_pur)), np.nan)
    n_table       = np.full((len(z_list), len(thresholds_pur)), 0, dtype=int)

    for j, thr in enumerate(thresholds_pur):
        mask_conf = conf_full >= thr
        for i, z in enumerate(z_list):
            mask_peak = (pred_Z_full >= z - HALF_WIN) & (pred_Z_full <= z + HALF_WIN)
            mask      = mask_conf & mask_peak
            n_sel     = mask.sum()
            if n_sel < 5:
                continue
            purity = (Z_L_full[mask] == z).mean()
            purity_table[i, j] = purity
            n_table[i, j]      = n_sel

    # Affichage heatmap
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Heatmap pureté
    ax = axes[0]
    im = ax.imshow(purity_table, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(thresholds_pur)))
    ax.set_xticklabels([f'{t:.2f}' for t in thresholds_pur])
    ax.set_yticks(range(len(z_list)))
    ax.set_yticklabels(z_list, fontsize=8)
    ax.set_xlabel('Seuil confiance'); ax.set_ylabel('Z_L vrai')
    ax.set_title('Pureté du pic\nP(Z_vrai=Z | Z_pred∈[Z±0.4], conf≥seuil)', fontsize=10)
    plt.colorbar(im, ax=ax)
    # Annotations
    for i in range(len(z_list)):
        for j in range(len(thresholds_pur)):
            if not np.isnan(purity_table[i, j]):
                ax.text(j, i, f'{purity_table[i,j]:.2f}',
                        ha='center', va='center', fontsize=6,
                        color='black' if purity_table[i,j] < 0.85 else 'white')

    # Heatmap N events
    ax = axes[1]
    im2 = ax.imshow(n_table.astype(float), cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(thresholds_pur)))
    ax.set_xticklabels([f'{t:.2f}' for t in thresholds_pur])
    ax.set_yticks(range(len(z_list)))
    ax.set_yticklabels(z_list, fontsize=8)
    ax.set_xlabel('Seuil confiance'); ax.set_ylabel('Z_L vrai')
    ax.set_title('N events dans le pic\n(Z_pred∈[Z±0.4], conf≥seuil)', fontsize=10)
    plt.colorbar(im2, ax=ax)
    for i in range(len(z_list)):
        for j in range(len(thresholds_pur)):
            if n_table[i, j] > 0:
                ax.text(j, i, f'{n_table[i,j]}',
                        ha='center', va='center', fontsize=6, color='black')

    fig.suptitle('Pureté et statistique des pics Z_L — fenêtre ±0.4 autour de chaque Z',
                 fontsize=12)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Fig 8 : même table mais pour conf < seuil
    purity_table_low = np.full((len(z_list), len(thresholds_pur)), np.nan)
    n_table_low      = np.full((len(z_list), len(thresholds_pur)), 0, dtype=int)

    for j, thr in enumerate(thresholds_pur):
        mask_conf = conf_full < thr
        for i, z in enumerate(z_list):
            mask_peak = (pred_Z_full >= z - HALF_WIN) & (pred_Z_full <= z + HALF_WIN)
            mask      = mask_conf & mask_peak
            n_sel     = mask.sum()
            if n_sel < 5:
                continue
            purity_table_low[i, j] = (Z_L_full[mask] == z).mean()
            n_table_low[i, j]      = n_sel

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    ax = axes[0]
    im = ax.imshow(purity_table_low, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(thresholds_pur)))
    ax.set_xticklabels([f'{t:.2f}' for t in thresholds_pur])
    ax.set_yticks(range(len(z_list))); ax.set_yticklabels(z_list, fontsize=8)
    ax.set_xlabel('Seuil confiance'); ax.set_ylabel('Z_L vrai')
    ax.set_title('Pureté du pic — conf < seuil\nP(Z_vrai=Z | Z_pred∈[Z±0.4], conf<seuil)',
                 fontsize=10)
    plt.colorbar(im, ax=ax)
    for i in range(len(z_list)):
        for j in range(len(thresholds_pur)):
            if not np.isnan(purity_table_low[i, j]):
                ax.text(j, i, f'{purity_table_low[i,j]:.2f}',
                        ha='center', va='center', fontsize=6,
                        color='black' if purity_table_low[i,j] < 0.85 else 'white')

    ax = axes[1]
    im2 = ax.imshow(n_table_low.astype(float), cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(thresholds_pur)))
    ax.set_xticklabels([f'{t:.2f}' for t in thresholds_pur])
    ax.set_yticks(range(len(z_list))); ax.set_yticklabels(z_list, fontsize=8)
    ax.set_xlabel('Seuil confiance'); ax.set_ylabel('Z_L vrai')
    ax.set_title('N events dans le pic — conf < seuil', fontsize=10)
    plt.colorbar(im2, ax=ax)
    for i in range(len(z_list)):
        for j in range(len(thresholds_pur)):
            if n_table_low[i, j] > 0:
                ax.text(j, i, f'{n_table_low[i,j]}',
                        ha='center', va='center', fontsize=6, color='black')

    fig.suptitle('Pureté et statistique — events sous le seuil de confiance\n'
                 'fenêtre ±0.4 autour de chaque Z', fontsize=12)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Fig 9 : pureté sans coupure sur la confiance — toute la stat
    purity_table_all = np.full((len(z_list), 1), np.nan)
    n_table_all      = np.full((len(z_list), 1), 0, dtype=int)

    for i, z in enumerate(z_list):
        mask_peak = (pred_Z_full >= z - HALF_WIN) & (pred_Z_full <= z + HALF_WIN)
        n_sel     = mask_peak.sum()
        if n_sel < 5:
            continue
        purity_table_all[i, 0] = (Z_L_full[mask_peak] == z).mean()
        n_table_all[i, 0]      = n_sel

    fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    ax = axes[0]
    im = ax.imshow(purity_table_all, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
    ax.set_xticks([0]); ax.set_xticklabels(['Toute stat'])
    ax.set_yticks(range(len(z_list))); ax.set_yticklabels(z_list, fontsize=8)
    ax.set_ylabel('Z_L vrai')
    ax.set_title('Pureté — toute la stat\nP(Z_vrai=Z | Z_pred∈[Z±0.4])', fontsize=10)
    plt.colorbar(im, ax=ax)
    for i in range(len(z_list)):
        if not np.isnan(purity_table_all[i, 0]):
            ax.text(0, i, f'{purity_table_all[i,0]:.2f}',
                    ha='center', va='center', fontsize=8,
                    color='black' if purity_table_all[i,0] < 0.85 else 'white')

    ax = axes[1]
    im2 = ax.imshow(n_table_all.astype(float), cmap='Blues', aspect='auto')
    ax.set_xticks([0]); ax.set_xticklabels(['Toute stat'])
    ax.set_yticks(range(len(z_list))); ax.set_yticklabels(z_list, fontsize=8)
    ax.set_ylabel('Z_L vrai')
    ax.set_title('N events — toute la stat', fontsize=10)
    plt.colorbar(im2, ax=ax)
    for i in range(len(z_list)):
        if n_table_all[i, 0] > 0:
            ax.text(0, i, f'{n_table_all[i,0]}',
                    ha='center', va='center', fontsize=8, color='black')

    fig.suptitle('Pureté et statistique — toute la stat, sans coupure confiance\n'
                 'fenêtre ±0.4 autour de chaque Z', fontsize=12)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

print(f"PDF généré : {OUT_PDF}")
