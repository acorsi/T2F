"""
explore_regression_mass.py
==========================
Diagnostic de régression sur A_L et A_H à partir du modèle dann_ae_mass_kin.
Figures :
  - Distribution des résidus argmax vs espérance
  - Sigma des résidus par A vrai
  - Scatter A_pred vs A_true
  - Sigma résidus vs confiance
  - Distribution des A prédits (binning=0.05) avec fit gaussien A_L=100
  - Séparation par seuil de confiance (4 panneaux)
  - Tables de pureté fenêtre ±0.4
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
MODEL_PATH = 'dann_ae_mass_kin_model_G4.pt'
OUT_PDF    = 'explore_regression_mass_G4.pdf'

PAD_LEN    = 50
VAL_FRAC   = 0.1
BATCH_SIZE = 2048

LATENT_DIM  = 8
HIDDEN_AE   = [256, 128, 64]
HIDDEN_CLF  = [64, 64]
HIDDEN_DISC = [32, 32]
CLF_IN_DIM  = LATENT_DIM * 2 + 1

DEVICE = ('mps' if torch.backends.mps.is_available() else
          'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")

# ── Architecture ──────────────────────────────────────────────────────────────
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

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [PAD_LEN] + HIDDEN_AE
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.LayerNorm(dims[i+1]), nn.GELU()]
        self.net    = nn.Sequential(*layers)
        self.z_head = nn.Linear(HIDDEN_AE[-1], LATENT_DIM)
    def forward(self, x):
        return self.z_head(self.net(x))

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [LATENT_DIM] + list(reversed(HIDDEN_AE))
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.LayerNorm(dims[i+1]), nn.GELU()]
        layers += [nn.Linear(HIDDEN_AE[0], PAD_LEN), nn.Softplus()]
        self.net = nn.Sequential(*layers)
    def forward(self, z):
        return self.net(z)

class DomainDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [LATENT_DIM * 2] + HIDDEN_DISC
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.LayerNorm(dims[i+1]), nn.GELU()]
        layers += [nn.Linear(HIDDEN_DISC[-1], 2)]
        self.net = nn.Sequential(*layers)
    def forward(self, z_L, z_H, alpha=1.0):
        x = grad_reverse(torch.cat([z_L, z_H], dim=1), alpha)
        return self.net(x)

def make_clf(n_classes):
    dims = [CLF_IN_DIM] + HIDDEN_CLF
    layers = []
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i+1]), nn.LayerNorm(dims[i+1]), nn.GELU()]
    layers += [nn.Linear(HIDDEN_CLF[-1], n_classes)]
    return nn.Sequential(*layers)

class DANN_AE_Mass(nn.Module):
    def __init__(self, n_classes_AL, n_classes_AH):
        super().__init__()
        self.encoder_L     = Encoder();  self.decoder_L = Decoder()
        self.encoder_H     = Encoder();  self.decoder_H = Decoder()
        self.discriminator = DomainDiscriminator()
        self.clf_AL        = make_clf(n_classes_AL)
        self.clf_AH        = make_clf(n_classes_AH)

    def forward(self, xL, xH, a_kin_L, a_kin_H, alpha=0.0):
        z_L = self.encoder_L(xL);  z_H = self.encoder_H(xH)
        clf_in_AL = torch.cat([z_L, z_H, a_kin_L / 252.0], dim=1)
        clf_in_AH = torch.cat([z_L, z_H, a_kin_H / 252.0], dim=1)
        logits_AL = self.clf_AL(clf_in_AL)
        logits_AH = self.clf_AH(clf_in_AH)
        return logits_AL, logits_AH

# ── Chargement modèle ─────────────────────────────────────────────────────────
print(f"Chargement {MODEL_PATH}...")
ckpt       = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
A_L_unique = ckpt['A_L_unique']
A_H_unique = ckpt['A_H_unique']
n_classes_AL = len(A_L_unique)
n_classes_AH = len(A_H_unique)
trace_max  = ckpt['trace_max']
print(f"  A_L : {A_L_unique.min()}..{A_L_unique.max()} ({n_classes_AL} classes)")
print(f"  A_H : {A_H_unique.min()}..{A_H_unique.max()} ({n_classes_AH} classes)")

model = DANN_AE_Mass(n_classes_AL, n_classes_AH).to(DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()

# ── Chargement données ────────────────────────────────────────────────────────
print(f"Chargement {H5_TARGET}...")
with h5py.File(H5_TARGET, 'r') as f:
    total     = f['L/Z'].shape[0]
    n_labeled = int(total * 0.1)
    sl        = slice(0, n_labeled)
    raw_L  = f['L/raw_trace'][sl]
    raw_H  = f['H/raw_trace'][sl]
    A_L    = f['L/A'][sl].astype(int)
    A_H    = f['H/A'][sl].astype(int)

def pad_traces(raw):
    N = len(raw)
    traces = np.zeros((N, PAD_LEN), dtype=np.float32)
    for i, t in enumerate(raw):
        k = min(len(t), PAD_LEN)
        traces[i, :k] = t[:k]
    return traces

def compute_akin(tL, tH):
    KE_L  = tL.sum(axis=1)
    KE_H  = tH.sum(axis=1)
    denom = np.clip(KE_L + KE_H, 1e-6, None)
    return (252.0 * KE_H / denom).astype(np.float32)[:, None], \
           (252.0 * KE_L / denom).astype(np.float32)[:, None]

tL = pad_traces(raw_L); tH = pad_traces(raw_H)
akin_L_all, akin_H_all = compute_akin(tL, tH)   # avant normalisation
tL /= trace_max; tH /= trace_max

rng     = np.random.default_rng(42)
idx_all = rng.permutation(n_labeled)
n_val   = int(n_labeled * VAL_FRAC)
idx_val = idx_all[-n_val:]
true_AL = A_L[idx_val]
true_AH = A_H[idx_val]

def to_gpu(arr): return torch.from_numpy(arr).to(DEVICE)

tL_gpu     = to_gpu(tL[idx_val])
tH_gpu     = to_gpu(tH[idx_val])
akin_L_gpu = to_gpu(akin_L_all[idx_val])
akin_H_gpu = to_gpu(akin_H_all[idx_val])

# ── Inférence val set ─────────────────────────────────────────────────────────
print("Inférence val set...")
all_logits_AL, all_logits_AH = [], []
n = len(idx_val)
with torch.no_grad():
    for start in range(0, n, BATCH_SIZE):
        sl = slice(start, start + BATCH_SIZE)
        lAL, lAH = model(tL_gpu[sl], tH_gpu[sl],
                         akin_L_gpu[sl], akin_H_gpu[sl])
        all_logits_AL.append(lAL.cpu().numpy())
        all_logits_AH.append(lAH.cpu().numpy())

logits_AL = np.concatenate(all_logits_AL)
logits_AH = np.concatenate(all_logits_AH)
proba_AL  = torch.softmax(torch.from_numpy(logits_AL), dim=1).numpy()
proba_AH  = torch.softmax(torch.from_numpy(logits_AH), dim=1).numpy()

pred_AL_clf = A_L_unique[proba_AL.argmax(axis=1)]
pred_AH_clf = A_H_unique[proba_AH.argmax(axis=1)]
pred_AL_reg = (torch.softmax(torch.from_numpy(logits_AL), dim=1) *
               torch.tensor(A_L_unique, dtype=torch.float32)).sum(dim=1).numpy()
pred_AH_reg = (torch.softmax(torch.from_numpy(logits_AH), dim=1) *
               torch.tensor(A_H_unique, dtype=torch.float32)).sum(dim=1).numpy()

conf_AL  = proba_AL.max(axis=1)
conf_AH  = proba_AH.max(axis=1)
conf     = np.minimum(conf_AL, conf_AH)

res_clf_AL = pred_AL_clf.astype(float) - true_AL.astype(float)
res_clf_AH = pred_AH_clf.astype(float) - true_AH.astype(float)
res_reg_AL = pred_AL_reg - true_AL.astype(float)
res_reg_AH = pred_AH_reg - true_AH.astype(float)

acc_AL     = (pred_AL_clf == true_AL).mean()
acc_AH     = (pred_AH_clf == true_AH).mean()
acc_pm1_AL = (np.abs(pred_AL_clf - true_AL) <= 1).mean()
acc_pm1_AH = (np.abs(pred_AH_clf - true_AH) <= 1).mean()
acc_pm2_AL = (np.abs(pred_AL_clf - true_AL) <= 2).mean()
acc_pm2_AH = (np.abs(pred_AH_clf - true_AH) <= 2).mean()

print(f"A_L : acc={acc_AL:.3f}  ±1={acc_pm1_AL:.3f}  ±2={acc_pm2_AL:.3f}"
      f"  σ(reg)={res_reg_AL.std():.3f}")
print(f"A_H : acc={acc_AH:.3f}  ±1={acc_pm1_AH:.3f}  ±2={acc_pm2_AH:.3f}"
      f"  σ(reg)={res_reg_AH.std():.3f}")

# Inférence toute la stat labelisée
print("Inférence toute la stat...")
tL_full_gpu     = to_gpu(tL[:n_labeled])
tH_full_gpu     = to_gpu(tH[:n_labeled])
akin_L_full_gpu = to_gpu(akin_L_all[:n_labeled])
akin_H_full_gpu = to_gpu(akin_H_all[:n_labeled])
A_L_full = A_L[:n_labeled]
A_H_full = A_H[:n_labeled]

all_lAL_full, all_lAH_full = [], []
with torch.no_grad():
    for start in range(0, n_labeled, BATCH_SIZE):
        sl = slice(start, start + BATCH_SIZE)
        lAL, lAH = model(tL_full_gpu[sl], tH_full_gpu[sl],
                         akin_L_full_gpu[sl], akin_H_full_gpu[sl])
        all_lAL_full.append(lAL.cpu())
        all_lAH_full.append(lAH.cpu())

logits_AL_full = torch.cat(all_lAL_full, dim=0)
logits_AH_full = torch.cat(all_lAH_full, dim=0)
proba_AL_full  = torch.softmax(logits_AL_full, dim=1).numpy()
proba_AH_full  = torch.softmax(logits_AH_full, dim=1).numpy()
pred_AL_full   = (torch.softmax(logits_AL_full, dim=1) *
                  torch.tensor(A_L_unique, dtype=torch.float32)).sum(dim=1).numpy()
pred_AH_full   = (torch.softmax(logits_AH_full, dim=1) *
                  torch.tensor(A_H_unique, dtype=torch.float32)).sum(dim=1).numpy()
conf_full      = np.minimum(proba_AL_full.max(axis=1),
                             proba_AH_full.max(axis=1))

# ── Figures ───────────────────────────────────────────────────────────────────
print(f"Génération {OUT_PDF}...")
with PdfPages(OUT_PDF) as pdf:

    # Fig 1 : distribution des résidus
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, res, sigma, title in [
        (axes[0,0], res_clf_AL, res_clf_AL.std(), f'A_L argmax  σ={res_clf_AL.std():.3f}'),
        (axes[0,1], res_reg_AL, res_reg_AL.std(), f'A_L espérance  σ={res_reg_AL.std():.3f}'),
        (axes[1,0], res_clf_AH, res_clf_AH.std(), f'A_H argmax  σ={res_clf_AH.std():.3f}'),
        (axes[1,1], res_reg_AH, res_reg_AH.std(), f'A_H espérance  σ={res_reg_AH.std():.3f}'),
    ]:
        bins = np.linspace(-10, 10, 80)
        ax.hist(res, bins=bins, color='steelblue', alpha=0.8, density=True)
        ax.set_xlabel('A_pred - A_true'); ax.set_ylabel('Densité')
        ax.set_title(title, fontsize=10); ax.grid(True, alpha=0.3)
    fig.suptitle('Distribution des résidus — argmax vs espérance softmax', fontsize=11)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Fig 2 : sigma par A vrai
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, true_A, res_clf, res_reg, title in [
        (axes[0], true_AL, res_clf_AL, res_reg_AL, 'A_L'),
        (axes[1], true_AH, res_clf_AH, res_reg_AH, 'A_H'),
    ]:
        a_vals = np.unique(true_A)
        s_clf, s_reg, a_valid = [], [], []
        for a in a_vals:
            mask = true_A == a
            if mask.sum() < 5: continue
            s_clf.append(res_clf[mask].std())
            s_reg.append(res_reg[mask].std())
            a_valid.append(a)
        x = np.arange(len(a_valid))
        ax.plot(x, s_clf, 'b-o', markersize=4, label=f'Argmax  (σ moy={np.mean(s_clf):.3f})')
        ax.plot(x, s_reg, 'r-o', markersize=4, label=f'Espérance (σ moy={np.mean(s_reg):.3f})')
        step = max(1, len(a_valid) // 20)
        ax.set_xticks(x[::step]); ax.set_xticklabels(a_valid[::step], fontsize=7, rotation=90)
        ax.set_xlabel(f'{title} vrai'); ax.set_ylabel('σ résidus')
        ax.set_title(f'Sigma résidus par {title}', fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle('Sigma des résidus par valeur de A', fontsize=11)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Fig 3 : scatter A_pred vs A_true
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, true_A, pred_A, res, title in [
        (axes[0], true_AL, pred_AL_reg, res_reg_AL,
         f'A_L  σ={res_reg_AL.std():.3f}  MAE={np.abs(res_reg_AL).mean():.3f}'),
        (axes[1], true_AH, pred_AH_reg, res_reg_AH,
         f'A_H  σ={res_reg_AH.std():.3f}  MAE={np.abs(res_reg_AH).mean():.3f}'),
    ]:
        ax.scatter(true_A, pred_A, alpha=0.05, s=2, color='steelblue')
        amin, amax = true_A.min(), true_A.max()
        ax.plot([amin, amax], [amin, amax], 'k--', alpha=0.5)
        ax.set_xlabel('A vrai'); ax.set_ylabel('A prédit (espérance)')
        ax.set_title(title, fontsize=10); ax.grid(True, alpha=0.3)
    fig.suptitle('Régression par espérance softmax', fontsize=11)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Fig 4 : sigma résidus vs confiance
    n_bins    = 10
    bin_edges = np.percentile(conf, np.linspace(0, 100, n_bins + 1))
    centers, s_clf_AL_c, s_reg_AL_c, s_clf_AH_c, s_reg_AH_c = [], [], [], [], []
    for i in range(n_bins):
        mask = (conf >= bin_edges[i]) & (conf < bin_edges[i+1])
        if mask.sum() < 5: continue
        centers.append(conf[mask].mean())
        s_clf_AL_c.append(res_clf_AL[mask].std())
        s_reg_AL_c.append(res_reg_AL[mask].std())
        s_clf_AH_c.append(res_clf_AH[mask].std())
        s_reg_AH_c.append(res_reg_AH[mask].std())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, s_clf, s_reg, title in [
        (axes[0], s_clf_AL_c, s_reg_AL_c, 'A_L'),
        (axes[1], s_clf_AH_c, s_reg_AH_c, 'A_H'),
    ]:
        ax.plot(centers, s_clf, 'b-o', label='Argmax')
        ax.plot(centers, s_reg, 'r-o', label='Espérance')
        ax.set_xlabel('Confiance'); ax.set_ylabel('σ résidus')
        ax.set_title(f'Sigma résidus vs confiance — {title}', fontsize=10)
        ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle('Les events incertains bénéficient-ils de la régression ?', fontsize=11)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Fig 5 : distribution A_L et A_H prédits binning=0.05 avec fit gaussien
    from scipy.optimize import curve_fit

    def gaussian(x, amp, mu, sigma):
        return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, pred_full, A_unique, true_A_full, title, fit_center in [
        (axes[0], pred_AL_full, A_L_unique, A_L_full, 'A_L', 100),
        (axes[1], pred_AH_full, A_H_unique, A_H_full, 'A_H', 140),
    ]:
        bins_fine = np.arange(A_unique.min() - 0.5, A_unique.max() + 0.55, 0.05)
        ax.hist(pred_full, bins=bins_fine, color='coral', alpha=0.8,
                label=f'N={len(pred_full):,}')

        # Fit gaussien autour du pic central
        half = 0.6
        mask_fit = (pred_full >= fit_center - half) & (pred_full <= fit_center + half)
        data_fit = pred_full[mask_fit]
        if len(data_fit) > 10:
            counts, edges = np.histogram(data_fit,
                                         bins=np.arange(fit_center - half,
                                                        fit_center + half + 0.05, 0.05))
            centers_fit = (edges[:-1] + edges[1:]) / 2
            try:
                popt, _ = curve_fit(gaussian, centers_fit, counts,
                                    p0=[counts.max(), float(fit_center), 0.3])
                x_fit = np.linspace(fit_center - half, fit_center + half, 200)
                ax.plot(x_fit, gaussian(x_fit, *popt), 'b-', linewidth=2,
                        label=f'Fit A={fit_center} : μ={popt[1]:.2f}  σ={abs(popt[2]):.3f}')
                ax.axvspan(fit_center - half, fit_center + half,
                           alpha=0.08, color='blue')
            except Exception as e:
                print(f"Fit {title} échoué : {e}")

        ax.set_xlabel(f'{title} prédit (espérance softmax)')
        ax.set_ylabel('N events')
        ax.set_title(f'Distribution {title} prédits — binning=0.05', fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(f'Distribution des masses prédites (N={n_labeled:,})', fontsize=11)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Fig 6 : séparation par seuil de confiance
    thresholds_plot = [0.15, 0.25, 0.35, 0.50]
    bins_fine_AL = np.arange(A_L_unique.min() - 0.5, A_L_unique.max() + 0.55, 0.05)
    bins_fine_AH = np.arange(A_H_unique.min() - 0.5, A_H_unique.max() + 0.55, 0.05)

    for pred_full, bins_fine, title in [
        (pred_AL_full, bins_fine_AL, 'A_L'),
        (pred_AH_full, bins_fine_AH, 'A_H'),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
        fig.suptitle(f'Distribution {title} prédits — séparation par seuil de confiance\n'
                     f'Bleu : conf ≥ seuil   |   Orange : conf < seuil', fontsize=11)
        for ax, thr in zip(axes.flat, thresholds_plot):
            mask_high = conf_full >= thr
            mask_low  = ~mask_high
            ax.hist(pred_full[mask_low],  bins=bins_fine, color='coral',
                    alpha=0.8, label=f'conf < {thr}  ({mask_low.mean()*100:.1f}%)')
            ax.hist(pred_full[mask_high], bins=bins_fine, color='steelblue',
                    alpha=0.8, label=f'conf ≥ {thr}  ({mask_high.mean()*100:.1f}%)')
            ax.set_xlabel(f'{title} prédit'); ax.set_ylabel('N events')
            ax.set_title(f'Seuil = {thr}', fontsize=10)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Fig 7 : tables de pureté fenêtre ±0.4 (conf ≥ seuil)
    HALF_WIN = 0.4
    thresholds_pur = [0.10, 0.15, 0.20, 0.25, 0.35]

    for pred_full, true_A_full, A_unique, title in [
        (pred_AL_full, A_L_full, A_L_unique, 'A_L'),
        (pred_AH_full, A_H_full, A_H_unique, 'A_H'),
    ]:
        a_list = sorted(np.unique(true_A_full))
        purity_table = np.full((len(a_list), len(thresholds_pur)), np.nan)
        n_table      = np.full((len(a_list), len(thresholds_pur)), 0, dtype=int)

        for j, thr in enumerate(thresholds_pur):
            mask_conf = conf_full >= thr
            for i, a in enumerate(a_list):
                mask_peak = ((pred_full >= a - HALF_WIN) &
                             (pred_full <= a + HALF_WIN) & mask_conf)
                if mask_peak.sum() < 5: continue
                purity_table[i, j] = (true_A_full[mask_peak] == a).mean()
                n_table[i, j]      = mask_peak.sum()

        fig, axes = plt.subplots(1, 2, figsize=(18, 10))
        ax = axes[0]
        im = ax.imshow(purity_table, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
        ax.set_xticks(range(len(thresholds_pur)))
        ax.set_xticklabels([f'{t:.2f}' for t in thresholds_pur])
        step = max(1, len(a_list) // 30)
        ax.set_yticks(range(0, len(a_list), step))
        ax.set_yticklabels(a_list[::step], fontsize=7)
        ax.set_xlabel('Seuil confiance'); ax.set_ylabel(f'{title} vrai')
        ax.set_title(f'Pureté — {title}\nP({title}_vrai=A | pred∈[A±{HALF_WIN}], conf≥seuil)',
                     fontsize=9)
        plt.colorbar(im, ax=ax)
        for i in range(0, len(a_list), step):
            for j in range(len(thresholds_pur)):
                if not np.isnan(purity_table[i, j]):
                    ax.text(j, i, f'{purity_table[i,j]:.2f}',
                            ha='center', va='center', fontsize=5,
                            color='black' if purity_table[i,j] < 0.85 else 'white')

        ax = axes[1]
        im2 = ax.imshow(n_table.astype(float), cmap='Blues', aspect='auto')
        ax.set_xticks(range(len(thresholds_pur)))
        ax.set_xticklabels([f'{t:.2f}' for t in thresholds_pur])
        ax.set_yticks(range(0, len(a_list), step))
        ax.set_yticklabels(a_list[::step], fontsize=7)
        ax.set_xlabel('Seuil confiance'); ax.set_ylabel(f'{title} vrai')
        ax.set_title(f'N events — {title}\n(pred∈[A±{HALF_WIN}], conf≥seuil)', fontsize=9)
        plt.colorbar(im2, ax=ax)

        fig.suptitle(f'Pureté et statistique — {title}  fenêtre ±{HALF_WIN}', fontsize=11)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Fig 8 : pureté toute stat sans coupure
    fig, axes = plt.subplots(1, 2, figsize=(8, 10))
    for ax_pur, ax_n, pred_full, true_A_full, A_unique, title in [
        (axes[0], axes[1], pred_AL_full, A_L_full, A_L_unique, 'A_L'),
    ]:
        a_list = sorted(np.unique(true_A_full))
        pur_all = np.full((len(a_list), 1), np.nan)
        n_all   = np.full((len(a_list), 1), 0, dtype=int)
        for i, a in enumerate(a_list):
            mask = (pred_full >= a - HALF_WIN) & (pred_full <= a + HALF_WIN)
            if mask.sum() < 5: continue
            pur_all[i, 0] = (true_A_full[mask] == a).mean()
            n_all[i, 0]   = mask.sum()

        im = ax_pur.imshow(pur_all, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
        ax_pur.set_xticks([0]); ax_pur.set_xticklabels(['Toute stat'])
        step = max(1, len(a_list) // 30)
        ax_pur.set_yticks(range(0, len(a_list), step))
        ax_pur.set_yticklabels(a_list[::step], fontsize=7)
        ax_pur.set_ylabel(f'{title} vrai')
        ax_pur.set_title(f'Pureté {title}\ntoute stat', fontsize=9)
        plt.colorbar(im, ax=ax_pur)

        im2 = ax_n.imshow(n_all.astype(float), cmap='Blues', aspect='auto')
        ax_n.set_xticks([0]); ax_n.set_xticklabels(['Toute stat'])
        ax_n.set_yticks(range(0, len(a_list), step))
        ax_n.set_yticklabels(a_list[::step], fontsize=7)
        ax_n.set_ylabel(f'{title} vrai')
        ax_n.set_title(f'N events {title}\ntoute stat', fontsize=9)
        plt.colorbar(im2, ax=ax_n)

    fig.suptitle(f'Pureté toute stat — fenêtre ±{HALF_WIN}', fontsize=11)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

print(f"\nPDF généré : {OUT_PDF}")
print(f"A_L : acc={acc_AL:.3f}  ±1={acc_pm1_AL:.3f}  ±2={acc_pm2_AL:.3f}"
      f"  σ={res_reg_AL.std():.3f}")
print(f"A_H : acc={acc_AH:.3f}  ±1={acc_pm1_AH:.3f}  ±2={acc_pm2_AH:.3f}"
      f"  σ={res_reg_AH.std():.3f}")
