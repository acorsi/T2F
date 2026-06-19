"""
train_nu.py
===========
Prédiction de nu_L et nu_H (multiplicité de neutrons, 0-9) à partir de
l'encodeur A figé (dann_ae_mass_kin_model_G4.pt).

Pipeline :
  - encoder_L, encoder_H figés (poids gelés, pas de fine-tuning)
  - z_L = encoder_L(xL), z_H = encoder_H(xH)   (dim 8 chacun)
  - clf_nuL, clf_nuH : [z_L ⊕ z_H] (dim 16) → 10 classes (nu=0..9)
  - Loss = CE(nu_L) + CE(nu_H)
  - Entraînement sur 100% du jeu source labelisé (T2F_LG_Corr.h5)
  - Pas de DANN, pas de AE loss, pas de A_kin
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
H5_SOURCE    = '../simulation_output/T2F_G4_Corr.h5'
H5_VALID_G4  = '../simulation_output/T2F_G4.h5'
MODEL_A_PATH = 'dann_ae_mass_kin_model_G4.pt'
OUT_PDF      = 'explore_nu.pdf'
OUT_MODEL    = 'nu_heads_model.pt'

PAD_LEN     = 50
HIDDEN_CLF  = [32, 32]
N_CLASSES   = 10   # nu = 0..9

BATCH_SIZE  = 2048
N_EPOCHS    = 200
LR          = 1e-3
VAL_FRAC    = 0.1
ES_PATIENCE = 30
LR_PATIENCE = 8
LR_FACTOR   = 0.5
LR_MIN      = 1e-5

DEVICE = ('mps' if torch.backends.mps.is_available() else
          'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")

# ── Architecture encodeur (identique à ModelA) ───────────────────────────────
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

LATENT_DIM = 8
HIDDEN_AE  = [256, 128, 64]

encoder_L = make_encoder(HIDDEN_AE, LATENT_DIM)
encoder_H = make_encoder(HIDDEN_AE, LATENT_DIM)

print(f"Chargement encodeurs depuis {MODEL_A_PATH}...")
ckpt_A = torch.load(MODEL_A_PATH, map_location=DEVICE, weights_only=False)
state  = ckpt_A['model_state']

enc_L_state = {k[len('encoder_L.'):]: v for k, v in state.items() if k.startswith('encoder_L.')}
enc_H_state = {k[len('encoder_H.'):]: v for k, v in state.items() if k.startswith('encoder_H.')}
encoder_L.load_state_dict(enc_L_state)
encoder_H.load_state_dict(enc_H_state)

encoder_L.to(DEVICE).eval()
encoder_H.to(DEVICE).eval()
for p in encoder_L.parameters(): p.requires_grad = False
for p in encoder_H.parameters(): p.requires_grad = False

trace_max = ckpt_A['trace_max']
print(f"  trace_max = {trace_max}")

# ── Têtes nu ──────────────────────────────────────────────────────────────────
def make_clf_head(clf_in, hidden, n_cls):
    dims = [clf_in] + hidden
    layers = []
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i+1]), nn.LayerNorm(dims[i+1]), nn.GELU()]
    layers.append(nn.Linear(hidden[-1], n_cls))
    return nn.Sequential(*layers)

clf_nuL = make_clf_head(LATENT_DIM*2, HIDDEN_CLF, N_CLASSES).to(DEVICE)
clf_nuH = make_clf_head(LATENT_DIM*2, HIDDEN_CLF, N_CLASSES).to(DEVICE)

ce_loss = nn.CrossEntropyLoss()
params  = list(clf_nuL.parameters()) + list(clf_nuH.parameters())
optimizer = torch.optim.Adam(params, lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN)
print(f"Paramètres entraînables : {sum(p.numel() for p in params):,}")

# ── Chargement données ────────────────────────────────────────────────────────
print(f"\nChargement {H5_SOURCE}...")
with h5py.File(H5_SOURCE, 'r') as f:
    total  = f['L/Z'].shape[0]
    raw_L  = f['L/raw_trace'][:]
    raw_H  = f['H/raw_trace'][:]
    nu_L   = f['L/n'][:].astype(int)
    nu_H   = f['H/n'][:].astype(int)
print(f"  total={total}")
print(f"  nu_L : {nu_L.min()}..{nu_L.max()}   nu_H : {nu_H.min()}..{nu_H.max()}")

# Clip aux classes 0..9
nu_L = np.clip(nu_L, 0, N_CLASSES-1)
nu_H = np.clip(nu_H, 0, N_CLASSES-1)

def pad_traces(raw):
    N = len(raw)
    t = np.zeros((N, PAD_LEN), dtype=np.float32)
    for i, tr in enumerate(raw):
        k = min(len(tr), PAD_LEN)
        t[i, :k] = tr[:k]
    return t

print("  Padding traces...")
tL = pad_traces(raw_L) / trace_max
tH = pad_traces(raw_H) / trace_max

rng     = np.random.default_rng(42)
idx_all = rng.permutation(total)
n_val   = int(total * VAL_FRAC)
idx_train = idx_all[:-n_val]
idx_val   = idx_all[-n_val:]
print(f"  train={len(idx_train)}  val={len(idx_val)}")

def to_gpu(arr): return torch.from_numpy(arr).to(DEVICE)
tL_gpu = to_gpu(tL); tH_gpu = to_gpu(tH)
nuL_gpu = to_gpu(nu_L.astype(np.int64))
nuH_gpu = to_gpu(nu_H.astype(np.int64))

def iter_batches(n, bs, shuffle=True):
    idx = torch.randperm(n, device=DEVICE) if shuffle else torch.arange(n, device=DEVICE)
    for start in range(0, n, bs):
        yield idx[start:start+bs]

# ── Extraction features (encodeur figé, calcul une seule fois) ───────────────
print("\nExtraction features latentes (encodeur figé)...")
t0 = time.time()
all_z = []
with torch.no_grad():
    for s in range(0, total, BATCH_SIZE):
        sl = slice(s, s + BATCH_SIZE)
        zL = encoder_L(tL_gpu[sl])
        zH = encoder_H(tH_gpu[sl])
        all_z.append(torch.cat([zL, zH], dim=1))
Z_feat = torch.cat(all_z, dim=0)   # (N, 16)
print(f"  Z_feat shape={Z_feat.shape}  en {time.time()-t0:.1f}s")

idx_train_t = torch.from_numpy(idx_train).to(DEVICE)
idx_val_t   = torch.from_numpy(idx_val).to(DEVICE)

# ── Entraînement ──────────────────────────────────────────────────────────────
history = {k: [] for k in ['train', 'val', 'acc_nuL', 'acc_nuH']}
best_acc, best_state, es_counter = 0.0, None, 0

for epoch in range(1, N_EPOCHS + 1):
    clf_nuL.train(); clf_nuH.train()
    tot, n_batches = 0, 0
    for sl in iter_batches(len(idx_train), BATCH_SIZE):
        idx_b = idx_train_t[sl]
        z   = Z_feat[idx_b]
        yL  = nuL_gpu[idx_b]; yH = nuH_gpu[idx_b]
        logits_L = clf_nuL(z); logits_H = clf_nuH(z)
        loss = ce_loss(logits_L, yL) + ce_loss(logits_H, yH)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        tot += loss.item(); n_batches += 1
    history['train'].append(tot / max(n_batches, 1))

    # Validation
    clf_nuL.eval(); clf_nuH.eval()
    tv, n_correct_L, n_correct_H, n_total = 0, 0, 0, 0
    with torch.no_grad():
        for sl in iter_batches(len(idx_val), BATCH_SIZE, shuffle=False):
            idx_b = idx_val_t[sl]
            z  = Z_feat[idx_b]
            yL = nuL_gpu[idx_b]; yH = nuH_gpu[idx_b]
            logits_L = clf_nuL(z); logits_H = clf_nuH(z)
            loss = ce_loss(logits_L, yL) + ce_loss(logits_H, yH)
            tv += loss.item()
            n_correct_L += (logits_L.argmax(1) == yL).sum().item()
            n_correct_H += (logits_H.argmax(1) == yH).sum().item()
            n_total     += len(yL)

    nv = max(len(idx_val) // BATCH_SIZE, 1)
    acc_L = n_correct_L / n_total
    acc_H = n_correct_H / n_total
    acc   = (acc_L + acc_H) / 2
    history['val'].append(tv/nv)
    history['acc_nuL'].append(acc_L); history['acc_nuH'].append(acc_H)

    scheduler.step(acc)
    cur_lr = optimizer.param_groups[0]['lr']
    if acc > best_acc:
        best_acc = acc
        best_state = {'clf_nuL': {k: v.cpu().clone() for k, v in clf_nuL.state_dict().items()},
                       'clf_nuH': {k: v.cpu().clone() for k, v in clf_nuH.state_dict().items()}}
        es_counter = 0
    else:
        es_counter += 1

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{N_EPOCHS}  train={history['train'][-1]:.5f}  "
              f"val={history['val'][-1]:.5f}  acc_nuL={acc_L:.3f}  acc_nuH={acc_H:.3f}  "
              f"lr={cur_lr:.2e}  es={es_counter}/{ES_PATIENCE}", flush=True)

    if es_counter >= ES_PATIENCE:
        print(f"Early stopping à epoch {epoch} (best acc={best_acc:.4f})", flush=True)
        break

clf_nuL.load_state_dict(best_state['clf_nuL'])
clf_nuH.load_state_dict(best_state['clf_nuH'])
torch.save({
    'clf_nuL_state': clf_nuL.state_dict(),
    'clf_nuH_state': clf_nuH.state_dict(),
    'N_CLASSES': N_CLASSES,
    'HIDDEN_CLF': HIDDEN_CLF,
    'LATENT_DIM': LATENT_DIM,
}, OUT_MODEL)
print(f"Modèle sauvegardé : {OUT_MODEL}")

# ── Collecte val set ──────────────────────────────────────────────────────────
print("Collecte val set...")
clf_nuL.eval(); clf_nuH.eval()
with torch.no_grad():
    z_val  = Z_feat[idx_val_t]
    logits_L = clf_nuL(z_val); logits_H = clf_nuH(z_val)
pred_nuL = logits_L.argmax(1).cpu().numpy()
pred_nuH = logits_H.argmax(1).cpu().numpy()
true_nuL = nu_L[idx_val]
true_nuH = nu_H[idx_val]

acc_nuL = (pred_nuL == true_nuL).mean()
acc_nuH = (pred_nuH == true_nuH).mean()
pm1_nuL = (np.abs(pred_nuL - true_nuL) <= 1).mean()
pm1_nuH = (np.abs(pred_nuH - true_nuH) <= 1).mean()

print(f"\nnu_L : acc={acc_nuL:.3f}  ±1={pm1_nuL:.3f}")
print(f"nu_H : acc={acc_nuH:.3f}  ±1={pm1_nuH:.3f}")

# Distribution nu_total
nu_total_pred = pred_nuL + pred_nuH
nu_total_true = true_nuL + true_nuH
print(f"nu_bar pred = {nu_total_pred.mean():.3f}  |  nu_bar true = {nu_total_true.mean():.3f}")
print(f"Référence ²⁵²Cf : nu_bar = 3.757")

# ── Figures ───────────────────────────────────────────────────────────────────
print(f"\nGénération {OUT_PDF}...")
with PdfPages(OUT_PDF) as pdf:

    # Fig 1 : loss/acc curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    ep = range(1, len(history['train']) + 1)
    axes[0].plot(ep, history['train'], label='train', color='steelblue')
    axes[0].plot(ep, history['val'],   label='val',   color='coral')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(ep, history['acc_nuL'], color='steelblue', label='nu_L')
    axes[1].plot(ep, history['acc_nuH'], color='coral',     label='nu_H')
    axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].axis('off')
    axes[2].text(0.1, 0.6, f"nu_L : acc={acc_nuL:.3f}  ±1={pm1_nuL:.3f}\n"
                            f"nu_H : acc={acc_nuH:.3f}  ±1={pm1_nuH:.3f}\n\n"
                            f"nu_bar pred = {nu_total_pred.mean():.3f}\n"
                            f"nu_bar true = {nu_total_true.mean():.3f}\n"
                            f"Réf ²⁵²Cf : 3.757", fontsize=11, family='monospace')
    fig.suptitle('Têtes nu_L / nu_H — encodeur A figé', fontsize=11)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Fig 2 : matrices de confusion
    for true_nu, pred_nu, title in [
        (true_nuL, pred_nuL, f'nu_L — acc={acc_nuL:.3f}  ±1={pm1_nuL:.3f}'),
        (true_nuH, pred_nuH, f'nu_H — acc={acc_nuH:.3f}  ±1={pm1_nuH:.3f}'),
    ]:
        cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
        for t, p in zip(true_nu, pred_nu):
            cm[t, p] += 1
        cm_n = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm_n, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
        ax.set_xlabel('nu prédit'); ax.set_ylabel('nu vrai')
        ax.set_title(title, fontsize=11)
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                if cm[i,j] > 0:
                    ax.text(j, i, f'{cm_n[i,j]:.2f}', ha='center', va='center',
                            fontsize=7, color='black' if cm_n[i,j]<0.5 else 'white')
        plt.colorbar(im, ax=ax)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Fig 3 : distribution nu_total
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(-0.5, 16.5, 1)
    ax.hist(nu_total_true, bins=bins, alpha=0.5, color='steelblue', density=True,
            label=f'Vrai  (moy={nu_total_true.mean():.3f})')
    ax.hist(nu_total_pred, bins=bins, alpha=0.5, color='coral', density=True,
            label=f'Prédit (moy={nu_total_pred.mean():.3f})')
    ax.axvline(3.757, color='green', ls='--', label='Réf ²⁵²Cf = 3.757')
    ax.set_xlabel('nu_L + nu_H'); ax.set_ylabel('Densité')
    ax.set_title('Distribution de nu_total', fontsize=11)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

print(f"\nPDF généré : {OUT_PDF}")
print(f"nu_L (val LG_Corr) : acc={acc_nuL:.3f}  ±1={pm1_nuL:.3f}")
print(f"nu_H (val LG_Corr) : acc={acc_nuH:.3f}  ±1={pm1_nuH:.3f}")

# ── Validation sur T2F_G4.h5 (100%) ──────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"Validation sur {H5_VALID_G4} (100%)...")
with h5py.File(H5_VALID_G4, 'r') as f:
    total_g4 = f['L/Z'].shape[0]
    raw_L_g4 = f['L/raw_trace'][:]
    raw_H_g4 = f['H/raw_trace'][:]
    nu_L_g4  = np.clip(f['L/n'][:].astype(int), 0, N_CLASSES-1)
    nu_H_g4  = np.clip(f['H/n'][:].astype(int), 0, N_CLASSES-1)
print(f"  total={total_g4}")

tL_g4 = pad_traces(raw_L_g4) / trace_max
tH_g4 = pad_traces(raw_H_g4) / trace_max
tL_g4_gpu = to_gpu(tL_g4); tH_g4_gpu = to_gpu(tH_g4)

print("  Extraction features + inférence...")
all_pred_L, all_pred_H = [], []
with torch.no_grad():
    for s in range(0, total_g4, BATCH_SIZE):
        sl = slice(s, s + BATCH_SIZE)
        zL = encoder_L(tL_g4_gpu[sl]); zH = encoder_H(tH_g4_gpu[sl])
        z  = torch.cat([zL, zH], dim=1)
        all_pred_L.append(clf_nuL(z).argmax(1).cpu().numpy())
        all_pred_H.append(clf_nuH(z).argmax(1).cpu().numpy())

pred_nuL_g4 = np.concatenate(all_pred_L)
pred_nuH_g4 = np.concatenate(all_pred_H)

acc_nuL_g4 = (pred_nuL_g4 == nu_L_g4).mean()
acc_nuH_g4 = (pred_nuH_g4 == nu_H_g4).mean()
pm1_nuL_g4 = (np.abs(pred_nuL_g4 - nu_L_g4) <= 1).mean()
pm1_nuH_g4 = (np.abs(pred_nuH_g4 - nu_H_g4) <= 1).mean()

nu_total_pred_g4 = pred_nuL_g4 + pred_nuH_g4
nu_total_true_g4 = nu_L_g4 + nu_H_g4

print(f"\nnu_L (G4) : acc={acc_nuL_g4:.3f}  ±1={pm1_nuL_g4:.3f}")
print(f"nu_H (G4) : acc={acc_nuH_g4:.3f}  ±1={pm1_nuH_g4:.3f}")
print(f"nu_bar pred (G4) = {nu_total_pred_g4.mean():.3f}  |  "
      f"nu_bar true (G4) = {nu_total_true_g4.mean():.3f}")
print(f"Référence ²⁵²Cf : nu_bar = 3.757")

# Figures additionnelles G4
with PdfPages(OUT_PDF.replace('.pdf', '_G4.pdf')) as pdf:

    # Matrices de confusion G4
    for true_nu, pred_nu, title in [
        (nu_L_g4, pred_nuL_g4, f'nu_L (G4) — acc={acc_nuL_g4:.3f}  ±1={pm1_nuL_g4:.3f}'),
        (nu_H_g4, pred_nuH_g4, f'nu_H (G4) — acc={acc_nuH_g4:.3f}  ±1={pm1_nuH_g4:.3f}'),
    ]:
        cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
        for t, p in zip(true_nu, pred_nu):
            cm[t, p] += 1
        cm_n = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm_n, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
        ax.set_xlabel('nu prédit'); ax.set_ylabel('nu vrai')
        ax.set_title(title, fontsize=11)
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                if cm[i,j] > 0:
                    ax.text(j, i, f'{cm_n[i,j]:.2f}', ha='center', va='center',
                            fontsize=7, color='black' if cm_n[i,j]<0.5 else 'white')
        plt.colorbar(im, ax=ax)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Distribution nu_total G4
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(-0.5, 16.5, 1)
    ax.hist(nu_total_true_g4, bins=bins, alpha=0.5, color='steelblue', density=True,
            label=f'Vrai  (moy={nu_total_true_g4.mean():.3f})')
    ax.hist(nu_total_pred_g4, bins=bins, alpha=0.5, color='coral', density=True,
            label=f'Prédit (moy={nu_total_pred_g4.mean():.3f})')
    ax.axvline(3.757, color='green', ls='--', label='Réf ²⁵²Cf = 3.757')
    ax.set_xlabel('nu_L + nu_H'); ax.set_ylabel('Densité')
    ax.set_title(f'Distribution de nu_total — validation G4 (N={total_g4:,})', fontsize=11)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

print(f"\nPDF G4 généré : {OUT_PDF.replace('.pdf', '_G4.pdf')}")
