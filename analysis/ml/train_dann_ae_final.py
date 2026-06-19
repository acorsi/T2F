"""
train_dann_ae_final.py
======================
Pipeline simplifié DANN-AE :
  - AE pur (pas de VAE, pas de KL, pas de reparameterize)
  - Pas de features physiques (R, ME0, A_L_kin)
  - Pas de triplet loss
  - Classifieur sur [z_L ⊕ z_H] uniquement
  - DANN pour alignement domaine source/cible
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
H5_SOURCE  = '../../simulation_output/T2F_G4_Corr.h5'
H5_TARGET  = '../../simulation_output/T2F_G4.h5'
OUT_PDF    = 'explore_dann_ae_final.pdf'
OUT_MODEL  = 'dann_ae_final_model.pt'

PAD_LEN      = 50
LATENT_DIM   = 8
HIDDEN_AE    = [64, 64, 64]
HIDDEN_CLF   = [32, 32]
HIDDEN_DISC  = [32, 32]
CLF_IN_DIM   = LATENT_DIM * 2   # z_L ⊕ z_H

LAMBDA_SMOOTH = 0.5
LAMBDA_CLF    = 0.1
LAMBDA_DANN   = 0.01

BATCH_SIZE  = 2048
N_EPOCHS    = 250
LR          = 1e-3
VAL_FRAC    = 0.1

ES_PATIENCE  = 50
LR_PATIENCE  = 10
LR_FACTOR    = 0.5
LR_MIN       = 1e-5

DEVICE = ('mps' if torch.backends.mps.is_available() else
          'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")

# ── Gradient Reversal Layer ───────────────────────────────────────────────────
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

# ── Architecture ──────────────────────────────────────────────────────────────
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
        x = torch.cat([z_L, z_H], dim=1)
        x = grad_reverse(x, alpha)
        return self.net(x)

class DANN_AE(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.encoder_L    = Encoder();  self.decoder_L = Decoder()
        self.encoder_H    = Encoder();  self.decoder_H = Decoder()
        self.discriminator = DomainDiscriminator()
        dims = [CLF_IN_DIM] + HIDDEN_CLF
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.LayerNorm(dims[i+1]), nn.GELU()]
        layers += [nn.Linear(HIDDEN_CLF[-1], n_classes)]
        self.clf = nn.Sequential(*layers)

    def forward(self, xL, xH, alpha=1.0):
        z_L = self.encoder_L(xL)
        z_H = self.encoder_H(xH)
        xhL = self.decoder_L(z_L)
        xhH = self.decoder_H(z_H)
        logits        = self.clf(torch.cat([z_L, z_H], dim=1))
        domain_logits = self.discriminator(z_L, z_H, alpha)
        return xhL, xhH, z_L, z_H, logits, domain_logits

# ── Loss ──────────────────────────────────────────────────────────────────────
ce_loss = nn.CrossEntropyLoss()

def recon_loss(x, x_hat, mask):
    n_valid = mask.sum(dim=1).clamp(min=1)
    mse     = ((x_hat - x)**2 * mask).sum(dim=1) / n_valid
    d2      = x_hat[:, 2:] - 2*x_hat[:, 1:-1] + x_hat[:, :-2]
    m_d2    = mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]
    n_d2    = m_d2.sum(dim=1).clamp(min=1)
    smooth  = ((d2**2) * m_d2).sum(dim=1) / n_d2
    return mse.mean(), smooth.mean()

def loss_labeled(xL, xH, mL, mH, xhL, xhH, logits, y,
                 domain_logits, domain_label):
    mse_L, sm_L = recon_loss(xL, xhL, mL)
    mse_H, sm_H = recon_loss(xH, xhH, mH)
    mse    = (mse_L + mse_H) / 2
    smooth = (sm_L  + sm_H)  / 2
    clf    = ce_loss(logits, y)
    domain = ce_loss(domain_logits, domain_label)
    total  = mse + LAMBDA_SMOOTH*smooth + LAMBDA_CLF*clf + LAMBDA_DANN*domain
    return total, mse, smooth, clf, domain

def loss_unlabeled(xL, xH, mL, mH, xhL, xhH, domain_logits, domain_label):
    mse_L, sm_L = recon_loss(xL, xhL, mL)
    mse_H, sm_H = recon_loss(xH, xhH, mH)
    recon  = (mse_L + mse_H) / 2 + LAMBDA_SMOOTH * (sm_L + sm_H) / 2
    domain = ce_loss(domain_logits, domain_label)
    return recon + LAMBDA_DANN * domain

# ── Chargement ────────────────────────────────────────────────────────────────
def load_dataset(path, labeled_frac=1.0):
    with h5py.File(path, 'r') as f:
        total     = f['L/Z'].shape[0]
        n_labeled = int(total * labeled_frac)
        print(f"  {path.split('/')[-1]} : total={total}  labelisés={n_labeled}")
        sl_lab = slice(0, n_labeled)
        data = dict(
            raw_L_lab = f['L/raw_trace'][sl_lab],
            raw_H_lab = f['H/raw_trace'][sl_lab],
            Z_L_lab   = f['L/Z'][sl_lab].astype(int),
        )
        if labeled_frac < 1.0:
            sl_unl = slice(n_labeled, total)
            data.update(dict(
                raw_L_unl = f['L/raw_trace'][sl_unl],
                raw_H_unl = f['H/raw_trace'][sl_unl],
            ))
    return data, total, n_labeled

def pad_traces(raw):
    N = len(raw)
    traces = np.zeros((N, PAD_LEN), dtype=np.float32)
    masks  = np.zeros((N, PAD_LEN), dtype=np.float32)
    for i, t in enumerate(raw):
        k = min(len(t), PAD_LEN)
        traces[i, :k] = t[:k]; masks[i, :k] = 1.0
    return traces, masks

print("\nChargement données...")
t0 = time.time()
src, src_total, src_n_lab = load_dataset(H5_SOURCE, labeled_frac=1.0)
tgt, tgt_total, tgt_n_lab = load_dataset(H5_TARGET, labeled_frac=0.1)

print("  Padding traces...")
src_tL, src_mL = pad_traces(src['raw_L_lab']); src_tH, src_mH = pad_traces(src['raw_H_lab'])
tgt_tL_lab, tgt_mL_lab = pad_traces(tgt['raw_L_lab'])
tgt_tH_lab, tgt_mH_lab = pad_traces(tgt['raw_H_lab'])
tgt_tL_unl, tgt_mL_unl = pad_traces(tgt['raw_L_unl'])
tgt_tH_unl, tgt_mH_unl = pad_traces(tgt['raw_H_unl'])
print(f"  Chargement terminé en {time.time()-t0:.1f}s")

trace_max = max(src_tL.max(), src_tH.max(),
                tgt_tL_lab.max(), tgt_tH_lab.max(),
                tgt_tL_unl.max(), tgt_tH_unl.max())
for arr in [src_tL, src_tH, tgt_tL_lab, tgt_tH_lab, tgt_tL_unl, tgt_tH_unl]:
    arr /= trace_max

Z_L_unique = np.unique(src['Z_L_lab'])
z_to_idx   = {z: i for i, z in enumerate(Z_L_unique)}
n_classes  = len(Z_L_unique)

def make_y(Z): return np.array([z_to_idx[z] for z in Z], dtype=np.int64)

src_y = make_y(src['Z_L_lab'])
tgt_y = make_y(tgt['Z_L_lab'])

rng           = np.random.default_rng(42)
idx_all       = rng.permutation(tgt_n_lab)
n_val         = int(tgt_n_lab * VAL_FRAC)
idx_train_tgt = idx_all[:-n_val]
idx_val       = idx_all[-n_val:]

n_src = src_total

# ── GPU ───────────────────────────────────────────────────────────────────────
print("\nPréchargement GPU...")
def to_gpu(arr): return torch.from_numpy(arr).to(DEVICE)

src_tL_gpu = to_gpu(src_tL); src_mL_gpu = to_gpu(src_mL)
src_tH_gpu = to_gpu(src_tH); src_mH_gpu = to_gpu(src_mH)
src_y_gpu  = to_gpu(src_y)

tgt_tL_lab_gpu = to_gpu(tgt_tL_lab); tgt_mL_lab_gpu = to_gpu(tgt_mL_lab)
tgt_tH_lab_gpu = to_gpu(tgt_tH_lab); tgt_mH_lab_gpu = to_gpu(tgt_mH_lab)
tgt_y_gpu      = to_gpu(tgt_y)

tgt_tL_unl_gpu = to_gpu(tgt_tL_unl); tgt_mL_unl_gpu = to_gpu(tgt_mL_unl)
tgt_tH_unl_gpu = to_gpu(tgt_tH_unl); tgt_mH_unl_gpu = to_gpu(tgt_mH_unl)

n_tgt_unl = len(tgt['raw_L_unl'])
print(f"  GPU chargé")

def iter_batches(n, bs, shuffle=True):
    idx = torch.randperm(n, device=DEVICE) if shuffle else torch.arange(n, device=DEVICE)
    for start in range(0, n, bs):
        yield idx[start:start+bs]

def domain_label(n, d):
    return torch.full((n,), d, dtype=torch.long, device=DEVICE)

# ── Entraînement ──────────────────────────────────────────────────────────────
model     = DANN_AE(n_classes).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN)
print(f"Paramètres : {sum(p.numel() for p in model.parameters()):,}")

history    = {k: [] for k in ['train', 'val', 'mse', 'clf', 'domain', 'acc']}
best_acc   = 0.0
best_state = None
es_counter = 0

for epoch in range(1, N_EPOCHS + 1):
    model.train()
    tot, n_batches = 0, 0

    p     = epoch / N_EPOCHS
    alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

    perm_src         = iter_batches(n_src, BATCH_SIZE)
    perm_tgt_unl_ptr = 0
    perm_tgt_unl     = torch.randperm(n_tgt_unl, device=DEVICE)

    for sl_tgt in iter_batches(len(idx_train_tgt), BATCH_SIZE):
        idx_t = torch.from_numpy(idx_train_tgt).to(DEVICE)[sl_tgt]

        # ── Cible labelisée ──
        xL = tgt_tL_lab_gpu[idx_t]; mL = tgt_mL_lab_gpu[idx_t]
        xH = tgt_tH_lab_gpu[idx_t]; mH = tgt_mH_lab_gpu[idx_t]
        y  = tgt_y_gpu[idx_t]
        xhL, xhH, zL, zH, logits, d_logits = model(xL, xH, alpha)
        loss_t, *_ = loss_labeled(xL, xH, mL, mH, xhL, xhH, logits, y,
                                   d_logits, domain_label(len(idx_t), 1))

        # ── Source ──
        try:
            sl_src = next(perm_src)
        except StopIteration:
            perm_src = iter_batches(n_src, BATCH_SIZE)
            sl_src   = next(perm_src)
        xL  = src_tL_gpu[sl_src]; mL  = src_mL_gpu[sl_src]
        xH  = src_tH_gpu[sl_src]; mH  = src_mH_gpu[sl_src]
        y_s = src_y_gpu[sl_src]
        xhL, xhH, zL, zH, logits, d_logits = model(xL, xH, alpha)
        loss_s, *_ = loss_labeled(xL, xH, mL, mH, xhL, xhH, logits, y_s,
                                   d_logits, domain_label(len(sl_src), 0))

        # ── Cible non-labelisée ──
        sl_u = perm_tgt_unl[perm_tgt_unl_ptr:perm_tgt_unl_ptr+BATCH_SIZE]
        perm_tgt_unl_ptr = (perm_tgt_unl_ptr + BATCH_SIZE) % n_tgt_unl
        xLu = tgt_tL_unl_gpu[sl_u]; mLu = tgt_mL_unl_gpu[sl_u]
        xHu = tgt_tH_unl_gpu[sl_u]; mHu = tgt_mH_unl_gpu[sl_u]
        xhLu, xhHu, _, _, _, d_logits_u = model(xLu, xHu, alpha)
        loss_u = loss_unlabeled(xLu, xHu, mLu, mHu, xhLu, xhHu,
                                d_logits_u, domain_label(len(sl_u), 1))

        loss = loss_t + loss_s + loss_u
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        tot += (loss_t + loss_s).item(); n_batches += 1

    history['train'].append(tot / max(n_batches, 1))

    # ── Validation ───────────────────────────────────────────────────────────
    model.eval()
    tv, tm, tc, td, n_correct, n_total = 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for sl in iter_batches(len(idx_val), BATCH_SIZE, shuffle=False):
            idx_v = torch.from_numpy(idx_val).to(DEVICE)[sl]
            xL = tgt_tL_lab_gpu[idx_v]; mL = tgt_mL_lab_gpu[idx_v]
            xH = tgt_tH_lab_gpu[idx_v]; mH = tgt_mH_lab_gpu[idx_v]
            y  = tgt_y_gpu[idx_v]
            xhL, xhH, zL, zH, logits, d_logits = model(xL, xH, alpha=0.0)
            loss, mse, smooth, clf, domain = loss_labeled(
                xL, xH, mL, mH, xhL, xhH, logits, y,
                d_logits, domain_label(len(idx_v), 1))
            tv += loss.item(); tm += mse.item()
            tc += clf.item();  td += domain.item()
            n_correct += (logits.argmax(1) == y).sum().item()
            n_total   += len(y)

    nv  = max(len(idx_val) // BATCH_SIZE, 1)
    acc = n_correct / n_total
    history['val'].append(tv/nv);  history['mse'].append(tm/nv)
    history['clf'].append(tc/nv);  history['domain'].append(td/nv)
    history['acc'].append(acc)

    scheduler.step(acc)
    cur_lr = optimizer.param_groups[0]['lr']
    if acc > best_acc:
        best_acc   = acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        es_counter = 0
    else:
        es_counter += 1

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{N_EPOCHS}  train={history['train'][-1]:.5f}  "
              f"val={history['val'][-1]:.5f}  acc={acc:.3f}  "
              f"domain={history['domain'][-1]:.4f}  "
              f"α={alpha:.3f}  lr={cur_lr:.2e}  es={es_counter}/{ES_PATIENCE}",
              flush=True)

    if es_counter >= ES_PATIENCE:
        print(f"Early stopping à epoch {epoch} (best acc={best_acc:.4f})", flush=True)
        break

model.load_state_dict(best_state)
torch.save({
    'model_state': model.state_dict(),
    'trace_max':   trace_max,
    'Z_L_unique':  Z_L_unique,
}, OUT_MODEL)
print(f"Modèle sauvegardé : {OUT_MODEL}")

# ── Collecte val set ──────────────────────────────────────────────────────────
print("Collecte val set...")
model.eval()
all_logits = []
ZL_val = tgt['Z_L_lab'][idx_val]

with torch.no_grad():
    for sl in iter_batches(len(idx_val), BATCH_SIZE, shuffle=False):
        idx_v = torch.from_numpy(idx_val).to(DEVICE)[sl]
        xL = tgt_tL_lab_gpu[idx_v]; xH = tgt_tH_lab_gpu[idx_v]
        _, _, _, _, logits, _ = model(xL, xH, alpha=0.0)
        all_logits.append(logits.cpu().numpy())

logits_v  = np.concatenate(all_logits)
proba_val = torch.softmax(torch.from_numpy(logits_v), dim=1).numpy()
pred_Z    = Z_L_unique[proba_val.argmax(axis=1)]
true_Z    = ZL_val[:len(pred_Z)]
conf      = proba_val.max(axis=1)

acc_strict = (pred_Z == true_Z).mean()
acc_pm1    = (np.abs(pred_Z - true_Z) <= 1).mean()
print(f"\nAccuracy stricte (cible) : {acc_strict:.3f}  |  ±1 : {acc_pm1:.3f}")

thresholds = np.percentile(conf, np.linspace(0, 90, 60))
acc_cov, cov_cov = [], []
for thr in thresholds:
    mask = conf >= thr
    if mask.sum() < 10: continue
    acc_cov.append((pred_Z[mask] == true_Z[mask]).mean())
    cov_cov.append(mask.mean())

# ── Figures ───────────────────────────────────────────────────────────────────
print(f"Génération {OUT_PDF}...")
with PdfPages(OUT_PDF) as pdf:

    # Fig 1 : loss curves
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle('DANN-AE final — traces seules, AE+DANN+CLF', fontsize=11)
    ep = range(1, len(history['train']) + 1)
    for ax, key, title, col in zip(axes,
        ['train', 'mse', 'clf', 'domain', 'acc'],
        ['Loss totale', 'MSE recon', 'CE classif', 'Domain loss', 'Accuracy'],
        ['steelblue', 'coral', 'seagreen', 'orchid', 'goldenrod']):
        if key == 'train':
            ax.plot(ep, history['train'], color='steelblue', label='train')
            ax.plot(ep, history['val'],   color='coral',     label='val')
            ax.legend(fontsize=8)
        else:
            ax.plot(ep, history[key], color=col)
        ax.set_title(title, fontsize=9); ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Fig 2 : accuracy vs coverage
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(cov_cov, acc_cov, 'b-o', markersize=3, label='DANN-AE final')
    ax.set_xlabel('Coverage'); ax.set_ylabel('Accuracy stricte Z_L')
    ax.set_title(f'DANN-AE final — accuracy sur domaine cible\n'
                 f'acc={acc_strict:.3f}, ±1={acc_pm1:.3f}', fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Fig 3 : matrice de confusion
    z_min, z_max = true_Z.min(), true_Z.max()
    z_range  = np.arange(z_min, z_max + 1)
    z_to_row = {z: i for i, z in enumerate(z_range)}
    cm = np.zeros((len(z_range), len(z_range)), dtype=int)
    for t, p in zip(true_Z, pred_Z):
        if t in z_to_row and p in z_to_row:
            cm[z_to_row[t], z_to_row[p]] += 1
    cm_n = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_n, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(z_range))); ax.set_xticklabels(z_range, fontsize=6, rotation=90)
    ax.set_yticks(range(len(z_range))); ax.set_yticklabels(z_range, fontsize=6)
    ax.set_xlabel('Z_L prédit'); ax.set_ylabel('Z_L vrai')
    ax.set_title(f'Matrice de confusion — acc={acc_strict:.3f}', fontsize=11)
    plt.colorbar(im, ax=ax)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Fig 4 : PCA 2D des z_L encodés (centroïdes par classe)
    all_zL, all_y = [], []
    with torch.no_grad():
        for sl in iter_batches(len(idx_val), BATCH_SIZE, shuffle=False):
            idx_v = torch.from_numpy(idx_val).to(DEVICE)[sl]
            xL = tgt_tL_lab_gpu[idx_v]; xH = tgt_tH_lab_gpu[idx_v]
            _, _, zL, _, _, _ = model(xL, xH, alpha=0.0)
            all_zL.append(zL.cpu().numpy())
            all_y.append(tgt_y_gpu[idx_v].cpu().numpy())
    zL_np = np.concatenate(all_zL)
    y_np  = np.concatenate(all_y)

    centroids_list, valid_z = [], []
    for c in range(n_classes):
        mask = (y_np == c)
        if mask.sum() > 0:
            centroids_list.append(zL_np[mask].mean(axis=0))
            valid_z.append(Z_L_unique[c])
    centroids_np = np.array(centroids_list)
    valid_z_np   = np.array(valid_z)

    if LATENT_DIM >= 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        c2d = pca.fit_transform(centroids_np)
        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(c2d[:, 0], c2d[:, 1],
                        c=valid_z_np, cmap='viridis', s=80, zorder=3)
        for i, z in enumerate(valid_z_np):
            ax.annotate(str(z), (c2d[i, 0], c2d[i, 1]),
                        fontsize=7, ha='center', va='bottom')
        plt.colorbar(sc, ax=ax, label='Z_L')
        ax.set_title('Centroïdes z_L encodés — PCA 2D\nStructure ordinale en Z',
                     fontsize=11)
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
        ax.grid(True, alpha=0.3)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

print(f"\nPDF généré : {OUT_PDF}")
print(f"Accuracy stricte (cible) : {acc_strict:.3f}  |  ±1 : {acc_pm1:.3f}")
