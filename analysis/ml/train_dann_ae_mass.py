"""
train_dann_ae_mass_kin.py
=========================
Modifications par rapport à train_dann_ae_mass.py :
  - A_kin_L et A_kin_H calculés depuis les traces brutes (somme des pads)
    A_kin_L = 252 * KE_H / (KE_L + KE_H)
    A_kin_H = 252 * KE_L / (KE_L + KE_H)
  - A_kin_L ajouté comme feature de la tête clf_AL
  - A_kin_H ajouté comme feature de la tête clf_AH
  - Loss cinématique : relu(E[A_L] - A_kin_L)² + relu(E[A_H] - A_kin_H)²
  - CLF_IN_DIM = LATENT_DIM * 2 + 1 pour chaque tête
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
H5_SOURCE  = '../../simulation_output/T2F_G4_svd.h5'
H5_TARGET  = '../../simulation_output/T2F_G4_native.h5'
OUT_PDF    = 'explore_dann_ae_mass_kin_G4.pdf'
OUT_MODEL  = 'dann_ae_mass_kin_model_G4.pt'

PAD_LEN      = 50
LATENT_DIM   = 8
HIDDEN_AE    = [256, 128, 64]
HIDDEN_CLF   = [64, 64]
HIDDEN_DISC  = [32, 32]
CLF_IN_DIM   = LATENT_DIM * 2 + 1   # z_L ⊕ z_H ⊕ A_kin (scalaire)

A_TOTAL      = 252   # A_L + A_H ≤ 252

LAMBDA_SMOOTH = 0.5
LAMBDA_CLF    = 0.1
LAMBDA_DANN   = 0.01
LAMBDA_CONS   = 0.1   # poids contrainte A_L + A_H ≤ 252

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

    def forward(self, xL, xH, a_kin_L, a_kin_H, alpha=1.0):
        z_L = self.encoder_L(xL);  z_H = self.encoder_H(xH)
        xhL = self.decoder_L(z_L); xhH = self.decoder_H(z_H)
        # A_kin normalisé par 252 pour le classifieur, brut pour la loss
        clf_in_AL     = torch.cat([z_L, z_H, a_kin_L / 252.0], dim=1)
        clf_in_AH     = torch.cat([z_L, z_H, a_kin_H / 252.0], dim=1)
        logits_AL     = self.clf_AL(clf_in_AL)
        logits_AH     = self.clf_AH(clf_in_AH)
        domain_logits = self.discriminator(z_L, z_H, alpha)
        return xhL, xhH, z_L, z_H, logits_AL, logits_AH, domain_logits

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

def loss_labeled(xL, xH, mL, mH, xhL, xhH,
                 logits_AL, logits_AH, y_AL, y_AH,
                 domain_logits, d_label,
                 a_vals_AL, a_vals_AH,
                 a_kin_L, a_kin_H):
    mse_L, sm_L = recon_loss(xL, xhL, mL)
    mse_H, sm_H = recon_loss(xH, xhH, mH)
    mse    = (mse_L + mse_H) / 2
    smooth = (sm_L  + sm_H)  / 2
    clf_AL = ce_loss(logits_AL, y_AL)
    clf_AH = ce_loss(logits_AH, y_AH)
    domain = ce_loss(domain_logits, d_label)
    p_AL   = torch.softmax(logits_AL, dim=1)
    p_AH   = torch.softmax(logits_AH, dim=1)
    e_AL   = (p_AL * a_vals_AL).sum(dim=1)
    e_AH   = (p_AH * a_vals_AH).sum(dim=1)
    # Contrainte conservation A_L + A_H ≤ 252
    cons   = torch.relu(e_AL + e_AH - A_TOTAL).pow(2).mean()
    total  = (mse + LAMBDA_SMOOTH*smooth
              + LAMBDA_CLF*(clf_AL + clf_AH)
              + LAMBDA_DANN*domain
              + LAMBDA_CONS*cons)
    return total, mse, smooth, clf_AL, clf_AH, domain, cons

def loss_unlabeled(xL, xH, mL, mH, xhL, xhH, domain_logits, d_label):
    mse_L, sm_L = recon_loss(xL, xhL, mL)
    mse_H, sm_H = recon_loss(xH, xhH, mH)
    recon  = (mse_L + mse_H) / 2 + LAMBDA_SMOOTH * (sm_L + sm_H) / 2
    domain = ce_loss(domain_logits, d_label)
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
            A_L_lab   = f['L/A'][sl_lab].astype(int),
            A_H_lab   = f['H/A'][sl_lab].astype(int),
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

# ── Calcul A_kin AVANT normalisation ─────────────────────────────────────────
def compute_akin(tL, tH):
    """A_kin_L = 252 * KE_H / (KE_L + KE_H), en unités de A (0-252)."""
    KE_L   = tL.sum(axis=1)
    KE_H   = tH.sum(axis=1)
    denom  = np.clip(KE_L + KE_H, 1e-6, None)
    akin_L = (252.0 * KE_H / denom).astype(np.float32)   # ∈ [0, 252]
    akin_H = (252.0 * KE_L / denom).astype(np.float32)
    return akin_L[:, None], akin_H[:, None]   # (N,1)

src_akin_L,     src_akin_H     = compute_akin(src_tL, src_tH)
tgt_akin_L_lab, tgt_akin_H_lab = compute_akin(tgt_tL_lab, tgt_tH_lab)
tgt_akin_L_unl, tgt_akin_H_unl = compute_akin(tgt_tL_unl, tgt_tH_unl)

for arr in [src_tL, src_tH, tgt_tL_lab, tgt_tH_lab, tgt_tL_unl, tgt_tH_unl]:
    arr /= trace_max

# Classes A_L et A_H
A_L_unique = np.unique(src['A_L_lab'])
A_H_unique = np.unique(src['A_H_lab'])
aL_to_idx  = {a: i for i, a in enumerate(A_L_unique)}
aH_to_idx  = {a: i for i, a in enumerate(A_H_unique)}
n_classes_AL = len(A_L_unique)
n_classes_AH = len(A_H_unique)
print(f"  A_L : {A_L_unique.min()}..{A_L_unique.max()} ({n_classes_AL} classes)")
print(f"  A_H : {A_H_unique.min()}..{A_H_unique.max()} ({n_classes_AH} classes)")

def make_yAL(A): return np.array([aL_to_idx[a] for a in A], dtype=np.int64)
def make_yAH(A): return np.array([aH_to_idx[a] for a in A], dtype=np.int64)

src_yAL = make_yAL(src['A_L_lab']); src_yAH = make_yAH(src['A_H_lab'])
tgt_yAL = make_yAL(tgt['A_L_lab']); tgt_yAH = make_yAH(tgt['A_H_lab'])

rng           = np.random.default_rng(42)
idx_all       = rng.permutation(tgt_n_lab)
n_val         = int(tgt_n_lab * VAL_FRAC)
idx_train_tgt = idx_all[:-n_val]
idx_val       = idx_all[-n_val:]
n_src         = src_total

# ── GPU ───────────────────────────────────────────────────────────────────────
print("\nPréchargement GPU...")
def to_gpu(arr): return torch.from_numpy(arr).to(DEVICE)

src_tL_gpu = to_gpu(src_tL); src_mL_gpu = to_gpu(src_mL)
src_tH_gpu = to_gpu(src_tH); src_mH_gpu = to_gpu(src_mH)
src_yAL_gpu = to_gpu(src_yAL); src_yAH_gpu = to_gpu(src_yAH)
src_akin_L_gpu = to_gpu(src_akin_L); src_akin_H_gpu = to_gpu(src_akin_H)

tgt_tL_lab_gpu = to_gpu(tgt_tL_lab); tgt_mL_lab_gpu = to_gpu(tgt_mL_lab)
tgt_tH_lab_gpu = to_gpu(tgt_tH_lab); tgt_mH_lab_gpu = to_gpu(tgt_mH_lab)
tgt_yAL_gpu    = to_gpu(tgt_yAL);    tgt_yAH_gpu    = to_gpu(tgt_yAH)
tgt_akin_L_lab_gpu = to_gpu(tgt_akin_L_lab); tgt_akin_H_lab_gpu = to_gpu(tgt_akin_H_lab)

tgt_tL_unl_gpu = to_gpu(tgt_tL_unl); tgt_mL_unl_gpu = to_gpu(tgt_mL_unl)
tgt_tH_unl_gpu = to_gpu(tgt_tH_unl); tgt_mH_unl_gpu = to_gpu(tgt_mH_unl)
tgt_akin_L_unl_gpu = to_gpu(tgt_akin_L_unl); tgt_akin_H_unl_gpu = to_gpu(tgt_akin_H_unl)

# Vecteurs A pour la contrainte de conservation
a_vals_AL = torch.tensor(A_L_unique, dtype=torch.float32, device=DEVICE)
a_vals_AH = torch.tensor(A_H_unique, dtype=torch.float32, device=DEVICE)

n_tgt_unl = len(tgt['raw_L_unl'])
print(f"  GPU chargé")

def iter_batches(n, bs, shuffle=True):
    idx = torch.randperm(n, device=DEVICE) if shuffle else torch.arange(n, device=DEVICE)
    for start in range(0, n, bs):
        yield idx[start:start+bs]

def domain_label(n, d):
    return torch.full((n,), d, dtype=torch.long, device=DEVICE)

# ── Entraînement ──────────────────────────────────────────────────────────────
model     = DANN_AE_Mass(n_classes_AL, n_classes_AH).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN)
print(f"Paramètres : {sum(p.numel() for p in model.parameters()):,}")

history    = {k: [] for k in ['train', 'val', 'clf_AL', 'clf_AH',
                               'domain', 'cons', 'acc_AL', 'acc_AH']}
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
        xL  = tgt_tL_lab_gpu[idx_t]; mL  = tgt_mL_lab_gpu[idx_t]
        xH  = tgt_tH_lab_gpu[idx_t]; mH  = tgt_mH_lab_gpu[idx_t]
        yAL = tgt_yAL_gpu[idx_t];    yAH = tgt_yAH_gpu[idx_t]
        akL = tgt_akin_L_lab_gpu[idx_t]; akH = tgt_akin_H_lab_gpu[idx_t]
        xhL, xhH, zL, zH, logits_AL, logits_AH, d_logits = model(xL, xH, akL, akH, alpha)
        loss_t, *_ = loss_labeled(
            xL, xH, mL, mH, xhL, xhH,
            logits_AL, logits_AH, yAL, yAH,
            d_logits, domain_label(len(idx_t), 1),
            a_vals_AL, a_vals_AH, akL, akH)

        # ── Source ──
        try:
            sl_src = next(perm_src)
        except StopIteration:
            perm_src = iter_batches(n_src, BATCH_SIZE)
            sl_src   = next(perm_src)
        xL   = src_tL_gpu[sl_src]; mL   = src_mL_gpu[sl_src]
        xH   = src_tH_gpu[sl_src]; mH   = src_mH_gpu[sl_src]
        yALs = src_yAL_gpu[sl_src]; yAHs = src_yAH_gpu[sl_src]
        akLs = src_akin_L_gpu[sl_src]; akHs = src_akin_H_gpu[sl_src]
        xhL, xhH, zL, zH, logits_AL, logits_AH, d_logits = model(xL, xH, akLs, akHs, alpha)
        loss_s, *_ = loss_labeled(
            xL, xH, mL, mH, xhL, xhH,
            logits_AL, logits_AH, yALs, yAHs,
            d_logits, domain_label(len(sl_src), 0),
            a_vals_AL, a_vals_AH, akLs, akHs)

        # ── Cible non-labelisée ──
        sl_u = perm_tgt_unl[perm_tgt_unl_ptr:perm_tgt_unl_ptr+BATCH_SIZE]
        perm_tgt_unl_ptr = (perm_tgt_unl_ptr + BATCH_SIZE) % n_tgt_unl
        xLu = tgt_tL_unl_gpu[sl_u]; mLu = tgt_mL_unl_gpu[sl_u]
        xHu = tgt_tH_unl_gpu[sl_u]; mHu = tgt_mH_unl_gpu[sl_u]
        akLu = tgt_akin_L_unl_gpu[sl_u]; akHu = tgt_akin_H_unl_gpu[sl_u]
        xhLu, xhHu, _, _, _, _, d_logits_u = model(xLu, xHu, akLu, akHu, alpha)
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
    tv, tcAL, tcAH, td, tcons = 0, 0, 0, 0, 0
    n_correct_AL, n_correct_AH, n_total = 0, 0, 0
    with torch.no_grad():
        for sl in iter_batches(len(idx_val), BATCH_SIZE, shuffle=False):
            idx_v = torch.from_numpy(idx_val).to(DEVICE)[sl]
            xL  = tgt_tL_lab_gpu[idx_v]; mL  = tgt_mL_lab_gpu[idx_v]
            xH  = tgt_tH_lab_gpu[idx_v]; mH  = tgt_mH_lab_gpu[idx_v]
            yAL = tgt_yAL_gpu[idx_v];    yAH = tgt_yAH_gpu[idx_v]
            akL = tgt_akin_L_lab_gpu[idx_v]; akH = tgt_akin_H_lab_gpu[idx_v]
            xhL, xhH, zL, zH, logits_AL, logits_AH, d_logits = model(
                xL, xH, akL, akH, alpha=0.0)
            loss, mse, smooth, clf_AL, clf_AH, domain, cons = loss_labeled(
                xL, xH, mL, mH, xhL, xhH,
                logits_AL, logits_AH, yAL, yAH,
                d_logits, domain_label(len(idx_v), 1),
                a_vals_AL, a_vals_AH, akL, akH)
            tv    += loss.item()
            tcAL  += clf_AL.item(); tcAH += clf_AH.item()
            td    += domain.item(); tcons += cons.item()
            n_correct_AL += (logits_AL.argmax(1) == yAL).sum().item()
            n_correct_AH += (logits_AH.argmax(1) == yAH).sum().item()
            n_total      += len(yAL)

    nv     = max(len(idx_val) // BATCH_SIZE, 1)
    acc_AL = n_correct_AL / n_total
    acc_AH = n_correct_AH / n_total
    acc    = (acc_AL + acc_AH) / 2

    history['val'].append(tv/nv)
    history['clf_AL'].append(tcAL/nv); history['clf_AH'].append(tcAH/nv)
    history['domain'].append(td/nv);   history['cons'].append(tcons/nv)
    history['acc_AL'].append(acc_AL);  history['acc_AH'].append(acc_AH)

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
              f"acc_AL={acc_AL:.3f}  acc_AH={acc_AH:.3f}  "
              f"cons={tcons/nv:.4f}  domain={td/nv:.4f}  "
              f"α={alpha:.3f}  lr={cur_lr:.2e}  es={es_counter}/{ES_PATIENCE}",
              flush=True)

    if es_counter >= ES_PATIENCE:
        print(f"Early stopping à epoch {epoch} (best acc={best_acc:.4f})", flush=True)
        break

model.load_state_dict(best_state)
torch.save({
    'model_state':  model.state_dict(),
    'trace_max':    trace_max,
    'A_L_unique':   A_L_unique,
    'A_H_unique':   A_H_unique,
    'A_TOTAL':      A_TOTAL,
}, OUT_MODEL)
print(f"Modèle sauvegardé : {OUT_MODEL}")

# ── Collecte val set ──────────────────────────────────────────────────────────
print("Collecte val set...")
model.eval()
all_logits_AL, all_logits_AH = [], []
AL_val = tgt['A_L_lab'][idx_val]
AH_val = tgt['A_H_lab'][idx_val]

with torch.no_grad():
    for sl in iter_batches(len(idx_val), BATCH_SIZE, shuffle=False):
        idx_v = torch.from_numpy(idx_val).to(DEVICE)[sl]
        xL  = tgt_tL_lab_gpu[idx_v]; xH  = tgt_tH_lab_gpu[idx_v]
        akL = tgt_akin_L_lab_gpu[idx_v]; akH = tgt_akin_H_lab_gpu[idx_v]
        _, _, _, _, logits_AL, logits_AH, _ = model(xL, xH, akL, akH, alpha=0.0)
        all_logits_AL.append(logits_AL.cpu().numpy())
        all_logits_AH.append(logits_AH.cpu().numpy())

logits_AL_v = np.concatenate(all_logits_AL)
logits_AH_v = np.concatenate(all_logits_AH)
proba_AL    = torch.softmax(torch.from_numpy(logits_AL_v), dim=1).numpy()
proba_AH    = torch.softmax(torch.from_numpy(logits_AH_v), dim=1).numpy()
pred_AL     = A_L_unique[proba_AL.argmax(axis=1)]
pred_AH     = A_H_unique[proba_AH.argmax(axis=1)]
true_AL     = AL_val[:len(pred_AL)]
true_AH     = AH_val[:len(pred_AH)]

acc_AL = (pred_AL == true_AL).mean()
acc_AH = (pred_AH == true_AH).mean()
acc_pm1_AL = (np.abs(pred_AL - true_AL) <= 1).mean()
acc_pm1_AH = (np.abs(pred_AH - true_AH) <= 1).mean()
acc_cons   = ((pred_AL + pred_AH) <= A_TOTAL).mean()

print(f"\nAccuracy A_L stricte : {acc_AL:.3f}  |  ±1 : {acc_pm1_AL:.3f}")
print(f"Accuracy A_H stricte : {acc_AH:.3f}  |  ±1 : {acc_pm1_AH:.3f}")
print(f"Conservation A_L+A_H≤{A_TOTAL} : {acc_cons:.3f}")

# Espérance softmax
pred_AL_reg = (torch.softmax(torch.from_numpy(logits_AL_v), dim=1) *
               torch.tensor(A_L_unique, dtype=torch.float32)).sum(dim=1).numpy()
pred_AH_reg = (torch.softmax(torch.from_numpy(logits_AH_v), dim=1) *
               torch.tensor(A_H_unique, dtype=torch.float32)).sum(dim=1).numpy()

sigma_AL = (pred_AL_reg - true_AL).std()
sigma_AH = (pred_AH_reg - true_AH).std()
print(f"σ(A_L) espérance : {sigma_AL:.3f}  |  σ(A_H) espérance : {sigma_AH:.3f}")

# Coverage
conf_AL = proba_AL.max(axis=1)
conf_AH = proba_AH.max(axis=1)
conf    = np.minimum(conf_AL, conf_AH)   # confiance conjointe

thresholds = np.percentile(conf, np.linspace(0, 90, 60))
acc_cov_AL, acc_cov_AH, cov_cov = [], [], []
for thr in thresholds:
    mask = conf >= thr
    if mask.sum() < 10: continue
    acc_cov_AL.append((pred_AL[mask] == true_AL[mask]).mean())
    acc_cov_AH.append((pred_AH[mask] == true_AH[mask]).mean())
    cov_cov.append(mask.mean())

# ── Figures ───────────────────────────────────────────────────────────────────
print(f"Génération {OUT_PDF}...")
with PdfPages(OUT_PDF) as pdf:

    # Fig 1 : loss curves
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle('DANN-AE Mass Kin — A_L et A_H avec A_kin comme feature', fontsize=11)
    ep = range(1, len(history['train']) + 1)
    for ax, key, title, col in zip(axes.flat,
        ['train', 'clf_AL', 'clf_AH', 'domain', 'cons', 'acc_AL'],
        ['Loss totale', 'CE A_L', 'CE A_H', 'Domain loss', 'Conservation', 'Accuracy'],
        ['steelblue', 'seagreen', 'limegreen', 'orchid', 'crimson', 'goldenrod']):
        if key == 'train':
            ax.plot(ep, history['train'], color='steelblue', label='train')
            ax.plot(ep, history['val'],   color='coral',     label='val')
            ax.legend(fontsize=8)
        elif key == 'acc_AL':
            ax.plot(ep, history['acc_AL'], color='steelblue', label='A_L')
            ax.plot(ep, history['acc_AH'], color='coral',     label='A_H')
            ax.legend(fontsize=8)
        else:
            ax.plot(ep, history[key], color=col)
        ax.set_title(title, fontsize=9); ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Fig 2 : accuracy vs coverage
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(cov_cov, acc_cov_AL, 'b-o', markersize=3, label=f'A_L (acc={acc_AL:.3f})')
    ax.plot(cov_cov, acc_cov_AH, 'r-o', markersize=3, label=f'A_H (acc={acc_AH:.3f})')
    ax.set_xlabel('Coverage'); ax.set_ylabel('Accuracy stricte')
    ax.set_title(f'DANN-AE Mass — accuracy vs coverage\n'
                 f'A_L : acc={acc_AL:.3f} ±1={acc_pm1_AL:.3f} | '
                 f'A_H : acc={acc_AH:.3f} ±1={acc_pm1_AH:.3f}', fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Fig 3 : matrices de confusion A_L et A_H
    for true_A, pred_A, A_unique, title in [
        (true_AL, pred_AL, A_L_unique, f'A_L — acc={acc_AL:.3f}'),
        (true_AH, pred_AH, A_H_unique, f'A_H — acc={acc_AH:.3f}'),
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
        tick_step = max(1, len(a_range) // 30)
        ticks = range(0, len(a_range), tick_step)
        ax.set_xticks(list(ticks))
        ax.set_xticklabels([a_range[i] for i in ticks], fontsize=6, rotation=90)
        ax.set_yticks(list(ticks))
        ax.set_yticklabels([a_range[i] for i in ticks], fontsize=6)
        ax.set_xlabel('A prédit'); ax.set_ylabel('A vrai')
        ax.set_title(f'Matrice de confusion — {title}', fontsize=11)
        plt.colorbar(im, ax=ax)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Fig 4 : distribution des résidus
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, res, sigma, title in [
        (axes[0], pred_AL_reg - true_AL, sigma_AL, f'A_L  σ={sigma_AL:.3f}'),
        (axes[1], pred_AH_reg - true_AH, sigma_AH, f'A_H  σ={sigma_AH:.3f}'),
    ]:
        bins = np.linspace(-10, 10, 80)
        ax.hist(res, bins=bins, color='steelblue', alpha=0.8, density=True)
        ax.set_xlabel('A_pred - A_true'); ax.set_ylabel('Densité')
        ax.set_title(title, fontsize=11); ax.grid(True, alpha=0.3)
    fig.suptitle('Distribution des résidus — espérance softmax', fontsize=11)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Fig 5 : distribution A_L + A_H prédit
    a_sum_pred = pred_AL + pred_AH
    fig, ax = plt.subplots(figsize=(8, 5))
    vals, counts = np.unique(a_sum_pred, return_counts=True)
    ax.bar(vals, counts / counts.sum(), color='steelblue', alpha=0.8)
    ax.axvline(A_TOTAL, color='red', linestyle='--', label=f'A_TOTAL={A_TOTAL}')
    ax.set_xlabel('A_L + A_H prédit'); ax.set_ylabel('Fraction')
    ax.set_title(f'Distribution de A_L + A_H prédit\n'
                 f'≤{A_TOTAL} : {acc_cons:.3f}', fontsize=11)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

print(f"\nPDF généré : {OUT_PDF}")
print(f"Accuracy A_L : {acc_AL:.3f}  ±1={acc_pm1_AL:.3f}  σ={sigma_AL:.3f}")
print(f"Accuracy A_H : {acc_AH:.3f}  ±1={acc_pm1_AH:.3f}  σ={sigma_AH:.3f}")
print(f"Conservation A_L+A_H≤{A_TOTAL} : {acc_cons:.3f}")
