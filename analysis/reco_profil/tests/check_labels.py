#!/usr/bin/env python3
"""
check_labels.py : controle des labels ajoutes par prod_refine.py (version 2).
Usage : python3 check_labels.py ANCIEN_prod.h5 NOUVEAU_prod.h5 FICHIER_DRIFT.h5
  1. toutes les colonnes de l'ancien fichier sont IDENTIQUES dans le nouveau (la reconstruction n'a pas change) ;
  2. controle independant de l'attribution (recouvrement de pads) : le demi-plan de H est deduit du signe de uy
     (axe d'emission lu dans /metadata). Les deux methodes ne peuvent differer que si le vertex n'est pas dans le plan
     y = y_plane (ici /vertex/y est affiche pour verification) ;
  3. statistiques : fraction de fragment 0 = H, cosinus de controle (mediane, minimum, nombre < 0.9) ;
  4. controle par /P/pflag (1 = H seul, 2 = L seul, 3 = H et L, 0 = fantome) : les pads « H seul » doivent etre dans le
     demi-plan auquel les labels ont attribue H, les pads « L seul » dans celui de L. Fraction attendue : tres proche de
     100 % (la diffusion de derive peut deposer un peu de charge de l'autre cote du plan) ; evenements dont la MAJORITE
     des pads « H seul » (ou « L seul ») est du mauvais cote : attendu 0.
"""
import sys

import h5py
import numpy as np

if len(sys.argv) != 4:
    sys.exit(__doc__)
old, new, src = h5py.File(sys.argv[1], "r"), h5py.File(sys.argv[2], "r"), h5py.File(sys.argv[3], "r")

print("=== 1. colonnes de l'ancien fichier ===")
bad = 0
for k in old.keys():
    if k not in new:
        print("  ABSENTE du nouveau : %s" % k)
        bad += 1
        continue
    a, b = np.array(old[k]), np.array(new[k])
    if a.shape != b.shape:
        print("  FORME differente : %s %s %s" % (k, a.shape, b.shape))
        bad += 1
    elif not np.array_equal(a, b):
        print("  DIFFERENTE : %s (ecart max %.3g)" % (k, np.nanmax(np.abs(a.astype(float) - b.astype(float)))))
        bad += 1
print("  %d colonne(s) differente(s) sur %d" % (bad, len(list(old.keys()))))

print("=== 2. attribution : recouvrement de pads (nouveau fichier) contre signe de uy (calcul independant) ===")
idx = np.array(new["event_index"])
axis = src["/metadata"].attrs["emission_axis"]
axis = axis.decode() if isinstance(axis, bytes) else str(axis)
t = np.array(src["/H/theta"])[idx]
p = np.array(src["/H/phi"])[idx]
uy = np.cos(t) if axis == "Y" else np.sin(t) * np.sin(p)
h0 = uy > 0.0
vy = np.array(src["/vertex/y"])[idx]
print("  axe d'emission : %s ; /vertex/y : min %.4g, max %.4g" % (axis, vy.min(), vy.max()))
H = dict((k, np.array(src["/H/" + k])[idx]) for k in ("Z", "A", "KE"))
L = dict((k, np.array(src["/L/" + k])[idx]) for k in ("Z", "A", "KE"))
diff = np.zeros(len(idx), dtype=bool)
for name, col in (("Z", "frag_Z"), ("A", "frag_A"), ("KE", "frag_KE_true")):
    c = np.array(new[col]).astype(float)
    e0 = np.where(h0, H[name], L[name]).astype(float)
    e1 = np.where(h0, L[name], H[name]).astype(float)
    diff |= ~(np.isclose(c[:, 0], e0) & np.isclose(c[:, 1], e1))
print("  evenements dont l'attribution differe du signe de uy : %d sur %d" % (int(diff.sum()), len(idx)))
if diff.any():
    print("  premiers indices d'evenement (event_index) :", idx[diff][:10])
print("  evenements sans labels (Z = -1) : %d" % int((np.array(new["frag_Z"]) < 0).any(axis=1).sum()))

print("=== 3. statistiques ===")
print("  evenements : %d ; fragment 0 = H (selon le nouveau fichier) : %.1f %%"
      % (len(idx), 100.0 * (np.array(new["frag_Z"])[:, 0] == H["Z"]).mean()))
cs = np.array(new["label_cos"])
for k in (0, 1):
    print("  label_cos fragment %d : mediane %.4f ; minimum %.4f ; < 0.9 : %d"
          % (k, np.nanmedian(cs[:, k]), np.nanmin(cs[:, k]), int((cs[:, k] < 0.9).sum())))
print("=== 4. /P/pflag contre demi-plans attribues (1 = H seul, 2 = L seul, 3 = H et L, 0 = fantome) ===")
nwrong = 0
if "/P/pflag" not in src or "/pads" not in src or "y_plane" not in new.attrs:
    print("  controle impossible : /P/pflag, /pads ou l'attribut y_plane absent")
else:
    yp = float(new.attrs["y_plane"])
    pid_all = np.array(src["/pads/id"], dtype=int)
    yy = np.full(pid_all.max() + 1, np.nan)
    yy[pid_all] = np.array(src["/pads/y"], dtype=float)
    fz, fa, fk = np.array(new["frag_Z"]), np.array(new["frag_A"]), np.array(new["frag_KE_true"])
    h_in0 = (fz[:, 0] == H["Z"]) & (fa[:, 0] == H["A"]) & np.isclose(fk[:, 0], H["KE"])
    good_n = {1: 0, 2: 0}
    tot_n = {1: 0, 2: 0}
    wrong = {1: [], 2: []}
    n0 = n3 = 0
    for e in range(len(idx)):
        i = int(idx[e])
        pid = np.array(src["/P/pid"][i], dtype=int)
        fl = np.array(src["/P/pflag"][i], dtype=int)
        y = yy[pid]
        up, dn = y > yp, y < yp
        n0 += int((fl == 0).sum())
        n3 += int((fl == 3).sum())
        for code, in0 in ((1, bool(h_in0[e])), (2, not bool(h_in0[e]))):
            m = fl == code
            nn = int(m.sum())
            gg = int((m & (up if in0 else dn)).sum())
            good_n[code] += gg
            tot_n[code] += nn
            if nn > 0 and 2 * gg < nn:
                wrong[code].append(i)
    for code, name in ((1, "H seul"), (2, "L seul")):
        print("  pads « %s » dans le bon demi-plan : %d / %d (%.2f %%) ; evenements a majorite du mauvais cote : %d"
              % (name, good_n[code], tot_n[code], 100.0 * good_n[code] / max(tot_n[code], 1), len(wrong[code])))
        if wrong[code]:
            print("    premiers indices d'evenement :", wrong[code][:10])
        nwrong += len(wrong[code])
    print("  pads « H et L » : %d ; pads fantomes : %d (non utilises)" % (n3, n0))
print("RESULTAT : %s" % ("OK" if (bad == 0 and not diff.any() and nwrong == 0) else "A EXAMINER"))
