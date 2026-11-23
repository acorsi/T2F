#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cfg_from_h5.py : reconstruit le fichier .cfg du simulateur (drift) a partir du fichier h5 qu'il a produit.

Le h5 de sortie de ./process contient le texte du .cfg utilise (fichier + surcharges de la ligne de
commande) dans /metadata, attribut config_text. Ce script :
  1. relit ce texte ;
  2. si une cle est definie plusieurs fois, ne garde que la derniere definition (celle qui a gagne :
     ConfigFile refuse les cles en double), les autres sont mises en commentaire ;
  3. ecrit le .cfg, SANS rien changer d'autre : c'est exactement la configuration de la simulation
     qui a produit les donnees, donc le serveur profile_basis_fast calcule avec la meme physique.
  4. controle (avertissement seulement) que quelques attributs de /metadata valent la meme chose que
     la cle du meme nom dans le cfg.

Usage :
    python3 cfg_from_h5.py fichier.h5 sortie.cfg
(un sortie.cfg existant est remplace ; le resultat est le meme a chaque appel)

Necessite h5py et numpy.
"""

import os
import sys

import numpy as np
import h5py

# attributs de /metadata censes valoir la meme chose que la cle de meme nom du cfg (controle indicatif)
CHECK_KEYS = ["delta_t", "dkg", "dga", "ve_kg", "ve_ga", "W", "pad_size", "inter_pad", "noise_enable"]


def as_text(v):
    if isinstance(v, bytes):
        return v.decode("utf-8")
    if isinstance(v, np.ndarray):
        return as_text(v.item())
    return str(v)


def parse_key(line):
    """Meme regle que ConfigFile : commentaire apres '#', cle = valeur ; None si la ligne ne definit rien."""
    s = line.split("#", 1)[0].strip()
    if not s or "=" not in s:
        return None
    key, val = s.split("=", 1)
    return key.strip(), val.strip()


def build_cfg(config_text, h5name):
    lines = config_text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()

    # derniere definition de chaque cle = celle qui gagne
    last = {}
    for i, ln in enumerate(lines):
        kv = parse_key(ln)
        if kv:
            last[kv[0]] = i

    out = []
    n_dup = 0
    for i, ln in enumerate(lines):
        kv = parse_key(ln)
        if kv and last[kv[0]] != i:
            out.append("# (remplace plus bas) " + ln)
            n_dup += 1
        else:
            out.append(ln)

    header = [
        "# ---------------------------------------------------------------------------",
        "# Genere par cfg_from_h5.py a partir de : " + os.path.basename(h5name),
        "# Configuration exacte de la simulation qui a produit ce fichier (config_text de /metadata).",
        "# Ne pas modifier a la main.",
        "# ---------------------------------------------------------------------------",
    ]
    text = "\n".join(header + out) + "\n"

    # controle : plus aucune cle en double
    values = {}
    for ln in text.split("\n"):
        kv = parse_key(ln)
        if kv:
            if kv[0] in values:
                raise SystemExit("cle en double dans le cfg genere : " + kv[0])
            values[kv[0]] = kv[1]
    return text, values, n_dup


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage : python3 cfg_from_h5.py fichier.h5 sortie.cfg")
    h5name, cfg_path = sys.argv[1], sys.argv[2]

    with h5py.File(h5name, "r") as f:
        if "/metadata" not in f:
            raise SystemExit("pas de /metadata dans %s : fichier non produit par ./process ?" % h5name)
        attrs = f["/metadata"].attrs
        if "config_text" not in attrs:
            raise SystemExit("pas de config_text dans /metadata de %s : fichier non produit par ./process ?" % h5name)
        text = as_text(attrs["config_text"])
        meta = {}
        for k in CHECK_KEYS:
            if k in attrs:
                meta[k] = attrs[k]

    cfg_text, values, n_dup = build_cfg(text, h5name)

    n_warn = 0
    for k in CHECK_KEYS:
        if k in values and k in meta:
            try:
                a = float(np.asarray(meta[k]).reshape(-1)[0])
                c = float(values[k])
            except (ValueError, TypeError, IndexError):
                continue
            if abs(a - c) > 1e-6 * max(abs(a), abs(c), 1e-30):
                sys.stderr.write("ATTENTION : %s = %g dans /metadata, %g dans le cfg reconstruit\n" % (k, a, c))
                n_warn += 1

    with open(cfg_path, "w") as g:
        g.write(cfg_text)
    os.chmod(cfg_path, 0o644)

    print("cfg ecrit : %s (%d cles, %d definition(s) multiple(s) resolue(s), %d avertissement(s))"
          % (cfg_path, len(values), n_dup, n_warn))


if __name__ == "__main__":
    main()
