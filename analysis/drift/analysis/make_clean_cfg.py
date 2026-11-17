#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# make_clean_cfg.py : genere le .cfg du run SANS bruit correspondant a un fichier h5 AVEC bruit.
#
# Le fichier h5 de sortie de ./process contient le texte du .cfg utilise (fichier + surcharges
# de la ligne de commande) dans /metadata, attribut config_text. Ce script :
#   1. relit ce texte et applique les surcharges (ConfigFile refuse les cles en double) ;
#   2. met noise_enable = 0 et change output_file ;
#   3. ecrit le nouveau .cfg ; tout le reste est identique (input_file, base_seed, n_events,
#      seuil_qt, noise_seed, reponse du preampli, ...), donc la verite physique est la meme.
#
# Avec --check clean.h5, compare ensuite les deux fichiers h5 (apres avoir lance le run sans bruit) :
# memes evenements, memes verites, memes phases, memes charges vraies et memes fenetres pour les pads
# communs, et rms de (bruite - propre), qui doit valoir enc_total.
#
# Usage :
#   python3 make_clean_cfg.py bruite.h5 [-o propre.cfg] [--output-file propre.h5]
#   python3 make_clean_cfg.py bruite.h5 --check propre.h5 [--max-events N]
#
# Necessite h5py et numpy.
# ---------------------------------------------------------------------------

import argparse
import os
import sys

import numpy as np
import h5py


def as_text(v):
    if isinstance(v, bytes):
        return v.decode("utf-8")
    if isinstance(v, np.ndarray):
        v = v.item()
        return as_text(v)
    return str(v)


def parse_key(line):
    """Meme regle que ConfigFile : commentaire apres '#', cle = valeur ; None si la ligne ne definit rien."""
    s = line.split("#", 1)[0].strip()
    if not s or "=" not in s:
        return None
    key, val = s.split("=", 1)
    return key.strip(), val.strip()


def build_clean_cfg(config_text, noisy_h5, new_output=None):
    lines = config_text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()

    # derniere definition de chaque cle = celle qui gagne (surcharge de la ligne de commande)
    last = {}
    for i, ln in enumerate(lines):
        kv = parse_key(ln)
        if kv:
            last[kv[0]] = i

    out = []
    for i, ln in enumerate(lines):
        kv = parse_key(ln)
        if kv and last[kv[0]] != i:
            out.append("# (remplace plus bas) " + ln)
        else:
            out.append(ln)

    def set_key(key, value):
        if key in last:
            out[last[key]] = "%s = %s   # modifie par make_clean_cfg.py" % (key, value)
        else:
            out.append("%s = %s   # ajoute par make_clean_cfg.py" % (key, value))

    old_output = None
    if "output_file" in last:
        old_output = parse_key(lines[last["output_file"]])[1]
    if new_output is None:
        if old_output is None:
            raise SystemExit("output_file absent de config_text : donner --output-file")
        base = old_output[:-3] if old_output.endswith(".h5") else old_output
        new_output = base + "_clean.h5"
    if old_output is not None and os.path.abspath(new_output) == os.path.abspath(old_output):
        raise SystemExit("le fichier de sortie du run sans bruit serait celui du run avec bruit : " + new_output)

    set_key("noise_enable", "0")
    set_key("output_file", new_output)

    header = [
        "# ---------------------------------------------------------------------------",
        "# Genere par make_clean_cfg.py a partir de : " + noisy_h5,
        "# Run SANS bruit : meme chaine (reponse, CAN), sans bruit ni seuil de lecture.",
        "# A lancer depuis le dossier d'ou le run avec bruit a ete lance (chemins relatifs).",
        "# ---------------------------------------------------------------------------",
    ]
    text = "\n".join(header + out) + "\n"

    # controle : plus aucune cle en double (sinon ./process s'arreterait)
    seen = {}
    for ln in text.split("\n"):
        kv = parse_key(ln)
        if kv:
            if kv[0] in seen:
                raise SystemExit("cle en double dans le cfg genere : " + kv[0])
            seen[kv[0]] = kv[1]
    return text, seen, new_output


def read_vlen(ds, i0, i1):
    return ds[i0:i1]


def check_runs(noisy_path, clean_path, max_events):
    ok = True
    with h5py.File(noisy_path, "r") as a, h5py.File(clean_path, "r") as b:
        na, nb = a["/event_id"].shape[0], b["/event_id"].shape[0]
        n = min(na, nb)
        if max_events > 0:
            n = min(n, max_events)
        print("evenements : bruite %d, propre %d, compares %d" % (na, nb, n))

        ea, eb = a["/event_id"][:n], b["/event_id"][:n]
        if not np.array_equal(ea, eb):
            print("ECHEC : /event_id differents")
            ok = False
        else:
            print("ok   : /event_id identiques")

        # verites : KE, theta, phi par fragment ; vertex (groupe /vertex, ou /H/vx.. dans les fichiers anciens)
        paths = ["/%s/%s" % (frag, name) for name in ["KE", "theta", "phi"] for frag in ["H", "L"]]
        if "/vertex" in a and "/vertex" in b:
            paths += ["/vertex/x", "/vertex/y", "/vertex/z"]
        elif "/vertex" not in a and "/vertex" not in b:
            paths += ["/%s/%s" % (frag, name) for name in ["vx", "vy", "vz"] for frag in ["H", "L"]]
        else:
            print("ECHEC : un seul des deux fichiers a un groupe /vertex (formats differents)")
            ok = False
        for path in paths:
            if not np.array_equal(a[path][:n], b[path][:n]):
                print("ECHEC : %s differe (verite physique non identique)" % path)
                ok = False
        print("ok   : verites (KE, theta, phi de H et L, vertex) comparees")

        if not np.array_equal(a["/P/phase"][:n], b["/P/phase"][:n]):
            print("ECHEC : /P/phase differentes")
            ok = False
        else:
            print("ok   : phases identiques")

        enc_total = float(a["/metadata"].attrs["enc_total"])
        n_common = n_only_clean = n_only_noisy = n_ghost = 0
        s2 = 0.0
        n_samp = 0
        bad_q = bad_flag = bad_win = 0
        step = 2000
        for i0 in range(0, n, step):
            i1 = min(n, i0 + step)
            A = {k: a["/P/" + k][i0:i1] for k in ["pid", "pq", "pflag", "pk0", "pnt", "pqn"]}
            B = {k: b["/P/" + k][i0:i1] for k in ["pid", "pq", "pflag", "pk0", "pnt", "pqn"]}
            for ev in range(i1 - i0):
                ida, idb = A["pid"][ev], B["pid"][ev]
                offa = np.concatenate(([0], np.cumsum(A["pnt"][ev])))
                offb = np.concatenate(([0], np.cumsum(B["pnt"][ev])))
                posb = dict((int(p), j) for j, p in enumerate(idb))
                seen_b = set()
                for ja, p in enumerate(ida):
                    p = int(p)
                    if A["pflag"][ev][ja] == 0:
                        n_ghost += 1
                    jb = posb.get(p)
                    if jb is None:
                        n_only_noisy += 1
                        continue
                    seen_b.add(jb)
                    n_common += 1
                    if A["pq"][ev][ja] != B["pq"][ev][jb]:
                        bad_q += 1
                    if A["pflag"][ev][ja] != B["pflag"][ev][jb]:
                        bad_flag += 1
                    if A["pk0"][ev][ja] != B["pk0"][ev][jb] or A["pnt"][ev][ja] != B["pnt"][ev][jb]:
                        bad_win += 1
                        continue
                    sa = A["pqn"][ev][offa[ja]:offa[ja + 1]].astype(np.float64)
                    sb = B["pqn"][ev][offb[jb]:offb[jb + 1]].astype(np.float64)
                    d = sa - sb
                    s2 += float(np.sum(d * d))
                    n_samp += d.size
                n_only_clean += len(idb) - len(seen_b)
        print("pads communs %d, seulement sans bruit (non lus avec bruit) %d, seulement avec bruit %d (dont fantomes %d)"
              % (n_common, n_only_clean, n_only_noisy, n_ghost))
        for lab, nbad in [("pq (charge vraie)", bad_q), ("pflag", bad_flag), ("fenetre (pk0, pnt)", bad_win)]:
            if nbad:
                print("ECHEC : %s differe pour %d pads communs" % (lab, nbad))
                ok = False
            else:
                print("ok   : %s identique pour tous les pads communs" % lab)
        if n_samp > 0:
            rms = np.sqrt(s2 / n_samp)
            print("rms de (bruite - propre) sur %d echantillons : %.1f e- ; enc_total = %.1f e- (rapport %.3f)"
                  % (n_samp, rms, enc_total, rms / enc_total if enc_total > 0 else float("nan")))
    return ok


def main():
    ap = argparse.ArgumentParser(description="Genere le .cfg du run sans bruit a partir du h5 avec bruit.")
    ap.add_argument("noisy_h5")
    ap.add_argument("-o", "--cfg", default=None, help="fichier .cfg a ecrire (defaut : <nom du h5>_clean.cfg)")
    ap.add_argument("--output-file", default=None, help="output_file du run sans bruit (defaut : <output_file>_clean.h5)")
    ap.add_argument("--check", default=None, metavar="CLEAN_H5", help="compare les deux h5 (apres le run sans bruit)")
    ap.add_argument("--max-events", type=int, default=0, help="limite le nombre d'evenements compares")
    args = ap.parse_args()

    if args.check:
        sys.exit(0 if check_runs(args.noisy_h5, args.check, args.max_events) else 1)

    with h5py.File(args.noisy_h5, "r") as f:
        attrs = f["/metadata"].attrs
        if "config_text" not in attrs:
            raise SystemExit("pas de config_text dans /metadata : fichier non produit par ./process ?")
        text = as_text(attrs["config_text"])
        if "noise_enable" in attrs and int(attrs["noise_enable"]) != 1:
            print("ATTENTION : ce fichier a ete produit avec noise_enable = %d (pas un run avec bruit)" % int(attrs["noise_enable"]))
        input_file = as_text(attrs["input_file"]) if "input_file" in attrs else "?"

    cfg_text, keys, new_out = build_clean_cfg(text, args.noisy_h5, args.output_file)
    cfg_path = args.cfg
    if cfg_path is None:
        cfg_path = os.path.splitext(os.path.basename(args.noisy_h5))[0] + "_clean.cfg"
    if os.path.exists(cfg_path):
        raise SystemExit("%s existe deja : choisir un autre nom avec -o" % cfg_path)
    with open(cfg_path, "w") as g:
        g.write(cfg_text)
    os.chmod(cfg_path, 0o644)

    print("ecrit : %s" % cfg_path)
    print("  input_file  = %s" % input_file)
    print("  output_file = %s" % new_out)
    print("  noise_enable = 0, tout le reste identique")
    print("A lancer depuis le dossier d'ou le run avec bruit a ete lance :  ./process %s" % cfg_path)
    print("Puis :  python3 %s %s --check %s" % (os.path.basename(sys.argv[0]), args.noisy_h5, new_out))


if __name__ == "__main__":
    main()
