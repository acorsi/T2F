#!/bin/bash
# Tests qui ont besoin de HDF5 et de Python (h5py, numpy, matplotlib) : contenu du h5, reconstruction de
# trace, make_clean_cfg --check, viewer sans affichage. Ils tournent sur de VRAIES sorties de ./process
# (court : 300 evenements par run), donc ils ont besoin du programme compile et d'un fichier d'entree.
#
# Usage (a lancer depuis le dossier drift/tests) :
#   bash run_tests_h5.sh CHEMIN/process CHEMIN/drift.cfg CHEMIN/input.h5 [CHEMIN/view_pads.py]
#
#   process       le programme compile (par exemple ../build/process)
#   drift.cfg     votre fichier de config (les valeurs ci-dessous le surchargent : il n'est pas modifie)
#   input.h5      un fichier d'entree (verites des fragments) d'au moins 300 evenements
#   view_pads.py  facultatif : si absent, les tests du viewer sont sautes
#
# Les runs sont ecrits dans le dossier out_h5/ de ce dossier. Deux configurations :
#   A : pads carres de 5 mm, sans anneaux (comme votre config)
#   B : pads de 3 mm avec anneaux (pad_size = 0.3, r_quinconce = 8.1, r_transition = 8.2, r_active = 10, 6 anneaux)
# chacune avec bruit et fantomes, puis le run sans bruit correspondant (make_clean_cfg.py).
# Puis, en modele de bruit "capacitance" (config A) :
#   C : capacite unique par defaut (c_cable_pf + c_routing_pf), avec bruit puis sans bruit
#   D : fichier de capacites a deux valeurs (60 et 150 pF, selon la parite de l'identifiant), avec bruit puis sans bruit,
#       et reproductibilite a 1 et 8 threads ; un fichier de capacites invalide doit faire refuser le run
set -e
if [ $# -lt 3 ]; then
    echo "usage : bash run_tests_h5.sh CHEMIN/process CHEMIN/drift.cfg CHEMIN/input.h5 [CHEMIN/view_pads.py]"
    exit 2
fi
PROC=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
CFG=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
INPUT=$(cd "$(dirname "$3")" && pwd)/$(basename "$3")
VIEWER=""
if [ $# -ge 4 ]; then VIEWER=$(cd "$(dirname "$4")" && pwd)/$(basename "$4"); fi
HERE=$(cd "$(dirname "$0")" && pwd)
ANA="$HERE/../analysis"
OUT="$HERE/out_h5"
mkdir -p "$OUT"
cd "$OUT"

# fichiers produits par un lancement precedent : make_clean_cfg.py refuse d'ecraser, donc on les retire
# (uniquement les fichiers de ce script, par leur nom : rien d'autre dans out_h5 n'est touche)
for f in A.h5 A.log A_clean.cfg A_clean.h5 A_clean.log B.h5 B.log B_clean.cfg B_clean.h5 B_clean.log \
         B_t1.h5 B_t1.log B_t8.h5 B_t8.log C.h5 C.log C_clean.cfg C_clean.h5 C_clean.log \
         D.h5 D.log D_clean.cfg D_clean.h5 D_clean.log D_t1.h5 D_t1.log D_t8.h5 D_t8.log cap_2cl.txt cap_bad.txt cap_bad.log; do
    rm -f "$OUT/$f"
done

COMMON_NT="input_file=$INPUT n_events=300 noise_enable=1 ghost_enable=1"
COMMON="$COMMON_NT n_threads=2"
GEO_B="pad_size=0.3 inter_pad=0.015 r_quinconce=8.1 r_transition=8.2 r_active=10.0 n_rings=6 reach_pitch=3.5"

echo "=== run A (5 mm, avec bruit) ==="
"$PROC" "$CFG" $COMMON output_file="$OUT/A.h5" > A.log 2>&1
python3 "$ANA/make_clean_cfg.py" "$OUT/A.h5" -o "$OUT/A_clean.cfg" --output-file "$OUT/A_clean.h5" > /dev/null
echo "=== run A (sans bruit) ==="
"$PROC" "$OUT/A_clean.cfg" > A_clean.log 2>&1

echo "=== run B (3 mm + anneaux, avec bruit) ==="
"$PROC" "$CFG" $COMMON $GEO_B output_file="$OUT/B.h5" > B.log 2>&1
python3 "$ANA/make_clean_cfg.py" "$OUT/B.h5" -o "$OUT/B_clean.cfg" --output-file "$OUT/B_clean.h5" > /dev/null
echo "=== run B (sans bruit) ==="
"$PROC" "$OUT/B_clean.cfg" > B_clean.log 2>&1

echo "=== run B avec 1 thread, puis avec 8 threads (reproductibilite : doivent etre identiques a B, fait avec 2 threads) ==="
"$PROC" "$CFG" $COMMON_NT $GEO_B n_threads=1 output_file="$OUT/B_t1.h5" > B_t1.log 2>&1
"$PROC" "$CFG" $COMMON_NT $GEO_B n_threads=8 output_file="$OUT/B_t8.h5" > B_t8.log 2>&1

echo "=== run C (5 mm, modele capacitance, capacite unique par defaut, avec bruit puis sans bruit) ==="
CAPM="noise_model=capacitance"
"$PROC" "$CFG" $COMMON $CAPM output_file="$OUT/C.h5" > C.log 2>&1
python3 "$ANA/make_clean_cfg.py" "$OUT/C.h5" -o "$OUT/C_clean.cfg" --output-file "$OUT/C_clean.h5" > /dev/null
"$PROC" "$OUT/C_clean.cfg" > C_clean.log 2>&1

echo "=== run D (modele capacitance, fichier de capacites 60 / 150 pF, avec bruit puis sans bruit ; 1 et 8 threads) ==="
python3 - "$OUT/C.h5" "$OUT/cap_2cl.txt" <<'PYEOF'
import sys, h5py
ids = h5py.File(sys.argv[1], "r")["/pads/id"][:]
with open(sys.argv[2], "w") as g:
    g.write("# pad_id C_pF : 60 pF pour les identifiants pairs, 150 pF pour les impairs (test)\n")
    for i in ids:
        g.write("%d %g\n" % (i, 60.0 if i % 2 == 0 else 150.0))
PYEOF
CAPD="$CAPM pad_capacitance_file=$OUT/cap_2cl.txt"
"$PROC" "$CFG" $COMMON $CAPD output_file="$OUT/D.h5" > D.log 2>&1
python3 "$ANA/make_clean_cfg.py" "$OUT/D.h5" -o "$OUT/D_clean.cfg" --output-file "$OUT/D_clean.h5" > /dev/null
"$PROC" "$OUT/D_clean.cfg" > D_clean.log 2>&1
"$PROC" "$CFG" $COMMON_NT $CAPD n_threads=1 output_file="$OUT/D_t1.h5" > D_t1.log 2>&1
"$PROC" "$CFG" $COMMON_NT $CAPD n_threads=8 output_file="$OUT/D_t8.h5" > D_t8.log 2>&1

echo "=== fichier de capacites invalide : le programme doit refuser (code retour != 0) ==="
printf "99999 50\n" > "$OUT/cap_bad.txt"
if "$PROC" "$CFG" $COMMON $CAPM pad_capacitance_file="$OUT/cap_bad.txt" n_events=2 output_file="$OUT/cap_bad.h5" > cap_bad.log 2>&1; then
    echo "ECHEC : un fichier de capacites invalide a ete accepte"; exit 1
else
    echo "  ok   fichier invalide refuse : $(grep -m1 'Arret' cap_bad.log)"
fi
rm -f "$OUT/cap_bad.h5"

echo "=== tests ==="
if [ -n "$VIEWER" ]; then
    python3 "$HERE/test_h5.py" --pair "$OUT/A.h5" "$OUT/A_clean.h5" --pair "$OUT/B.h5" "$OUT/B_clean.h5" --pair "$OUT/C.h5" "$OUT/C_clean.h5" \
            --same "$OUT/B.h5" "$OUT/B_t1.h5" --same "$OUT/B.h5" "$OUT/B_t8.h5" \
            --same "$OUT/D.h5" "$OUT/D_t1.h5" --same "$OUT/D.h5" "$OUT/D_t8.h5" \
            --cap "$OUT/C.h5" "$OUT/C_clean.h5" --cap "$OUT/D.h5" "$OUT/D_clean.h5" \
            --analysis "$ANA" --viewer "$VIEWER"
else
    python3 "$HERE/test_h5.py" --pair "$OUT/A.h5" "$OUT/A_clean.h5" --pair "$OUT/B.h5" "$OUT/B_clean.h5" --pair "$OUT/C.h5" "$OUT/C_clean.h5" \
            --same "$OUT/B.h5" "$OUT/B_t1.h5" --same "$OUT/B.h5" "$OUT/B_t8.h5" \
            --same "$OUT/D.h5" "$OUT/D_t1.h5" --same "$OUT/D.h5" "$OUT/D_t8.h5" \
            --cap "$OUT/C.h5" "$OUT/C_clean.h5" --cap "$OUT/D.h5" "$OUT/D_clean.h5" \
            --analysis "$ANA"
fi
