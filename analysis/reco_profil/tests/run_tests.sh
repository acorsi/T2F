#!/bin/bash
# Tests de non-regression : profile_basis_fast contre profile_basis (production), sur requetes SYNTHETIQUES.
# Usage (depuis le dossier reco_profil) :  bash tests/run_tests.sh /chemin/vers/profile_basis /chemin/vers/drift.cfg
# Resultat attendu : "ECART MAX GLOBAL : 0.000e+00" partout (reponses identiques bit a bit).
REF="$1"
CFG="$2"
FAST="./build/profile_basis_fast"
T="tests/tmp"
if [ -z "$REF" ] || [ ! -x "$REF" ] || [ -z "$CFG" ] || [ ! -f "$CFG" ]; then
    echo "Usage : bash tests/run_tests.sh /chemin/vers/profile_basis /chemin/vers/drift.cfg"; exit 1
fi
if [ ! -x "$FAST" ]; then echo "profile_basis_fast absent : compiler d'abord (voir README.md)"; exit 1; fi
mkdir -p "$T"
run_case () {   # nom  n_evt  graine  portee_pads  step  delta  threads
  python3 tests/make_req.py "$T/req.txt" $2 $3 $4 $5 $6 || exit 1
  "$REF"  "$CFG" "$T/req.txt" "$T/ref.bin"  n_threads=$7 > "$T/ref.out"  2>&1 || { echo "echec profile_basis"; tail -3 "$T/ref.out"; exit 1; }
  "$FAST" "$CFG" "$T/req.txt" "$T/fast.bin" n_threads=$7 > "$T/fast.out" 2>&1 || { echo "echec profile_basis_fast"; tail -3 "$T/fast.out"; exit 1; }
  echo "== $1 : $(grep calcul "$T/ref.out" | head -1)  ->  $(grep calcul "$T/fast.out" | head -1) ; $(grep cellules "$T/fast.out")"
  python3 tests/cmp_rsp.py "$T/ref.bin" "$T/fast.bin" | tail -1
}
run_case "pads proches de la trace (1.5 pitch), 1 thread" 40 7 1.5 0.05 0.2 1
run_case "pads larges (3 pitchs), 1 thread"               40 8 3.0 0.05 0.2 1
run_case "delta 0.4"                                      25 11 2.0 0.05 0.4 2
run_case "step 0.1"                                       25 12 2.0 0.1 0.2 2
run_case "step 0.03, delta 0.25"                          25 13 2.0 0.03 0.25 2
run_case "delta = step = 0.05"                            25 14 2.0 0.05 0.05 2
# format etendu (masque de fragments)
python3 tests/make_req.py "$T/req.txt"  30 21 2.0 0.05 0.2
python3 tests/make_req.py "$T/reqf.txt" 30 21 2.0 0.05 0.2 fast
"$REF"  "$CFG" "$T/req.txt"  "$T/ref.bin" n_threads=2 > /dev/null 2>&1
"$FAST" "$CFG" "$T/reqf.txt" "$T/fast.bin" n_threads=2 > /dev/null 2>&1
echo "== format etendu (masque 1/2/3)"
python3 tests/cmp_mask.py "$T/ref.bin" "$T/fast.bin"
rm -rf "$T"
