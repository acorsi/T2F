#!/bin/bash
# Lancer depuis le dossier tests/ de VOTRE depot (le dossier drift/tests, pas l'archive extraite) :   bash run_tests.sh
# Il utilise ../src et ../inc : ces dossiers doivent contenir tous vos fichiers sources.
# Compile et lance les tests des niveaux 0 et 1 (sans HDF5) ; arrete au premier probleme de compilation.
# Les ecarts sont "simu - exact" ; les figures sont ecrites dans out/.
set -e
mkdir -p out
INC=../inc
SRC=../src
LIB="$SRC/psi_table.cpp $SRC/psi_builder.cpp $SRC/weighting_potential.cpp"
CXX="g++ -O2 -std=c++17 -Wall -I$INC"

# OpenMP (necessaire seulement pour le niveau 2 : pad_geometry.cpp inclut prof.hpp, donc omp.h).
# Meme methode que votre CMakeLists.txt sur macOS : clang + libomp de Homebrew.
if [ "$(uname)" = "Darwin" ]; then
    LIBOMP=$(brew --prefix libomp)
    OMPC="-Xpreprocessor -fopenmp -I$LIBOMP/include"
    OMPL="-L$LIBOMP/lib -lomp"
else
    OMPC="-fopenmp"
    OMPL=""
fi

$CXX test_level0.cpp $SRC/weighting_potential.cpp -o test_level0
$CXX test_level1.cpp $LIB -o test_level1
$CXX test_level1_plateau.cpp $LIB -o test_level1_plateau
$CXX test_selfcheck.cpp $LIB -o test_selfcheck
$CXX test_track.cpp $LIB -o test_track
$CXX test_track_mc.cpp $SRC/weighting_potential.cpp -o test_track_mc
$CXX -I$SRC test_level3.cpp $SRC/processor.cpp -o test_level3
$CXX $OMPC test_level2.cpp $SRC/pad_geometry.cpp $LIB $OMPL -o test_level2

echo "=== niveau 0 : potentiel de ponderation ==="
./test_level0
python3 test_level0.py || true

echo "=== niveau 1 : table d'induction, cas A (drift.cfg), B (P10, pad 3 mm), C (P10, gap 5 mm), table_R = 4 ==="
for c in A B C; do ./test_level1 $c 4 > out/l1_$c.log; done
python3 test_level1.py A:4 B:4 C:4 || true

echo "=== charge finale par pad, 20000 points (formule en erf), d de 0.02 a 20 cm ==="
for c in A B C; do ./test_level1_plateau $c 4; done

echo "=== chaine de production : choix de R, table, auto-controle (cas A, B, C ; R automatique) ==="
for c in A B C; do
  echo "--- cas $c"
  ./test_selfcheck $c | grep -E "resolve_table_R|noeuds u,v|source phi|table estimee|construction|controle de la table|passes|Arret|ERREUR"
done

echo "=== trace courte : induction sur P0, P1, P2 (cas A ; cas 1 horizontal, cas 2 a 45 degres ; pas 0.1 mm) ==="
./test_track 5 | grep -E "quadrature|ecrit|Arret"
python3 test_track.py || true
echo "--- controle Monte-Carlo de la partie collectee de la reference"
./test_track_mc 2000000 | tail -3

echo "=== niveau 2 : PadPlane::project_pair (cas A) : conservation, symetries, linearite, selection par seuil, reference exacte ==="
./test_level2 5 | grep -v "^PadPlane"
python3 test_level2.py || true

echo "=== niveau 3 : tirage du nombre d'electrons (Processor) ; la partie donnees reelles demande input_1000.h5 dans ce dossier ==="
if [ -f input_1000.h5 ]; then python3 test_level3.py export input_1000.h5; fi
./test_level3
