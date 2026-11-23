#!/bin/bash
# Usage : submit_prod.sh FICHIER.h5 DOSSIER_SORTIE N_EVENEMENTS [SLICE=2000] [MAXPAR=150]
# Envoie un job array : une tache par tranche de SLICE evenements, MAXPAR taches simultanees au plus.
# Relancer la meme commande ne recalcule que les tranches manquantes.
# Le cfg du simulateur est reconstruit depuis le h5 (reco/cfg_from_h5.py) dans DOSSIER_SORTIE/drift_from_h5.cfg.
if [ $# -lt 3 ]; then
    echo "Usage : $0 FICHIER.h5 DOSSIER_SORTIE N_EVENEMENTS [SLICE=2000] [MAXPAR=150]"; exit 1
fi
H5=$(readlink -f $1); OUT=$(readlink -f $2 2>/dev/null || echo $2); N=$3; SLICE=${4:-2000}; MAXPAR=${5:-150}
FB=$(cd "$(dirname "$0")/.." && pwd)
PYBIN=${PYBIN:-/opt/ohpc/pub/apps/anaconda/3.11/2023.09/bin}
PYLIB=${PYLIB:-/feynman/scratch/dphn/learn/lg211663/pylib}
[ -f "$H5" ] || { echo "fichier introuvable : $H5"; exit 1; }
mkdir -p $OUT/slices $OUT/logs
# cfg du simulateur : celui qui a produit CE h5
PATH=$PYBIN:$PATH PYTHONPATH=$PYLIB python3 $FB/reco/cfg_from_h5.py $H5 $OUT/drift_from_h5.cfg || { echo "echec : cfg reconstruit depuis le h5"; exit 1; }
export PRARGS=${PRARGS:-}
NT=$(( (N + SLICE - 1) / SLICE ))
# tranches deja faites : on ne soumet que les manquantes
MISSING=""
for ((k = 0; k < NT; k++)); do
    f=$(printf "$OUT/slices/s_%05d_prod.h5" $k)
    [ -s $f ] || MISSING="$MISSING,$k"
done
MISSING=${MISSING#,}
if [ -z "$MISSING" ]; then echo "toutes les tranches sont deja calculees ($NT)"; exit 0; fi
NM=$(echo $MISSING | tr ',' '\n' | wc -l)
echo "$NM tranche(s) a calculer sur $NT (SLICE=$SLICE, $MAXPAR simultanees)"
sbatch -p htc -N 1 -n 1 -c 1 --mem=4G -t 01:30:00 -J t2f_prod \
    --array=${MISSING}%${MAXPAR} -o $OUT/logs/task_%a_%A.log \
    --export=ALL,FB=$FB,H5=$H5,OUT=$OUT,SLICE=$SLICE,PYBIN=$PYBIN,PYLIB=$PYLIB \
    $FB/slurm/task_prod.sh
