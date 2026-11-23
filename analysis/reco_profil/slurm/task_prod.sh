#!/bin/bash
# Une tache du job array : traite une tranche de SLICE evenements avec prod_refine.py (1 CPU).
# Variables (donnees par submit_prod.sh) : FB (dossier reco_profil), H5, OUT, SLICE, PYBIN (dossier du python), PYLIB
# Le cfg du simulateur est reconstruit depuis le h5 par submit_prod.sh : $OUT/drift_from_h5.cfg
set -u
export PATH=$PYBIN:$PATH
export PYTHONPATH=$PYLIB
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export WHITEN_FAST=1 JAC_FAST=1

K=${SLURM_ARRAY_TASK_ID}
START=$((K * SLICE))
NAME=$(printf "s_%05d" $K)
RES=$OUT/slices/$NAME
CFG=$OUT/drift_from_h5.cfg
if [ -s ${RES}_prod.h5 ]; then
    echo "[task $K] tranche deja calculee : ${RES}_prod.h5 (sautee)"
    exit 0
fi
if [ ! -s $CFG ]; then
    echo "[task $K] cfg absent : $CFG (lancer submit_prod.sh)"
    exit 1
fi

# fichiers d'echange sur le disque local du noeud, supprimes en fin de tache meme en cas d'erreur
TMPD=/tmp/${USER}_${SLURM_JOB_ID:-0}_${K}
mkdir -p $TMPD
trap 'rm -rf $TMPD' EXIT

echo "[task $K] noeud $(hostname), evenements $START a $((START + SLICE - 1)), debut $(date)"
cd $FB/reco
python3 prod_refine.py $H5 --start $START --n $SLICE --out $RES --tmpdir $TMPD \
    --exe ../build/profile_basis_fast --cfg $CFG --exe_args n_threads=1 --lam 1e-5 --chunk 10 ${PRARGS:-}
RC=$?
echo "[task $K] fin $(date), code $RC"
exit $RC
