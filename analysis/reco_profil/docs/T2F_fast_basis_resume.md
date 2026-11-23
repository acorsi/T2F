# T2F – chaîne `fast_basis` : résumé fonctionnel et exploitation cluster

(Document de reprise pour une nouvelle discussion. Utilisateur : Laurent, physicien nucléaire expérimentateur, TPC fragments de fission ; MacBook Pro M4 (zsh) + cluster Slurm « feynman ».)

## 0. Règles de collaboration (à rappeler en début de nouvelle discussion)

- Réponses en français.
- Proposer avant de coder, attendre la validation de Laurent avant toute modification de code.
- Commandes toujours complètes, copiables d'un bloc.
- Ne rien supposer (chemins, noms de clés, contenu du code) : demander fichiers/infos manquants.
- C++ simple (style ≤ C++11).
- Archives tar : fichiers en rw-r--r-- (jamais lecture seule).
- Signaler défauts et alternatives, pas de flatterie. Ne jamais modifier ses fichiers d'origine.
- **RÈGLE ABSOLUE** : ne jamais utiliser l'identité lourd/léger (H/L) des fragments dans la reconstruction. Seule convention admise : demi-plan, fragment 0 = y > y_plane, fragment 1 = y < y_plane.

## 1. Objectif

Reconstruire pour chaque évènement de fission le profil dE/dx des deux fragments (vertex, directions, profil aux nœuds + incertitudes, drapeau « non contenu »), rapidement, pour traiter **1 M d'évènements** (un seul fichier HDF5, simulation sans faisceau de neutrons ; les 100 000 premiers évènements = le fichier actuel si les seeds n'ont pas changé).

## 2. Chaîne de traitement

Scripts Python (dossier `py/`) + serveur C++ (`src/profile_basis_fast.cpp`, binaire `build/profile_basis_fast --serve`).

1. **Estimation de départ** (`reco_split.estimate`, utilise `reco_dir.strong_pads`) :
   - pads « forts » : amplitude ≥ max(k_thr·seuil, f_max·amp_max), qint > 0,5·amp, ≥ n_nb voisins à moins de 1,5 pitch (défauts k_thr 10, f_max 0,1, n_nb 2, min_half 3) ;
   - ACP des centres de pads pondérés par l'amplitude crête, par demi-plan ;
   - vertex commun x_v par recherche de section dorée (somme des plus petites valeurs propres) ;
   - pentes en z par régression temporelle pondérée ;
   - pitch des pads 0,515 cm, aucun pad centré sur y = 0.
2. **`refine_geom.py`** : modèle linéaire, blanchiment, profil NNLS régularisé, puis Levenberg–Marquardt sur 6 paramètres [x, z, a0, m0, a1, m1] (jacobien par différences finies ; lam_geom 1e-4 ; `--max_iter 8` par défaut ; `--tol 1e-4` en décroissance relative de F ; le compteur d'itérations compte aussi les pas rejetés ; mu×5 au rejet ; arrêt si mu > 1e6). Extension de fin de profil ; lissage réduit en fin de trace (lam_end_fac 0,3 sur 1,2 cm).
3. **`fit_profile.py`** : ajustement du profil ; le serveur C++ fait les calculs lourds.
4. **Mode production** : `prod_refine.py` (pas de visualisation ni de vérité) + `prod_io.py` (format HDF5). Sortie `<out>_prod.h5` écrite d'abord en `.h5.part` puis renommée (un `.h5` existant = fichier complet).

Colonnes du HDF5 de production : `event_index, event_id, vertex, u0, u1, params(6), chi2_start, chi2_geom, chi2_prof, n_iter, L_base, dL, not_contained, tail_ratio, tail_sigma, n_nodes, prof_off, prof_val, prof_sig`.
Options utiles de `prod_refine.py` : `--start`, `--n`, `--out`, `--tmpdir`, `--exe`, `--cfg`, `--exe_args n_threads=1`, `--lam 1e-5`, `--chunk 10`, ainsi que les options de l'estimateur (`--k_thr`, `--f_max`, `--n_nb`) et `--max_iter`.

## 3. Cluster feynman : environnement

- Slurm 20.11.8, Rocky 8.5, partition `htc`, nœuds AMD EPYC 7513 (un « CPU » Slurm = un hyperthread), `/tmp` local NVMe 1,5 To, `/dev/shm` 126 Go. Cluster partagé, ≈ 84 % occupé.
- Connexion : `ssh lg211663@feynman` (mot de passe).
- Chemins : `HOME=/feynman/home/dphn/learn/lg211663` (code dans `T2F/fast_basis`), `SCRATCH=/feynman/scratch/dphn/learn/lg211663/t2f` (données et résultats). Données : `$SCRATCH/drift_pad5mm_new.h5` (100 k évènements actuels).
- Python : `/opt/ohpc/pub/apps/anaconda/3.11/2023.09/bin` (numpy 1.24.3, scipy 1.11.1, h5py 3.9.0) + shapely 2.2.0 dans `/feynman/scratch/dphn/learn/lg211663/pylib` (PYTHONPATH). **Ne pas** installer numpy par pip dans pylib (numpy 2.x casse h5py).
- Environnement à positionner dans chaque script :

```
export PATH=/opt/ohpc/pub/apps/anaconda/3.11/2023.09/bin:$PATH
export PYTHONPATH=/feynman/scratch/dphn/learn/lg211663/pylib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
```

- Compilation : gcc 8.5, C++17 : OK (binaire `build/profile_basis_fast`).

## 3 bis. Scripts Slurm (`slurm/`)

- `task_prod.sh` : une tâche de tableau. Variables d'environnement : `FB, H5, OUT, SLICE, PYBIN, PYLIB, PRARGS`. Début = `K*SLICE`. Saute la tranche si `$OUT/slices/s_%05d_prod.h5` existe (reprise). Utilise `/tmp/${USER}_${SLURM_JOB_ID}_${K}` comme dossier d'échange (nettoyé par `trap`) et lance `prod_refine.py … --tmpdir … --exe ../build/profile_basis_fast --cfg ../drift.cfg --exe_args n_threads=1 --lam 1e-5 --chunk 10 ${PRARGS:-}`.
- `submit_prod.sh` :

```
bash slurm/submit_prod.sh FICHIER.h5 DOSSIER_SORTIE N_EVENEMENTS [SLICE=2000] [MAXPAR=150]
```

  Ne soumet que les tranches manquantes : `sbatch -p htc -N 1 -n 1 -c 1 --mem=4G -t 01:30:00 -J t2f_prod --array=<manquantes>%MAXPAR -o OUT/logs/task_%a_%A.log`. Arguments supplémentaires pour `prod_refine.py` : variable `PRARGS` (par ex. `PRARGS="--k_thr 5 --f_max 0.03"`).
- `merge_prod.py` :

```
python3 py/merge_prod.py DOSSIER N [SLICE=2000] [--out f.h5]
```

  Refuse de fusionner si une tranche manque ; écrit `all_prod.h5` et `all_prod_skipped.csv`.

### Séquence type (100 000 évènements, tranches de 334, 300 tâches)

```
cd /feynman/home/dphn/learn/lg211663/T2F/fast_basis
bash slurm/submit_prod.sh /feynman/scratch/dphn/learn/lg211663/t2f/drift_pad5mm_new.h5 /feynman/scratch/dphn/learn/lg211663/t2f/run100k_300 100000 334 300
squeue -u $USER
sacct -S <date> -X -o JobID,JobName,State,Elapsed,ExitCode
python3 py/merge_prod.py /feynman/scratch/dphn/learn/lg211663/t2f/run100k_300 100000 334
```

(Si une tâche échoue : relancer `submit_prod.sh` avec les mêmes arguments, seules les tranches manquantes repartent.)

### Lancement prévu pour 1 M évènements

```
bash slurm/submit_prod.sh <fichier_1M.h5> <dossier_sortie> 1000000 2000 300
```

puis `merge_prod.py <dossier_sortie> 1000000 2000`. (Tranches de 1000 avec `-t` plus court possibles si le diagnostic de concurrence le suggère.) Emplacement/nom du fichier 1 M : **à demander** (≈ 21 Go à transférer vers le scratch si le fichier est sur le Mac).

## 4. Résultats mesurés

- Performance : ≈ 0,52 s/évènement par hyperthread + 17 s de démarrage. Run 100 k : 300 tâches × 334 évènements, 217 s en moyenne par tâche, 16 min 21 s de durée réelle, ≈ 66 cœurs en moyenne (≤ 76). Extrapolation 1 M : ≈ 170–180 h-CPU ≈ 2,5 h à ≈ 70 cœurs.
- Équivalence Mac / cluster validée (`tests/cmp_h5.py` : discrets identiques, continus ≤ 1e-4 relatif).
- Run 100 k : 99 275 évènements, 726 lignes écartées (`all_prod_skipped.csv`), χ²_prof médian 0,994, 300 tâches COMPLETED, ≈ 20 % des évènements (20 430/99 275) atteignent le plafond de 8 itérations, avec qualité finale identique aux autres.
- Estimateur de départ (20 000 évènements) : `k_thr` 10→5 donne −8,5 % sur l'erreur angulaire (médiane 0,659→0,603°), x_v −29 %, évènements écartés 141→34 ; `f_max` < 0,03 sans effet ; `n_nb`=3 nuisible. Excès d'erreur dans les bandes d'azimut ≈ 57–75° et 108–123° persistant (hypothèse non prouvée : quantification de la grille de pads). Gain attendu en temps de calcul d'un meilleur départ : ≈ 1–10 %.

## 5. Outils de test et d'analyse

- `tests/analyze_iter.py`, `az_fine.py`, `start_bias.py`, `select_events.py` (sélection de groupes d'évènements au plafond), `cmp_h5.py`.
- Mac : `py/view_list.py` (visualisation d'une liste d'évènements, galerie HTML) et `py/viz_summary.py` (statistiques par groupe) : `python3 viz_summary.py liste.txt DIR`.
- Scripts d'aide cluster dans `/feynman/home/dphn/learn/lg211663/T2F/` : `test200.sh`, `test_t1.sh`, `rs20000.sh`, `rsvar.sh`, `rsvar2.sh`.

## 6. Points ouverts

1. Résultat de `rsvar2.sh` (variantes k3, k2, k1f0, k3n1) : `rsvar2.log` à analyser.
2. Test `--max_iter 16` sur la liste des évènements au plafond (vérifier la convergence réelle), via `view_list.py … --out viz_iter16 … --max_iter 16` puis `viz_summary.py events_iter8.txt viz_iter16`.
3. Diagnostic de la concurrence cluster (pourquoi ≈ 66–76 cœurs seulement) : `sacct`, `sacctmgr`, `scontrol`, `sshare`, `sinfo -s`.
4. Générateur de simulation (GEF + Geant4) : peut-il produire des évènements induits par neutrons (5–40 MeV, pic à 15 MeV, faisceau selon y) ? Non-colinéarité attendue ≈ 1–3° ; l'estimateur actuel (demi-plans séparés) y est adapté, un ajustement colinéaire global ne l'est pas (biais ≈ δ/2 par fragment).
5. Le 1 M reste-t-il bien la simulation sans faisceau ? Où se trouve le fichier ?
6. Évolutions possibles (seulement sur accord) : comparaison hors-ligne d'estimateurs de départ (courant / colinéaire / colinéaire avec a priori / colinéaire corrigé cinématique), test chaîne complète d'un meilleur départ via `PRARGS` sur 10 000 évènements, diagnostic d'historique des itérations LM, tests de tolérance/max_iter.