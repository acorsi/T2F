# reco_profil : reconstruction des profils dE/dx des fragments de fission

Entrée : fichier h5 produit par le simulateur `drift` (`analysis/drift`).
Sortie : pour chaque évènement, vertex, directions des deux fragments, profil dE/dx le long de chaque fragment
(valeurs aux nœuds et incertitudes), drapeau « fin non contenue ».

**Règle absolue : aucune identité lourd/léger (H/L) dans la reconstruction.** Convention : fragment 0 = demi-plan
y > y_plane, fragment 1 = demi-plan y < y_plane.

## Contenu

| Dossier | Rôle |
|---|---|
| `reco/` | Python, **à plat** (les scripts s'importent par leur nom, ne pas les répartir en sous-dossiers) |
| `src/` | `profile_basis_fast.cpp` : serveur C++ (base de fonctions « chapeau »). La physique vient de `drift` |
| `slurm/` | `submit_prod.sh`, `task_prod.sh` : production sur le cluster |
| `tests/` | non-régression et comparaisons |
| `docs/` | notes de référence (chaîne de traitement, exploitation cluster) |
| `CMakeLists.txt` | compile `profile_basis_fast` avec les sources de `drift` (option `DRIFT_DIR`) |

### Dans `reco/`

Cœur de la chaîne : `t2f_io`, `reco_dir`, `reco_split`, `rings`, `view_fit` (réponse électronique et modèle de bruit),
`view_pads`, `fit_profile`, `refine_geom`, `whiten_fast`, `jac_fast`, `prod_refine` (production), `prod_io`,
`merge_prod`, `cfg_from_h5`.
Outils : `view_list`, `viz_summary`, `view_profile` (visualisation), `diag_band`, `diag_couronnes` (diagnostics),
`refine_fast`, `run_par`, `profile_refine` (analyse avec vérité, exécution locale parallèle, profilage).

`view_pads.py` existe aussi dans `analysis/drift/analysis` (visualisation de la sortie de `drift`) : les deux copies sont
identiques au moment de la séparation.

## Dépendance à drift

Le serveur calcule la réponse à chaque chapeau avec la même chaîne que `./process`. Il compile donc des fichiers de
`drift` (`pad_geometry`, `weighting_potential`, `psi_builder`, `psi_table`, `noise` et leurs en-têtes) : aucune copie.
`PadPlane::project_hats` (balayage rapide de tous les chapeaux) fait partie de `drift/src/pad_geometry.cpp`.
Le cfg de la simulation n'est pas copié non plus : `reco/cfg_from_h5.py` le reconstruit depuis le h5 traité
(attribut `config_text` de `/metadata`), ce qui garantit que serveur et données ont la même configuration.

Le contrat entre les deux projets est le h5 de `drift`. La reconstruction ne lit que `/metadata`, `/pads`, `/event_id`
et `/P/{pid, pk0, pnt, pqn, phase}` (voir `docs/T2F_chaine_reco_reference.md`).

## Compilation

    cd reco_profil
    cmake -S . -B build && cmake --build build -j8

Si `drift` n'est pas à `../drift` : `cmake -S . -B build -DDRIFT_DIR=/chemin/vers/analysis/drift`.
Sur Mac, il faut un environnement Conda contenant libomp (ou `brew install libomp`).

## Utilisation locale

    python3 reco/cfg_from_h5.py sortie_drift.h5 drift_from_h5.cfg
    python3 reco/prod_refine.py sortie_drift.h5 --exe build/profile_basis_fast --cfg drift_from_h5.cfg \
        --exe_args n_threads=1 --lam 1e-5 --chunk 10 --tmpdir /tmp/mon_run --out mon_run

Sorties : `mon_run_prod.h5` (format : `reco/prod_io.py`) et `mon_run_skipped.csv` (évènements écartés et raison).
Options : `python3 reco/prod_refine.py --help`.

## Utilisation sur le cluster (Slurm)

    bash slurm/submit_prod.sh FICHIER.h5 DOSSIER_SORTIE N_EVENEMENTS [SLICE=2000] [MAXPAR=150]
    python3 reco/merge_prod.py DOSSIER_SORTIE N_EVENEMENTS [SLICE=2000]

`submit_prod.sh` reconstruit le cfg depuis le h5 et ne soumet que les tranches manquantes.
Détails d'exploitation, environnement et performances : `docs/T2F_fast_basis_resume.md`
(les chemins `fast_basis/py` de ce document correspondent à `reco/` ici).

## Tests

    bash tests/run_tests.sh /chemin/vers/profile_basis /chemin/vers/drift.cfg
    python3 tests/test_jac_fast.py 6
    python3 tests/cmp_h5.py ref_prod.h5 nouveau_prod.h5      # deux sorties de production : discrets identiques, continus <= 1e-4

`profile_basis` est l'exécutable de `drift` compilé avec `-DT2F_PROF=ON`.
