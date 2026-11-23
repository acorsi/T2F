# T2F – chaîne de reconstruction des profils dE/dx : référence sourcée

Chaque affirmation renvoie à un fichier et une fonction lus dans cette discussion. Ce qui n'a pas pu être lu est listé en section 13. Ce document **remplace** `T2F_flux_donnees_dedx.md` (qui contenait des approximations, corrigées en section 12, point 6). `T2F_fast_basis_resume.md` (cluster) reste valable.

**Règle absolue** : aucune identité H/L dans la reconstruction. Convention : fragment 0 = demi-plan y > y_plane, fragment 1 = demi-plan y < y_plane (`prod_io.py`, `refine_geom.geometry_est`).

---

## 1. Fichiers et rôles

| Fichier | Rôle |
|---|---|
| `t2f_io.py` | lecture du HDF5 : `pad_geometry`, `pad_xy`, `event_pads`, `read_vertex`, `load_pad_capacitances` |
| `reco_dir.py` | `pad_waveforms`, `strong_pads`, `true_direction` (vérité, analyse seulement), `pct` |
| `reco_split.py` | `estimate` : géométrie de départ (vertex commun, une demi-droite par demi-plan) |
| `rings.py` | `PadRings` (pads non lus : couronnes 1 et 2), `zero_window` |
| `fit_profile.py` | requête/réponse C++ (`write_request`, `read_response_iter`), `Whitener`, `solve_profile_r` (profil), `half_length`, `dist_to_segment` |
| `refine_geom.py` | `Env`, `Server`, `Ctx`, `evaluate`, `lm_step`, `refine_chunk` (Levenberg–Marquardt), `tail_state`, `extend_chunk` ; `main()` = version « analyse » avec vérité |
| `prod_refine.py` | version production (aucune vérité) : `build_state_prod`, `extend_chunk_prod`, `Collector` |
| `prod_io.py` | format de sortie et fusion |
| `whiten_fast.py`, `jac_fast.py` | accélérations sans changer le résultat (installées par `prod_refine.py`, désactivables par `WHITEN_FAST=0`, `JAC_FAST=0`) |
| `profile_basis_fast.cpp` (+ `pad_geometry.cpp`, `psi_table.cpp`, `psi_builder.cpp`, `weighting_potential.cpp`, `noise.cpp`) | serveur C++ : base linéaire (réponse de la chaîne induction + préampli + CAN à chaque fonction « chapeau ») |

---

## 2. Entrées lues par la production (`prod_refine.py`)

Seulement (docstring de `prod_refine.py`) : `/metadata`, `/pads`, `/event_id`, `/P/{pid, pk0, pnt, pqn, phase}`. Pas de `/H`, `/L`, `/vertex`, `/P/pq`, `/P/pflag`.

Structure du fichier (`h5ls -r drift_pad5mm_new.h5`, 100 000 évènements, 1114 pads) :
`/H`, `/L` {KE, dE, phi, theta} ; `/P` {n_ghost, pflag, phase, pid, pk0, pnt, pq, pqn} ; `/event/angle` ; `/event_id` ; `/metadata` (attributs) ; `/pads` {c_pF, enc, id, r_in, r_out, size, sx, sy, x, y} ; `/vertex` {x, y, z}.

Attributs `/metadata` utilisés (`refine_geom.Env`, `t2f_io`) : `delta_t`, `adc_ratio` (R), `ve_kg`, `dkg`, `dga`, `ve_ga`, `W`, `step_size_eff`, `noise_enable`, `read_thr_e` (seuil de lecture, mis à 0 si `noise_enable` ≠ 1), `pad_size`, `inter_pad` (pitch = somme), `emission_axis`. Les attributs de la réponse électronique sont lus par `view_fit.Response` et `FitModel` (non lu, voir §12). Capacités et ENC par pad : `/pads/c_pF`, `/pads/enc`.

Données par évènement (`event_pads`) : `pid` (pads écrits), `pk0` (premier échantillon CAN écrit), `pnt` (nombre d'échantillons écrits), `pqn` (échantillons concaténés des pads), `phase`. Temps d'un échantillon : t = (phase + j·R)·delta_t avec j = pk0 + indice (`reco_dir.pad_waveforms`).

Le fichier d'entrée de la simulation et ses paramètres sont dans `drift.cfg` (lu aussi par le serveur C++ ; il porte entre autres `preamp_response = minos`, `preamp_minos_peak_ns = 388`, `adc_period_ns = 20`, `read_thr_sigma = 4`, `noise_model = capacitance`, `noise_enable = 1`).

---

## 3. Flux d'un évènement (production)

```
read_event → pad_waveforms
   → reco_split.estimate       : vertex (x_v, z_v), directions (u_p, u_m)
   → half_length (x2)          : longueur de chaque demi-droite
   → pads proches de la trace  : pads utilisés + échantillons
   → (rings.split, zero_window): pseudo-données nulles (mises de côté)
   → refine_geom.refine_chunk  : Levenberg–Marquardt sur [x, z, a0, m0, a1, m1]
   → recalcul de L (lrecalc)
   → extend_chunk_prod         : rallonge de grille, profil final
   → Collector.add → prod_io (HDF5)
```
Les évènements sont traités par lots de `--chunk` (10 par défaut) : une requête C++ contient tous les évènements actifs du lot.

---

## 4. Estimation de départ : `reco_split.estimate`

1. Pour chaque pad écrit : amplitude `amp` = maximum des échantillons ; `qint` = Σ échantillons × (pas d'échantillon) / `area_h` (aire de la réponse h).
2. Seuil : `amin = max(k_thr · thr, f_max · amp.max())` (défauts k_thr 10, f_max 0,1 ; `thr` = `read_thr_e`).
3. `reco_dir.strong_pads(amp, qint, amin, x, y, pitch, n_nb)` : pad « fort » si `amp ≥ amin` et `qint > 0,5·amp` ; avec `n_nb` > 0 (défaut 2), il faut en plus au moins `n_nb` autres pads « de base » à distance < 1,5·pitch.
4. Pads centrés sur y_plane (|y − y_plane| < 1e-9) écartés ; demi-plans `ip` (y > y_plane) et `im` (y < y_plane) ; il faut ≥ `min_half` (3) pads dans chacun, sinon évènement écarté (« demi-plan avec moins de 3 pads forts »). Aucun pad écrit → « aucun pad ».
5. **Vertex commun** P_v = (x_v, y_plane) : pour x_v donné et un demi-plan, la direction est le vecteur propre de plus petite valeur propre de M_h(x_v) = Σ w_i D_i D_iᵀ avec D_i = X_i − P_v et w = amplitude. Cette valeur propre est le coût. `fit_vertex_x` minimise la somme des deux coûts : balayage de 700 points entre min(x)−3 et max(x)+3 cm, puis 40 itérations de section dorée autour du minimum. La demi-droite est orientée du vertex vers le nuage de pads (`half_direction`). Le vertex reste dans le plan y = y_plane.
6. **Temps** : tc_i = barycentre temporel du pad − m_h (barycentre de h). Modèle t_i = t_v + b_h·s_i, s_i = D_i·e_h. Moindres carrés pondérés (poids = amplitude) **conjoints** sur (t_v, b_p, b_m). Pente m_h = −sign_z · ve_kg · b_h ; z_v = dkg − ve_kg·(t_v − t_off) ; t_off = (dga/ve_ga)/2 − delta_t/2 par défaut.
7. Direction 3D : u = (e_x, e_y, m)/‖·‖ (`u_p`, `u_m`).
8. En plus, pour comparaison seulement, `estimate` calcule chaque demi-plan seul (ACP autour du centroïde, intersection avec y = y_plane si |e_y| ≥ 0,05) : non utilisé par la production.

La vérité (`true_direction`, `read_vertex`) n'intervient que dans `reco_split.main()` pour mesurer l'erreur.

---

## 5. Longueur et pads utilisés : `fit_profile.half_length`, `dist_to_segment`, `build_state_prod`

- `half_length(geo, k, ev, amp, y_plane, thr, k_end, axis_dist·pitch, lmargin)` : parmi les pads du bon demi-plan (signe selon k), à distance perpendiculaire ≤ `axis_dist`·pitch de l'axe (défaut 1,5) avec s > 0 et `amp ≥ k_end·thr` (k_end = 3), L = s_max/u_xy + `lmargin` (0,5 cm). Pas de marge adaptative en production : l'argument `pitch` n'est pas passé (valeur par défaut 0). Aucun pad → évènement écarté (« longueur : aucun pad fort dans un demi-plan »).
- Pads utilisés : pads écrits à distance du segment ≤ `pad_dist`·pitch (1,5) et `pnt > 0`. Aucun → « aucun pad proche de la trace ». Tous leurs échantillons écrits sont utilisés. **Cet ensemble de pads et les longueurs de départ restent fixes pendant l'affinage de la géométrie.**
- `y` = concaténation des échantillons `pqn` de ces pads.

---

## 6. Modèle linéaire (serveur C++)

D'après les commentaires de `profile_basis_fast.cpp` :

- Le profil inconnu q(l) (électrons par cm) est décrit par des fonctions « chapeau » de pas `delta` (0,2 cm par défaut), centrées en l_m = m·delta, m = 0…M−1, M = ceil(L/delta) + 1, par demi-droite.
- Pour chaque chapeau, le serveur calcule le signal produit sur les échantillons demandés par la même chaîne que `./process` : voxels (n_e = w(s)·step_eff, w = 1 − |s − m·delta|/delta, positions s = step_eff·(i+0,5)) → induction (table de Ψ) → convolution par la réponse h → échantillons CAN d'indice phase + j·adc_ratio. La trace s'arrête si un voxel atteint la grille (z ≥ dkg).
- Sortie binaire par évènement : en-tête (4 int32 : id, M_H, M_L, n_rows) puis matrice `A` de taille (M_H + M_L) × n_rows (lignes = chapeaux du fragment 0 puis du fragment 1, colonnes = échantillons demandés dans l'ordre de la requête). Unité : électrons équivalents par (électron/cm) de coefficient. **Les noms « H » et « L » dans le C++ désignent seulement les indices de fragment 0 et 1.**
- Mode serveur : `@@READY`, une ligne « requete reponse » par requête, `@@DONE` ou `@@ERROR`, `quit`. Option `n_threads=` (clé=valeur, `--exe_args`).
- Version rapide : format de requête étendu `#fast 1` avec `mask` par évènement (1 = fragment 0, 2 = fragment 1, 3 = les deux) ; la réponse ne contient alors que les colonnes demandées. Le résultat est identique à l'arrondi près à l'ancien (affirmé dans l'en-tête).
- Physique de l'induction (commentaires uniquement) : potentiel de pondération par angle solide de rectangle (`weighting_potential.cpp`), convolution gaussienne par cellule (`psi_builder.cpp`), interpolation de Lagrange à 4 points (`psi_table.cpp`).

---

## 7. Profil à géométrie fixée : `fit_profile.Whitener`, `solve_profile_r`

1. **Blanchiment** : bloc de bruit par pad, Cholesky L de `model.cov(n, ser_scale)` (`view_fit.FitModel`, non lu) ; `ser_scale` dépend de l'ENC du pad (`/pads/enc`). `A` et `y` sont blanchis par pad : A_w = A·L⁻ᵀ, y_w = L⁻¹y. Sans bruit (`noise_enable` ≠ 1) : pas de blanchiment. `whiten_fast.py` précalcule L⁻¹ (même résultat).
2. **Problème** : G0 = A_w A_wᵀ ; second membre A_w y_w ; pénalité de courbure d'ordre 2 : G = G0 + (lam·alpha2/delta³)·DᵀD + 1e-10·trace(G0)/nb·I, avec `D` = différences secondes appliquées séparément à chaque fragment, et `alpha2` = médiane des diag(G0)/delta² (calculée au premier appel, puis figée par l'appelant).
3. **Solution** : Cholesky G = Lc·Lcᵀ, puis NNLS (`scipy.optimize.nnls`) sur ‖Lcᵀc − Lc⁻¹ rhs‖² avec c ≥ 0.
4. **Incertitudes** : `sig = sqrt(diag(G⁻¹))`.
5. **Résidu complet** `r = [A_wᵀc − y_w ; sqrt(wreg)·D c]` (données puis pénalité).
6. **Lissage relâché en fin de trace** : `endw = (fac, js_0, js_1)` ; les lignes de D dont le nœud central a l'indice ≥ js sont multipliées par sqrt(fac). Dans `evaluate`, js_k = max(1, ceil((L_k − dL_k − zone)/delta − 1e-9)), avec fac = `lam_end_fac` (0,3), zone = `lam_end_zone` (1,2 cm). Actif seulement quand `S["dL"]` existe, c'est-à-dire après l'affinage de la géométrie.

---

## 8. Affinage de la géométrie : `refine_geom.refine_chunk`

- 6 paramètres p = [x, z, a0, m0, a1, m1] ; a = azimut de (u_x, u_y), m = u_z/u_xy (`params_from_geo`, `geo_from_params`). Vertex dans le plan y = y_plane.
- Objectif : F = ‖r‖² (résidu complet de §7, profil optimisé à chaque évaluation, lissage `lam_geom` = 1e-4). `alpha2` fixé à l'évaluation initiale.
- Jacobien par différences finies, pas FD_STEP = [0,02 cm, 0,05 cm, 0,005 rad, 0,02, 0,005 rad, 0,02] ; `jac_fast` ne recalcule que le fragment concerné pour a0, m0 (fragment 0) et a1, m1 (fragment 1), d'après sa docstring.
- Pas LM : (JᵀJ + mu·diag(JᵀJ))⁻¹ Jᵀr, limité par MAX_STEP = [0,15, 0,4, 0,06, 0,25, 0,06, 0,25] (`lm_step`). mu initial 1e-2.
- À chaque itération et pour chaque évènement actif : un seul pas d'essai ; `nit` += 1 (**y compris pour un pas rejeté**). Pas accepté si F diminue : mu ← max(mu/3, 1e-4) ; terminé si la diminution relative < `tol` (1e-4) ; sinon nouveau jacobien. Pas rejeté : mu ← 5·mu ; terminé si mu > 1e6.
- Au plus `max_iter` (8) itérations.
- En fin de `refine_chunk` : `chi2d1` = χ² réduit des lignes de données seules (`r[:nr]`), puis `use_full_rows` ajoute les pseudo-données nulles pour la suite.
- Les pseudo-données nulles ne sont pas dans l'affinage de la géométrie par défaut (`zero_in_geom` = 0).

---

## 9. Pads non lus comme contraintes : `rings.py`, `build_state_prod`

- Pads écrits sans voisin écrit à moins de `ring_fac`·pitch (1,5) : fantômes, retirés (une passe). Pads touchés = écrits non retirés.
- Couronne 1 : non touchés à moins du rayon d'un pad touché ; couronne 2 : non touchés, hors couronne 1, à moins du rayon d'une couronne 1.
- En production : pseudo-données nulles pour la couronne 2 (bruit propre du pad) et la couronne 1 en contrainte faible (lignes pondérées par 1/`ring1_k`, avec `ring1_k` = 4). Fenêtre commune `zero_window(...)` élargie du décalage temporel maximal d'une rallonge de `ext_max` cm.
- Elles sont rangées après les vraies lignes ; `nr` = nombre de vraies lignes, donc les χ²/ndl restent ceux des données.

---

## 10. Fin de profil et profil final : `tail_state`, `extend_chunk_prod`

1. Longueurs recalculées avec la géométrie affinée (`lrecalc` = 1) : `S["Lr"]`.
2. Première évaluation (`lam` = `--lam`, 1e-5 par défaut) avec les longueurs de base `Lb`, pseudo-données incluses et lissage de fin relâché.
3. `tail_state` par fragment : ratio = moyenne des `n_tail` (3) derniers nœuds / maximum du fragment ; fin non contenue si maximum > 0 et ratio > `f_tail` (0,10) ; `tail_sigma` = sqrt(Σ sig²)/n_tail/maximum.
4. Si `--ext` (1) : tant qu'un fragment est « non contenu », rallonge `dL` += `ext_step` (0,5 cm) jusqu'à `ext_max` (3,0 cm), ré-évaluation avec `alpha2` figé (celui de la première passe).
5. Le profil final est celui de la **dernière** évaluation de cet évènement (`S["fin"]`).
6. `chi2_prof` = ‖r[:nr]‖²/nr de ce profil.

---

## 11. Sortie : `prod_io.py`, `Collector`

Un fichier par tranche, écrit sous `.h5.part` puis renommé. Champs :

| champ | forme | contenu |
|---|---|---|
| `event_index`, `event_id` | n | indice dans le fichier d'entrée ; `/event_id` |
| `vertex` | n×3 | x, y (= y_plane), z (cm) |
| `u0`, `u1` | n×3 | directions unitaires des fragments 0 et 1 |
| `params` | n×6 | x, z, a0, m0, a1, m1 |
| `chi2_start`, `chi2_geom`, `chi2_prof` | n | χ²/ndl : départ et géométrie affinée (lissage `lam_geom`), profil final |
| `n_iter` | n | itérations LM (pas rejetés compris) |
| `L_base`, `dL` | n×2 | longueur avant rallonge ; rallonge (cm) |
| `not_contained` | n×2 | 1 = fin non contenue après rallonge |
| `tail_ratio`, `tail_sigma` | n×2 | voir §10 |
| `n_nodes` | n×2 | nombre de nœuds par fragment |
| `prof_off` | n+1 | `prof_val[prof_off[e]:prof_off[e+1]]` = nœuds du fragment 0 puis du fragment 1 |
| `prof_val`, `prof_sig` | concaténés | coefficients c et `sig` de §7 |

Attributs : `delta`, `lam`, `lam_geom`, `lam_end_fac`, `lam_end_zone`, `y_plane`, `ext_step`, `ext_max`, `convention`, `source`, `start`, `version`. Un fichier `<out>_skipped.csv` liste les évènements écartés et la raison.

Raisons d'écart en production : « estimation : aucun pad », « estimation : demi-plan avec moins de 3 pads forts », « longueur : aucun pad fort dans un demi-plan », « aucun pad proche de la trace ».

Nœuds : position l = m·delta le long de la demi-droite depuis le vertex ; les nœuds du fragment 0 sont `prof_val[off : off + n_nodes[0]]`, ceux du fragment 1 suivent.

---

## 12. Points à connaître (déduits de la lecture du code)

1. **Unité de `prof_val`** : `prod_io.py` annonce « e-/noeud, comme le serveur », alors que `fit_profile.py` (docstring) et l'en-tête C++ décrivent un profil de charge en **électrons par cm** (`truth_nodes` : q = dE/W/step). Le code lu ne tranche pas ; la comparaison avec `truth_nodes` plaide pour des e⁻/cm. Les valeurs sont une charge (dE/W), pas une énergie : passer à une énergie demande W et le facteur de conversion.
2. **`prof_sig`** = sqrt(diag(G⁻¹)) avec G = G0 + pénalité, c'est-à-dire non contraint et sans la formule de covariance de l'estimateur linéaire G⁻¹G0G⁻¹ (calculée séparément dans `refine_geom.main()` pour le diagnostic, non utilisée en production). Comme G0 ≤ G, `prof_sig` majore cette covariance. Le code ne dit pas de combien dans les évènements réels.
3. Le critère de fin de LM (`tol`) ne s'applique qu'aux pas acceptés ; un évènement qui n'a pas convergé après 8 pas essayés sort quand même (≈ 20 % sur le run 100 k).
4. L'ensemble de pads et les longueurs de départ sont figés pendant l'affinage ; seules les longueurs sont recalculées ensuite (`lrecalc`).
5. Dans `reco_split.py`, la docstring parle de « b_H, b_L » : c'est un reste de nomenclature ; le code utilise p et m (demi-plans). L'identité H/L n'intervient que dans les scripts d'analyse (`reco_split.main`, `fit_profile.geometry_true`, `refine_geom.main`) pour comparer à la vérité, pas en production.
6. Corrections par rapport à `T2F_flux_donnees_dedx.md` : l'estimation de départ n'est pas « une ACP par demi-plan » mais un vecteur propre autour du vertex commun (M_h(x_v)) ; la régression en temps est conjointe (t_v, b_p, b_m) ; la sortie `prof_off` a n+1 éléments (fragment 0 puis 1) ; le jacobien et le lissage de fin sont décrits ici d'après le code.

---

## 13. Ce qui n'a pas été lu (donc non décrit)

- `view_fit.py` : classes `Response` (réponse électronique, `tf`, `hcont`, `area`, `descr`) et `FitModel` (modèle de bruit : `cov`, `ser_scale_pad`). Le blanchiment dépend de `FitModel.cov` : son contenu n'est pas décrit ici.
- `view_pads.py` (importé par `view_fit.py`).
- Corps de `jac_fast.py` au-delà de sa docstring, `whiten_fast.py` au-delà de sa docstring, et corps des fichiers C++ au-delà des en-têtes et commentaires.

Pour compléter : fournir `view_fit.py` (et `view_pads.py` s'il intervient dans le modèle de bruit).