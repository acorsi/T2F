// ---------------------------------------------------------------------------
// Niveau 1 : table d'induction Psi(u, v, tau, d)  (psi_builder.cpp + psi_table.cpp)
//
// Usage :  ./test_level1 CAS R [TAG [cle=valeur ...]]
//            CAS = A, B ou C ;  R = table_R (pas de table = delta_t / R)
//            TAG : suffixe des fichiers de sortie (pour comparer plusieurs reglages)
//            cle = m_cells, step_out, n_d ou n_img : surcharge de TableOptions (comme table_m_cells, ... de drift.cfg)
//
// Ce programme construit la table exactement comme main.cpp (derive_params puis PsiTable),
// puis ecrit dans out/l1_<CAS>_R<R>.csv les valeurs lues dans la table par PsiTable
// (psi_vector, psi_vector_g, psi_at). AUCUNE comparaison ici : les references exactes sont
// calculees en Python (test_level1.py), par une methode independante.
//
// Lignes ecrites (colonne kind) :
//   node_node / node_mid : points tombant exactement sur les noeuds (u,v,d) de la table,
//                          donc teste build_psi seul ; _node = aux noeuds en tau, _mid = a mi-chemin (psi_at)
//   rand_node / rand_mid : points tires au hasard (interpolation cubique en u, v, d comprise)
//   sum_node  / sum_mid  : somme de Psi sur tout un pavage carre de tuiles, nuage en (x, y)
//
// C++ classique.
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include "detector.hpp"
#include "derived_params.hpp"      // inclut pad_geometry.hpp et psi_table.hpp

static void make_case(char c, DetectorConfig& cfg, PadPlaneParams& pp)
{
    cfg = DetectorConfig();        // defauts de detector.hpp (gaz P10, dkg = 20, dga = 0.3, delta_t = 10)
    pp  = PadPlaneParams();
    if (c == 'A') {                // drift.cfg actuel : 90% Ar + 10% CF4, pad 5 mm sans anneaux
        cfg.dga   = 0.3;
        cfg.DT_kg = 0.02;  cfg.DT_ga = 0.04;
        cfg.DL_kg = 0.012; cfg.DL_ga = 0.016;
        cfg.ve_kg = 0.010; cfg.ve_ga = 0.003;
        pp.pad_size = 0.5; pp.inter_pad = 0.015;
        pp.r_quinconce = 10; pp.r_transition = 10; pp.r_active = 10; pp.n_rings = 0;
    } else if (c == 'B') {         // P10, pad 3 mm avec anneaux, gap 3 mm (bloc commente de drift.cfg)
        cfg.dga = 0.3;
        pp.pad_size = 0.3; pp.inter_pad = 0.015;
    } else if (c == 'C') {         // P10, pad 5 mm, gap 5 mm (dga = 0.5 : choisi par moi)
        cfg.dga = 0.5;
        pp.pad_size = 0.5; pp.inter_pad = 0.015;
        pp.r_quinconce = 10; pp.r_transition = 10; pp.r_active = 10; pp.n_rings = 0;
    } else {
        throw std::runtime_error("CAS doit etre A, B ou C");
    }
}


int main(int argc, char* argv[])
{
    if (argc < 3) { std::cerr << "Usage : " << argv[0] << " CAS R [step_out=..] [n_d=..]\n"; return 1; }
    const char cs = argv[1][0];
    DetectorConfig cfg; PadPlaneParams pp; TableOptions opt;
    make_case(cs, cfg, pp);
    opt.R = atoi(argv[2]);
    for (int ia = 3; ia < argc; ++ia) {
        std::string s(argv[ia]); size_t eq = s.find('=');
        if (eq == std::string::npos) continue;
        if (s.substr(0, eq) == "step_out") opt.step_out = atof(s.substr(eq + 1).c_str());
        if (s.substr(0, eq) == "kappa") opt.kappa = atof(s.substr(eq + 1).c_str());
        if (s.substr(0, eq) == "n_d") opt.n_d = atoi(s.substr(eq + 1).c_str());
    }
    CheckList chk; DerivedParams D = derive_params(cfg, pp, opt, chk);
    PsiTable T(D.tp);
    const int nt = T.n_tau();
    std::vector<double> tmp(nt);
    const double p = D.pitch;
    const int N = 400000;
    double acc = 0;
    // (1) points au hasard (pire cas pour le cache)
    unsigned long seed = 42UL;
    std::vector<double> U(N), V(N), Dd(N);
    for (int n = 0; n < N; ++n) {
        U[n] = (2.0 * tau_rnd(seed) - 1.0) * 1.5 * p; V[n] = (2.0 * tau_rnd(seed) - 1.0) * 1.5 * p;
        Dd[n] = std::exp(std::log(0.02) + (std::log(cfg.dkg) - std::log(0.02)) * tau_rnd(seed));
    }
    clock_t t0 = clock();
    for (int n = 0; n < N; ++n) { T.psi_vector(U[n], V[n], Dd[n], &tmp[0]); acc += tmp[nt - 1]; }
    double t_rand = (double)(clock() - t0) / CLOCKS_PER_SEC / N * 1e6;
    // (2) marche coherente : positions voisines (voxels le long d'une trace), d lentement variable
    t0 = clock();
    double u = 0, v = 0, d = 1.0;
    for (int n = 0; n < N; ++n) {
        u += 0.002 * (2.0 * tau_rnd(seed) - 1.0); v += 0.002 * (2.0 * tau_rnd(seed) - 1.0);
        d = 1.0 + 0.5 * std::sin(n * 1e-4);
        T.psi_vector(u, v, d, &tmp[0]); acc += tmp[nt - 1];
    }
    double t_coh = (double)(clock() - t0) / CLOCKS_PER_SEC / N * 1e6;
    printf("cas %c R=%d step_out=%.3g mm n_d=%d : n_out=%d n_tau=%d table=%.0f Mo | %.2f us/appel (hasard)  %.2f us/appel (coherent)  [%g]\n",
           cs, opt.R, 10 * T.step_out(), D.tp.n_d, T.n_out(), nt, D.table_mb, t_rand, t_coh, acc);
    return 0;
}
