// ---------------------------------------------------------------------------
// (copie de test_level1 : auto-controle et choix de R) table d'induction Psi(u, v, tau, d)  (psi_builder.cpp + psi_table.cpp)
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


// Usage : ./test_selfcheck CAS [R] [cle=val ...]   (cles : kappa step_out n_d m_cells ; R = 0 : choix automatique)
int main(int argc, char* argv[])
{
    if (argc < 2) { std::cerr << "Usage : ./test_selfcheck CAS [R] [cle=val ...]\n"; return 1; }
    DetectorConfig cfg; PadPlaneParams pp; TableOptions opt;
    make_case(argv[1][0], cfg, pp);
    opt.R = (argc > 2 && std::string(argv[2]).find('=') == std::string::npos) ? atoi(argv[2]) : 0;
    for (int ia = 2; ia < argc; ++ia) {
        std::string s(argv[ia]); size_t eq = s.find('=');
        if (eq == std::string::npos) continue;
        std::string k = s.substr(0, eq); double v = atof(s.substr(eq + 1).c_str());
        if (k == "kappa") opt.kappa = v; else if (k == "step_out") opt.step_out = v;
        else if (k == "n_d") opt.n_d = (int)v; else if (k == "m_cells") opt.m_cells = (int)v;
    }
    CheckList chk;
    try {
        clock_t t0 = clock();
        opt.R = resolve_table_R(cfg, pp, opt, chk);
        std::cout << "resolve_table_R : R = " << opt.R << "   (" << (double)(clock() - t0) / CLOCKS_PER_SEC << " s)\n";
        DerivedParams D = derive_params(cfg, pp, opt, chk);
        print_derived(D, cfg);
        chk.finish();
        t0 = clock();
        PsiTable T(D.tp);
        std::cout << "construction de la table : " << (double)(clock() - t0) / CLOCKS_PER_SEC << " s\n";
        check_table(T, D, chk, opt.acc_tol);
        chk.finish();
    } catch (const std::exception& e) {
        std::cout << "\nArret : " << e.what() << "\n";
        return 1;
    }
    std::cout << "tous les controles sont passes\n";
    return 0;
}
