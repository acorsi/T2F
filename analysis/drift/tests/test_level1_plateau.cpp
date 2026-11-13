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


// charge finale par pad (plateau de Psi) : exacte = [Phi((a-u)/s)-Phi((-a-u)/s)] [Phi((a-v)/s)-Phi((-a-v)/s)],
// s^2 = DT_kg^2 d + DT_ga^2 g  (la charge finit toujours sur l'anode : c'est un melange gaussien du carre du pad)
static double Phi(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

int main(int argc, char* argv[])
{
    if (argc < 3) { std::cerr << "Usage : " << argv[0] << " CAS R [step_out=...]\n"; return 1; }
    const char cs = argv[1][0];
    DetectorConfig cfg; PadPlaneParams pp; TableOptions opt;
    make_case(cs, cfg, pp);
    opt.R = atoi(argv[2]);
    for (int ia = 3; ia < argc; ++ia) {
        std::string s(argv[ia]); size_t eq = s.find('=');
        if (eq != std::string::npos && s.substr(0, eq) == "step_out") opt.step_out = atof(s.substr(eq + 1).c_str());
        if (eq != std::string::npos && s.substr(0, eq) == "kappa") opt.kappa = atof(s.substr(eq + 1).c_str());
        if (eq != std::string::npos && s.substr(0, eq) == "n_d") opt.n_d = atoi(s.substr(eq + 1).c_str());
    }
    CheckList chk; DerivedParams D;
    D = derive_params(cfg, pp, opt, chk);
    PsiTable T(D.tp);
    const int nt = T.n_tau();
    const double p = D.pitch, a = D.a;
    std::vector<double> tmp(nt);
    unsigned long seed = 123456789UL;
    const double dlos[2] = { getenv("DLO") ? atof(getenv("DLO")) : 1.0, 0.02 };
    printf("cas %c step_out=%.3g mm :", cs, 10 * T.step_out());
    for (int id = 0; id < 2; ++id) {
        const int N = 20000;
        std::vector<double> e(N);
        double mx = 0, ss = 0, ux = 0, vx = 0, dx = 0;
        for (int n = 0; n < N; ++n) {
            double u = (2.0 * tau_rnd(seed) - 1.0) * 2.0 * p;
            double v = (2.0 * tau_rnd(seed) - 1.0) * 2.0 * p;
            double d = std::exp(std::log(dlos[id]) + (std::log(cfg.dkg) - std::log(dlos[id])) * tau_rnd(seed));
            T.psi_vector(u, v, d, &tmp[0]);
            double s = std::sqrt(cfg.DT_kg * cfg.DT_kg * d + cfg.DT_ga * cfg.DT_ga * cfg.dga);
            double ex = (Phi((a - u) / s) - Phi((-a - u) / s)) * (Phi((a - v) / s) - Phi((-a - v) / s));
            e[n] = std::fabs(tmp[nt - 1] - ex);
            ss += e[n] * e[n];
            if (e[n] > mx) { mx = e[n]; ux = u; vx = v; dx = d; }
        }
        std::sort(e.begin(), e.end());
        printf("  [d>=%.3g] max %.2e  p99 %.2e  rms %.2e (pire u=%.3f v=%.3f d=%.3f)", dlos[id], mx, e[(int)(0.99 * N)], std::sqrt(ss / N), ux, vx, dx);
    }
    printf("\n");
    return 0;
}
