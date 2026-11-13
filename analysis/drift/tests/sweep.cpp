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


static double Phi(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

// Usage : ./sweep NOM cle=val ...   (cles : DT_kg DT_ga DL_kg DL_ga ve_kg ve_ga dga pad kappa n_d m_cells R)
int main(int argc, char* argv[])
{
    if (argc < 2) { std::cerr << "Usage : ./sweep NOM cle=val ...\n"; return 1; }
    std::string name = argv[1];
    DetectorConfig cfg; PadPlaneParams pp; TableOptions opt;
    pp.inter_pad = 0.015; pp.n_rings = 0; pp.r_quinconce = 10; pp.r_transition = 10; pp.r_active = 10;
    opt.R = 4; opt.n_d = 20; opt.kappa = 3.0;
    for (int ia = 2; ia < argc; ++ia) {
        std::string s(argv[ia]); size_t eq = s.find('=');
        if (eq == std::string::npos) { std::cerr << "argument inconnu " << s << "\n"; return 1; }
        std::string k = s.substr(0, eq); double v = atof(s.substr(eq + 1).c_str());
        if      (k == "DT_kg") cfg.DT_kg = v;   else if (k == "DT_ga") cfg.DT_ga = v;
        else if (k == "DL_kg") cfg.DL_kg = v;   else if (k == "DL_ga") cfg.DL_ga = v;
        else if (k == "ve_kg") cfg.ve_kg = v;   else if (k == "ve_ga") cfg.ve_ga = v;
        else if (k == "dga") cfg.dga = v;       else if (k == "pad") pp.pad_size = v;
        else if (k == "kappa") opt.kappa = v;   else if (k == "n_d") opt.n_d = (int)v;
        else if (k == "m_cells") opt.m_cells = (int)v;  else if (k == "R") opt.R = (int)v;
        else { std::cerr << "cle inconnue " << k << "\n"; return 1; }
    }
    CheckList chk;
    DerivedParams D = derive_params(cfg, pp, opt, chk);      // pas de chk.finish() : on veut mesurer meme hors plafond
    if (D.table_mb > 1500.0) { printf("%s m_cells=%d SKIP table=%.0f Mo\n", name.c_str(), D.tp.m_cells, D.table_mb); return 0; }
    clock_t t0 = clock();
    PsiTable T(D.tp);
    double tb = (double)(clock() - t0) / CLOCKS_PER_SEC;
    const int nt = T.n_tau();
    const double p = D.pitch, a = D.a, g = cfg.dga;
    const double sig0 = std::sqrt(cfg.DT_ga * cfg.DT_ga * g);

    char base[200];
    snprintf(base, sizeof(base), "out/sw_%s_mc%d", name.c_str(), D.tp.m_cells);
    {
        FILE* m = fopen((std::string(base) + "_meta.txt").c_str(), "w");
        if (!m) { std::cerr << "mkdir out ?\n"; return 1; }
        fprintf(m, "case=%s\nR=%d\na=%.12g\ng=%.12g\npitch=%.12g\ndkg=%.12g\n", name.c_str(), opt.R, a, g, p, cfg.dkg);
        fprintf(m, "DT_kg=%.12g\nDT_ga=%.12g\nDL_kg=%.12g\nDL_ga=%.12g\nve_kg=%.12g\nve_ga=%.12g\n",
                cfg.DT_kg, cfg.DT_ga, cfg.DL_kg, cfg.DL_ga, cfg.ve_kg, cfg.ve_ga);
        fprintf(m, "delta_t=%.12g\ndtau=%.12g\nn_tau=%d\nm_min=%d\n", cfg.delta_t, T.dtau(), nt, T.m_min_index());
        fclose(m);
    }
    std::vector<double> vec(nt);
    unsigned long seed = 20261111UL;
    // --- points pour le profil temporel (references calculees en Python) ---
    FILE* f = fopen((std::string(base) + ".csv").c_str(), "w");
    fprintf(f, "u,v,d,tau,psi\n");
    for (int s = 0; s < 36; ++s) {
        double u, v;
        double d = std::exp(std::log(0.02) + (std::log(cfg.dkg) - std::log(0.02)) * tau_rnd(seed));
        if (s < 12) {
            u = (2.0 * tau_rnd(seed) - 1.0) * 2.0 * p; v = (2.0 * tau_rnd(seed) - 1.0) * 2.0 * p;
        } else {                                 // pres d'un bord de pad : a +- 3 sigma_gap
            double sg = 3.0 * sig0;
            u = a + (2.0 * tau_rnd(seed) - 1.0) * sg;
            v = (s % 2) ? a + (2.0 * tau_rnd(seed) - 1.0) * sg : tau_rnd(seed) * a;
            if (tau_rnd(seed) < 0.5) u = -u;
        }
        T.psi_vector(u, v, d, &vec[0]);
        for (int n = 0; n < nt; ++n)
            fprintf(f, "%.10g,%.10g,%.10g,%.10g,%.12g\n", u, v, d, (T.m_min_index() + n) * T.dtau(), vec[n]);
    }
    fclose(f);
    // --- charge par pad : 6000 points, formule en erf ---
    double mx = 0;
    for (int s = 0; s < 6000; ++s) {
        double u = (2.0 * tau_rnd(seed) - 1.0) * 2.0 * p, v = (2.0 * tau_rnd(seed) - 1.0) * 2.0 * p;
        double d = std::exp(std::log(0.02) + (std::log(cfg.dkg) - std::log(0.02)) * tau_rnd(seed));
        T.psi_vector(u, v, d, &vec[0]);
        double sg = std::sqrt(cfg.DT_kg * cfg.DT_kg * d + cfg.DT_ga * cfg.DT_ga * g);
        double ex = (Phi((a - u) / sg) - Phi((-a - u) / sg)) * (Phi((a - v) / sg) - Phi((-a - v) / sg));
        mx = std::max(mx, std::fabs(vec[nt - 1] - ex));
    }
    printf("%s m_cells=%d a=%.4f g=%.3f sigma_gap=%.4f cm step0=%.4f cm n_out=%d n_tau=%d table=%.0f Mo build=%.1fs plateau_max=%.2e\n",
           name.c_str(), D.tp.m_cells, a, g, sig0, T.step_out(), T.n_out(), nt, D.table_mb, tb, mx);
    return 0;
}
