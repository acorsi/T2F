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

// ecrit le vecteur vec (valeurs aux noeuds en tau) puis les valeurs a mi-chemin (psi_at)
static void emit(FILE* f, const std::string& kind, double u, double v, double d,
                 const PsiTable& T, const std::vector<double>& vec, const float* vg)
{
    const int nt = T.n_tau();
    for (int n = 0; n < nt; ++n) {
        double tau = (T.m_min_index() + n) * T.dtau();
        fprintf(f, "%s_node,%.10g,%.10g,%.10g,%.10g,%.12g,", kind.c_str(), u, v, d, tau, vec[n]);
        if (vg) fprintf(f, "%.12g\n", (double)vg[n]); else fprintf(f, "nan\n");
    }
    for (int n = 0; n + 1 < nt; ++n) {
        double tau = (T.m_min_index() + n + 0.5) * T.dtau();
        fprintf(f, "%s_mid,%.10g,%.10g,%.10g,%.10g,%.12g,nan\n", kind.c_str(), u, v, d, tau,
                T.psi_at(&vec[0], tau));
    }
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage : " << argv[0] << " CAS(A|B|C) R\n";
        return 1;
    }
    const char cs = argv[1][0];
    const int  R  = atoi(argv[2]);

    DetectorConfig cfg;
    PadPlaneParams pp;
    TableOptions   opt;
    try {
        make_case(cs, cfg, pp);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    opt.R = R;                     // les autres options gardent les defauts de TableOptions (comme main.cpp avec drift.cfg)
    std::string tag;
    if (argc >= 4) tag = std::string("_") + argv[3];
    for (int ia = 4; ia < argc; ++ia) {
        std::string s(argv[ia]);
        size_t eq = s.find('=');
        if (eq == std::string::npos) { std::cerr << "argument inconnu : " << s << "\n"; return 1; }
        std::string key = s.substr(0, eq);
        double val = atof(s.substr(eq + 1).c_str());
        if      (key == "m_cells")  opt.m_cells  = (int)val;
        else if (key == "step_out") opt.step_out = val;
        else if (key == "n_d")      opt.n_d      = (int)val;
        else if (key == "n_img")    opt.n_img    = (int)val;
        else if (key == "kappa")    opt.kappa    = val;
        else if (key == "step_cap") opt.step_cap = val;
        else { std::cerr << "cle inconnue : " << key << "\n"; return 1; }
    }

    CheckList     chk;
    DerivedParams D;
    PsiTable*     table = 0;
    try {
        D = derive_params(cfg, pp, opt, chk);
        print_derived(D, cfg);
        chk.finish();
        clock_t t0 = clock();
        table = new PsiTable(D.tp);
        std::cout << "table construite en " << (double)(clock() - t0) / CLOCKS_PER_SEC << " s\n";
        check_table(*table, D, chk, opt.acc_tol);
        chk.finish();
    } catch (const std::exception& e) {
        std::cerr << "\nArret : " << e.what() << "\n";
        return 1;
    }
    const PsiTable& T = *table;
    const int nt = T.n_tau();
    const double p = D.pitch, a = D.a;

    char fname[128];
    snprintf(fname, sizeof(fname), "out/l1_%c_R%d", cs, R);
    std::string base = std::string(fname) + tag;

    // ---- meta-donnees (lues par le Python) ----
    {
        FILE* m = fopen((base + "_meta.txt").c_str(), "w");
        if (!m) { std::cerr << "impossible d'ecrire dans out/ (lancer depuis tests/)\n"; return 1; }
        fprintf(m, "case=%c\nR=%d\na=%.12g\ng=%.12g\npitch=%.12g\ndkg=%.12g\n", cs, R, a, cfg.dga, p, cfg.dkg);
        fprintf(m, "DT_kg=%.12g\nDT_ga=%.12g\nDL_kg=%.12g\nDL_ga=%.12g\nve_kg=%.12g\nve_ga=%.12g\n",
                cfg.DT_kg, cfg.DT_ga, cfg.DL_kg, cfg.DL_ga, cfg.ve_kg, cfg.ve_ga);
        fprintf(m, "delta_t=%.12g\ndtau=%.12g\nn_tau=%d\nm_min=%d\nstep_out=%.12g\nn_out=%d\nn_d=%d\n",
                cfg.delta_t, T.dtau(), nt, T.m_min_index(), T.step_out(), T.n_out(), D.tp.n_d);
        fprintf(m, "n_h=%d\nm_cells=%d\nn_img=%d\nr_src=%.12g\nphi_edge_max=%.6g\nu_max=%.12g\ntable_mb=%.6g\n",
                D.n_h, D.tp.m_cells, D.tp.n_img, D.tp.r_src, T.phi_edge_max(), T.u_max(), D.table_mb);
        fclose(m);
    }

    FILE* f = fopen((base + ".csv").c_str(), "w");
    fprintf(f, "kind,u,v,d,tau,psi,psi_g\n");

    std::vector<double> vec(nt), vsum(nt);
    std::vector<float>  vg(nt);

    // ---- 1. points sur les noeuds de la table ----
    {
        const int nd = D.tp.n_d;
        const int kk[4] = { 0, nd / 3, (2 * nd) / 3, nd - 1 };
        for (int k = 0; k < 4; ++k) {
            const double step = T.step_node(kk[k]);              // pas de la grille de CE noeud en d
            const int ia = (int)std::floor(a / step + 0.5);
            const int ii[3] = { 0, ia, 2 * ia };
            const int jj[2] = { 0, ia };
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 2; ++j) {
                    double u = ii[i] * step, v = jj[j] * step, d = T.d_node(kk[k]);
                    T.psi_vector(u, v, d, &vec[0]);
                    T.psi_vector_g(u, v, d, &vg[0]);
                    emit(f, "node", u, v, d, T, vec, &vg[0]);
                }
        }
    }

    // ---- 2. points au hasard ----
    {
        unsigned long seed = 987654321UL;
        const double dlo = getenv("DLO") ? atof(getenv("DLO")) : 0.02;
        for (int s = 0; s < 40; ++s) {
            double u = (2.0 * tau_rnd(seed) - 1.0) * 2.0 * p;
            double v = (2.0 * tau_rnd(seed) - 1.0) * 2.0 * p;
            double d = std::exp(std::log(dlo) + (std::log(cfg.dkg) - std::log(dlo)) * tau_rnd(seed));
            T.psi_vector(u, v, d, &vec[0]);
            T.psi_vector_g(u, v, d, &vg[0]);
            emit(f, "rand", u, v, d, T, vec, &vg[0]);
        }
    }

    // ---- 3. somme sur un pavage carre de tuiles (pas p) ----
    {
        const int L = (int)std::floor(T.u_max() / p) + 1;
        const double xs[4][2] = { {0, 0}, {1, 0}, {1, 1}, {0.3, 0.7} };     // en unites de a
        const double ds[5] = { 0.05, 0.5, 2.0, 8.0, cfg.dkg };
        std::vector<double> tmp(nt);
        for (int ip = 0; ip < 4; ++ip)
            for (int id = 0; id < 5; ++id) {
                double x = xs[ip][0] * a, y = xs[ip][1] * a;
                for (int n = 0; n < nt; ++n) vsum[n] = 0.0;
                for (int i = -L; i <= L; ++i)
                    for (int j = -L; j <= L; ++j) {
                        T.psi_vector(x - i * p, y - j * p, ds[id], &tmp[0]);
                        for (int n = 0; n < nt; ++n) vsum[n] += tmp[n];
                    }
                emit(f, "sum", x, y, ds[id], T, vsum, 0);
            }
    }
    fclose(f);
    std::cout << "ecrit : " << base << ".csv  et  " << base << "_meta.txt\n";
    delete table;
    return 0;
}
