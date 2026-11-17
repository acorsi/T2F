// ---------------------------------------------------------------------------
// Test "trace" : induction sur les pads voisins d'une trace courte (cas A, drift.cfg).
//
// Usage :  ./test_track [R]        (R = table_R, 5 par defaut = valeur choisie par resolve_table_R pour A)
//
// Geometrie (repere local, pitch p) : le pad P0 est en (0,0), son voisin de la meme rangee P1 en (p,0),
// le pad en quinconce de la rangee du dessus P2 en (p/2, p) (au-dessus du milieu de la trace), le pad
// P3 en (0,-p) (rangee du dessous, aligne avec P0). Ce sont les positions de PadPlane::build_anode.
//
//   cas 1 : trace horizontale de (0,0) a (p,0), a d = 10 cm partout.
//   cas 2 : meme projection, mais inclinee a 45 degres dans le plan (x, d) : d = 10 + (x - p/2),
//           donc P0 est cote anode (d plus petit, arrivee a la grille plus tot).
// N voxels de poids egal (charge totale 1), au milieu de N tranches egales de la projection.
//
// Sorties (dans out/) :
//   track_simu.csv : Q(t) par pad, somme sur les voxels de Psi lu dans la table (psi_vector + psi_at)
//   track_lc.csv   : pour chaque (cas, pad, voxel) et chaque niveau h du gap, la partie "collectee"
//                    Lc(h) de la REFERENCE (electrons qui finissent sur le pad), par quadrature 2D
//   track_meta.txt : parametres
// La reference du total et le calcul en temps sont faits en Python (test_track.py).
//
// Partie collectee. Un electron a la hauteur h, en (x, y), finit sur l'anode a (x, y) + gaussienne de
// variance DT_ga^2 h par axe. Sa probabilite de finir sur la tuile |x'|<a, |y'|<a vaut le produit
// P(x) P(y), P(x) = Phi((a-x)/s_h) - Phi((-a-x)/s_h), s_h = DT_ga sqrt(h). Donc
//   Lc(u,v,h) = Int Int dx dy  G(x-u; sigma(h)) G(y-v; sigma(h)) phi(x,y,h) P(x) P(y)
// avec sigma(h)^2 = DT_kg^2 d + DT_ga^2 (g-h) (meme modele que le total). On utilise phi_point
// (valide au niveau 0 contre Fourier) et des noeuds de Gauss-Legendre, resserres pres des bords de la tuile.
//
// C++ classique.
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>
#include "detector.hpp"
#include "derived_params.hpp"      // inclut pad_geometry.hpp et psi_table.hpp
#include "weighting_potential.hpp"

static const double PI = 3.14159265358979323846;

static void make_case_A(DetectorConfig& cfg, PadPlaneParams& pp)
{
    cfg = DetectorConfig();
    pp  = PadPlaneParams();
    cfg.dga   = 0.3;
    cfg.DT_kg = 0.02;  cfg.DT_ga = 0.04;
    cfg.DL_kg = 0.012; cfg.DL_ga = 0.016;
    cfg.ve_kg = 0.010; cfg.ve_ga = 0.003;
    pp.pad_size = 0.5; pp.inter_pad = 0.015;
    pp.r_quinconce = 10; pp.r_transition = 10; pp.r_active = 10; pp.n_rings = 0;
}

static double Phi(double z) { return 0.5 * std::erfc(-z / std::sqrt(2.0)); }
static double npdf(double z, double s) { return std::exp(-0.5 * z * z / (s * s)) / (s * std::sqrt(2.0 * PI)); }

// ---- noeuds de quadrature 1D sur [-L, L], resserres pres de x = +-a ----
static void add_gl5(std::vector<double>& x, std::vector<double>& w, double lo, double hi)
{
    static const double xg[5] = { 0.0, 0.5384693101056831, -0.5384693101056831, 0.9061798459386640, -0.9061798459386640 };
    static const double wg[5] = { 0.5688888888888889, 0.4786286704993665, 0.4786286704993665, 0.2369268850561891, 0.2369268850561891 };
    const double c = 0.5 * (lo + hi), h = 0.5 * (hi - lo);
    for (int i = 0; i < 5; ++i) { x.push_back(c + h * xg[i]); w.push_back(h * wg[i]); }
}

// panneaux de [0, L] : largeur w0 au bord x = a, qui croit d'un facteur 1.5, plafonnee a wmax
static void build_axis(double a, double L, std::vector<double>& x, std::vector<double>& w)
{
    const double w0 = 1.0e-4, ratio = 1.5, wmax = 0.015;
    std::vector<double> xp, wp;
    // [0, a] : on part de a et on descend
    {
        double hi = a, wid = w0;
        while (hi > 0.0) {
            double lo = hi - wid;
            if (lo < 0.0) lo = 0.0;
            add_gl5(xp, wp, lo, hi);
            hi = lo;
            wid = std::min(wid * ratio, wmax);
        }
    }
    // [a, L] : on part de a et on monte
    {
        double lo = a, wid = w0;
        while (lo < L) {
            double hi = lo + wid;
            if (hi > L) hi = L;
            add_gl5(xp, wp, lo, hi);
            lo = hi;
            wid = std::min(wid * ratio, wmax);
        }
    }
    // x = moitie positive, puis son image en miroir (indice i + np)
    const size_t np = xp.size();
    x.assign(2 * np, 0.0);
    w.assign(2 * np, 0.0);
    for (size_t i = 0; i < np; ++i) {
        x[i] = xp[i];       w[i] = wp[i];
        x[i + np] = -xp[i]; w[i + np] = wp[i];
    }
}

struct Combo { int cs, pad, k; double u, v, d; };

int main(int argc, char* argv[])
{
    const int R = (argc >= 2) ? atoi(argv[1]) : 5;

    DetectorConfig cfg;
    PadPlaneParams pp;
    TableOptions   opt;
    make_case_A(cfg, pp);
    opt.R = R;

    CheckList     chk;
    DerivedParams D;
    PsiTable*     table = 0;
    try {
        D = derive_params(cfg, pp, opt, chk);
        chk.finish();
        table = new PsiTable(D.tp);
        check_table(*table, D, chk, opt.acc_tol);
        chk.finish();
    } catch (const std::exception& e) {
        std::cerr << "\nArret : " << e.what() << "\n";
        return 1;
    }
    const PsiTable& T = *table;
    const int nt = T.n_tau();
    const double p = D.pitch, a = D.a, g = cfg.dga;
    const double DTkg = cfg.DT_kg, DTga = cfg.DT_ga;

    // ---- geometrie ----
    const int NPAD = 4;
    const double px[NPAD] = { 0.0, p, 0.5 * p, 0.0 };
    const double py[NPAD] = { 0.0, 0.0, p, -p };
    const double step_target = 0.01;                          // 0.1 mm
    const int N = (int)std::floor(p / step_target + 0.5);
    const double dx = p / N;
    const double d_mid = 10.0;
    const double t_ref = d_mid / cfg.ve_kg;                    // temps de reference (ns) : t_rel = t - t_ref

    std::vector<double> vx(N);
    for (int k = 0; k < N; ++k) vx[k] = (k + 0.5) * dx;
    std::vector<double> vd[2];
    vd[0].assign(N, d_mid);
    vd[1].assign(N, 0.0);
    for (int k = 0; k < N; ++k) vd[1][k] = d_mid + (vx[k] - 0.5 * p);

    // ---- meta ----
    {
        FILE* m = fopen("out/track_meta.txt", "w");
        if (!m) { std::cerr << "impossible d'ecrire dans out/ (lancer depuis tests/)\n"; return 1; }
        fprintf(m, "R=%d\na=%.12g\ng=%.12g\npitch=%.12g\n", R, a, g, p);
        fprintf(m, "DT_kg=%.12g\nDT_ga=%.12g\nDL_kg=%.12g\nDL_ga=%.12g\nve_kg=%.12g\nve_ga=%.12g\n",
                DTkg, DTga, cfg.DL_kg, cfg.DL_ga, cfg.ve_kg, cfg.ve_ga);
        fprintf(m, "delta_t=%.12g\ndtau=%.12g\nn_vox=%d\nstep=%.12g\nd_mid=%.12g\nt_ref=%.12g\n",
                cfg.delta_t, T.dtau(), N, dx, d_mid, t_ref);
        for (int ip = 0; ip < NPAD; ++ip) fprintf(m, "pad%d_x=%.12g\npad%d_y=%.12g\n", ip, px[ip], ip, py[ip]);
        fprintf(m, "table_mb=%.6g\nn_img=%d\n", D.table_mb, D.tp.n_img);
        fclose(m);
    }

    // ---- 1. simu : somme de Psi (table) sur les voxels ----
    const double t_lo = -60.0, t_hi = 250.0, t_step = 1.0;
    const int nT = (int)std::floor((t_hi - t_lo) / t_step + 0.5) + 1;
    {
        FILE* f = fopen("out/track_simu.csv", "w");
        fprintf(f, "case,pad,t_rel,Q\n");
        std::vector<double> vec(nt), Q(nT);
        for (int cs = 0; cs < 2; ++cs)
            for (int ip = 0; ip < NPAD; ++ip) {
                for (int it = 0; it < nT; ++it) Q[it] = 0.0;
                for (int k = 0; k < N; ++k) {
                    const double u = vx[k] - px[ip], v = 0.0 - py[ip], d = vd[cs][k];
                    T.psi_vector(u, v, d, &vec[0]);
                    const double tg_rel = d / cfg.ve_kg - t_ref;       // arrivee du voxel a la grille
                    for (int it = 0; it < nT; ++it) {
                        const double t = t_lo + it * t_step;
                        Q[it] += T.psi_at(&vec[0], t - tg_rel) / N;
                    }
                }
                for (int it = 0; it < nT; ++it)
                    fprintf(f, "%d,%d,%.6g,%.12g\n", cs + 1, ip, t_lo + it * t_step, Q[it]);
            }
        fclose(f);
    }

    // ---- 2. reference : partie collectee Lc(h) pour chaque (cas, pad, voxel) ----
    std::vector<Combo> cb;
    for (int cs = 0; cs < 2; ++cs)
        for (int ip = 0; ip < NPAD; ++ip)
            for (int k = 0; k < N; ++k) {
                Combo c;
                c.cs = cs + 1; c.pad = ip; c.k = k;
                c.u = vx[k] - px[ip]; c.v = 0.0 - py[ip]; c.d = vd[cs][k];
                cb.push_back(c);
            }

    const double s_max = DTga * std::sqrt(g);
    const double L = a + 7.0 * s_max;
    std::vector<double> x, w;
    build_axis(a, L, x, w);
    const int nx = (int)x.size(), np = nx / 2;
    std::cout << "quadrature : " << nx << " noeuds par axe, domaine +-" << L << " cm\n";

    const int NH = 40;
    std::vector<double> phi((size_t)nx * nx), wx(nx), wy(nx);
    FILE* fl = fopen("out/track_lc.csv", "w");
    fprintf(fl, "case,pad,k,u,v,d,h,Lc\n");
    for (int j = 0; j <= NH; ++j) {
        const double h = g * (double)j * j / ((double)NH * NH);       // noeuds uniformes en sqrt(h)
        if (j >= 1 && j < NH) {
            for (int i = 0; i < np; ++i)
                for (int jj = 0; jj < np; ++jj) {
                    const double val = phi_point(x[i], x[jj], h, a, g, D.tp.n_img);
                    phi[(size_t)i * nx + jj] = val;
                    phi[(size_t)(i + np) * nx + jj] = val;
                    phi[(size_t)i * nx + jj + np] = val;
                    phi[(size_t)(i + np) * nx + jj + np] = val;
                }
        }
        for (size_t ic = 0; ic < cb.size(); ++ic) {
            const Combo& c = cb[ic];
            double Lc = 0.0;
            if (j == 0) {                                   // anode : formule en erf (plateau)
                const double s0 = std::sqrt(DTkg * DTkg * c.d + DTga * DTga * g);
                Lc = (Phi((a - c.u) / s0) - Phi((-a - c.u) / s0)) * (Phi((a - c.v) / s0) - Phi((-a - c.v) / s0));
            } else if (j < NH) {
                const double sig = std::sqrt(DTkg * DTkg * c.d + DTga * DTga * (g - h));
                const double s   = DTga * std::sqrt(h);
                for (int i = 0; i < nx; ++i) {
                    wx[i] = w[i] * npdf(x[i] - c.u, sig) * (Phi((a - x[i]) / s) - Phi((-a - x[i]) / s));
                    wy[i] = w[i] * npdf(x[i] - c.v, sig) * (Phi((a - x[i]) / s) - Phi((-a - x[i]) / s));
                }
                for (int i = 0; i < nx; ++i) {
                    double row = 0.0;
                    const double* ph = &phi[(size_t)i * nx];
                    for (int jj = 0; jj < nx; ++jj) row += ph[jj] * wy[jj];
                    Lc += wx[i] * row;
                }
            }
            fprintf(fl, "%d,%d,%d,%.10g,%.10g,%.10g,%.10g,%.12g\n", c.cs, c.pad, c.k, c.u, c.v, c.d, h, Lc);
        }
    }
    fclose(fl);
    std::cout << "ecrit : out/track_simu.csv  out/track_lc.csv  out/track_meta.txt\n";
    delete table;
    return 0;
}
