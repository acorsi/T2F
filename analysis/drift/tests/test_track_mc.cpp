// ---------------------------------------------------------------------------
// Controle independant de la partie "collectee" Lc(h) de la reference (test_track.cpp) par Monte-Carlo.
//
// Usage :  ./test_track_mc [N_tirages]      (apres ./test_track ; 2000000 par defaut)
//
// Pour quelques lignes de out/track_lc.csv (cas, pad, voxel, niveau h), on tire des electrons :
//   position a la hauteur h :  (u, v) + N(0, sigma(h)) par axe,   sigma(h)^2 = DT_kg^2 d + DT_ga^2 (g - h)
//   position d'arrivee      :  position + N(0, DT_ga sqrt(h)) par axe
// et on estime  Lc = E[ phi(position, h) * 1(arrivee dans la tuile |x|<a, |y|<a) ].
// Aucune formule d'erf pour la probabilite d'arrivee, aucune quadrature : c'est ce qui valide le calcul de la
// reference. Critere : ecart < 4 ecarts-types statistiques (et < 1e-3).
//
// C++11.
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include "weighting_potential.hpp"

int main(int argc, char* argv[])
{
    const long N = (argc >= 2) ? atol(argv[1]) : 2000000L;

    std::map<std::string, double> M;
    {
        std::ifstream fm("out/track_meta.txt");
        if (!fm) { std::cerr << "out/track_meta.txt introuvable : lancer ./test_track d'abord\n"; return 1; }
        std::string line;
        while (std::getline(fm, line)) {
            size_t eq = line.find('=');
            if (eq != std::string::npos) M[line.substr(0, eq)] = atof(line.substr(eq + 1).c_str());
        }
    }
    const double a = M["a"], g = M["g"], DTkg = M["DT_kg"], DTga = M["DT_ga"];
    const int n_img = (int)M["n_img"];

    struct Row { int cs, pad, k; double u, v, d, h, Lc; };
    std::vector<Row> rows;
    {
        std::ifstream fl("out/track_lc.csv");
        if (!fl) { std::cerr << "out/track_lc.csv introuvable\n"; return 1; }
        std::string line;
        std::getline(fl, line);                       // en-tete
        while (std::getline(fl, line)) {
            Row r;
            if (sscanf(line.c_str(), "%d,%d,%d,%lf,%lf,%lf,%lf,%lf", &r.cs, &r.pad, &r.k, &r.u, &r.v, &r.d, &r.h, &r.Lc) == 8)
                rows.push_back(r);
        }
    }

    // niveaux h : indices j = 8, 20, 30 de la grille h_j = g (j/40)^2 ; voxels 0, 13, 26, 51
    const int NH = 40;
    const int jj[3] = { 8, 20, 30 };
    const int kk[4] = { 0, 13, 26, 51 };
    std::mt19937 gen(12345);
    std::normal_distribution<double> nd(0.0, 1.0);

    int nbad = 0, ntest = 0;
    double worst = 0.0;
    printf("%-4s %-4s %-4s %-9s %-10s %-12s %-12s %-10s %-8s\n", "cas", "pad", "vox", "h (cm)", "u (cm)", "quadrature", "Monte-Carlo", "ecart", "ecart/sig");
    for (size_t i = 0; i < rows.size(); ++i) {
        const Row& r = rows[i];
        bool pick = false;
        for (int q = 0; q < 3; ++q) {
            const double hq = g * (double)jj[q] * jj[q] / ((double)NH * NH);
            if (std::fabs(r.h - hq) < 1.0e-9) pick = true;
        }
        bool pk = false;
        for (int q = 0; q < 4; ++q) if (r.k == kk[q]) pk = true;
        if (!pick || !pk || r.pad > 2 || r.cs != 1) continue;

        const double sig = std::sqrt(DTkg * DTkg * r.d + DTga * DTga * (g - r.h));
        const double s   = DTga * std::sqrt(r.h);
        double sum = 0.0, sum2 = 0.0;
        for (long n = 0; n < N; ++n) {
            const double x = r.u + sig * nd(gen), y = r.v + sig * nd(gen);
            const double xe = x + s * nd(gen), ye = y + s * nd(gen);
            double val = 0.0;
            if (std::fabs(xe) < a && std::fabs(ye) < a) val = phi_point(x, y, r.h, a, g, n_img);
            sum += val; sum2 += val * val;
        }
        const double mean = sum / N;
        const double err  = std::sqrt((sum2 / N - mean * mean) / N);
        const double diff = mean - r.Lc;
        const double nsig = (err > 0.0) ? std::fabs(diff) / err : 0.0;
        printf("%-4d P%-3d %-4d %-9.4f %-10.4f %-12.6f %-12.6f %-10.2e %-8.2f\n", r.cs, r.pad, r.k, r.h, r.u, r.Lc, mean, diff, nsig);
        ++ntest;
        if (nsig > 4.0 || std::fabs(diff) > 1.0e-3) ++nbad;
        if (std::fabs(diff) > worst) worst = std::fabs(diff);
    }
    printf("\n%d points compares, ecart max %.2e, %d hors critere\n", ntest, worst, nbad);
    printf("BILAN MONTE-CARLO : %s\n", nbad == 0 ? "tout est OK" : "ATTENTION");
    return nbad == 0 ? 0 : 1;
}
