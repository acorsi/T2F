// ---------------------------------------------------------------------------
// Niveau 3 : tirage du nombre d'electrons par voxel (Processor::process, branche ne > 30).
//
// Usage :  ./test_level3            (apres  python3 test_level3.py export input_1000.h5  pour la partie 4)
//
// On passe par Processor::process (public) avec une trace le long de x (theta = pi/2, phi = 0, axe Y),
// qui ne sort jamais de la grille. n_electrons est lu dans EventOutput.
//
//   1. ne = 31, 100, 1000, 25000 : moyenne = ne et variance = Fano*ne + 1/12 (le 1/12 vient de l'arrondi),
//      a 4 ecarts-types statistiques ; n entier et >= 0
//   2. dE = 0 : n = 0
//   3. reproductibilite : meme graine, meme resultat (constructeur et reseed)
//   4. donnees reelles (out/l3_real.txt, voxels deja fusionnes) : somme(n) contre somme(dE/W) par trace,
//      ecart / sqrt(Fano*somme) de moyenne 0 et d'ecart-type 1
// C++17 (comme processor.hpp).
// ---------------------------------------------------------------------------
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include "processor.hpp"

static int nfail = 0;
static void check(bool ok, const char* msg) {
    if (!ok) { ++nfail; printf("   ECHEC : %s\n", msg); }
}

static Direction dir_x() { Direction d; d.theta = M_PI / 2.0; d.phi = 0.0; return d; }

static void run(Processor& P, const std::vector<double>& dE, std::vector<double>& n)
{
    EventInput in;
    in.H.raw_trace = dE; in.H.Z = 0; in.H.A = 0; in.H.KE = 0.0;
    in.L.raw_trace = std::vector<double>(1, 0.0); in.L.Z = 0; in.L.A = 0; in.L.KE = 0.0;
    EventOutput out;
    P.process(in, out, dir_x(), dir_x());
    n = out.H.n_electrons;
}

int main()
{
    DetectorConfig cfg;
    cfg.step_size = 0.048;
    const double W = cfg.W, Fano = cfg.Fano;
    const long N = 400000;

    printf("1. moyenne et variance, %ld voxels par valeur de ne\n", N);
    printf("   %-8s %-12s %-10s %-10s %-12s %-10s %-10s\n", "ne", "moyenne", "ecart/sig", "variance", "attendue", "ecart/sig", "sans 1/12");
    const double nes[4] = { 31.0, 100.0, 1000.0, 25000.0 };
    for (int q = 0; q < 4; ++q) {
        const double ne = nes[q];
        Processor P(cfg, 1000 + q);
        std::vector<double> n;
        run(P, std::vector<double>(N, ne * W), n);
        check((long)n.size() == N, "nombre de voxels");
        double s1 = 0, s2 = 0, s4 = 0;
        bool good = true;
        for (size_t i = 0; i < n.size(); ++i) {
            if (n[i] < 0.0 || n[i] != std::floor(n[i])) good = false;
            s1 += n[i];
        }
        const double mean = s1 / n.size();
        for (size_t i = 0; i < n.size(); ++i) { const double d = n[i] - mean; s2 += d * d; s4 += d * d * d * d; }
        const double var = s2 / n.size();
        const double m4 = s4 / n.size();
        const double err_mean = std::sqrt(var / n.size());
        const double err_var = std::sqrt((m4 - var * var) / n.size());
        const double var_th = Fano * ne + 1.0 / 12.0;
        const double zm = (mean - ne) / err_mean;
        const double zv = (var - var_th) / err_var;
        const double zv0 = (var - Fano * ne) / err_var;
        printf("   %-8.0f %-12.3f %-10.2f %-10.3f %-12.3f %-10.2f %-10.2f\n", ne, mean, zm, var, var_th, zv, zv0);
        check(good, "n doit etre entier et >= 0");
        check(std::fabs(zm) < 4.0, "moyenne differente de ne");
        check(std::fabs(zv) < 4.0, "variance differente de Fano*ne + 1/12");
    }
    printf("   (colonne 'sans 1/12' : meme ecart si on oublie l'arrondi ; il n'est pas negligeable a ne = 31)\n");

    printf("2. dE = 0 : n = 0\n");
    {
        Processor P(cfg, 7);
        std::vector<double> n;
        run(P, std::vector<double>(1000, 0.0), n);
        bool z = true;
        for (size_t i = 0; i < n.size(); ++i) if (n[i] != 0.0) z = false;
        check(z && n.size() == 1000, "n non nul pour dE = 0");
        printf("   %s\n", z ? "OK" : "ECHEC");
    }

    printf("3. reproductibilite\n");
    {
        std::vector<double> dE(2000, 25000.0 * W), a, b, c;
        Processor P1(cfg, 42), P2(cfg, 42), P3(cfg, 1);
        run(P1, dE, a); run(P2, dE, b);
        P3.reseed(42); run(P3, dE, c);
        check(a == b, "deux Processor de meme graine different");
        check(a == c, "reseed ne redonne pas le meme resultat");
        printf("   constructeur : %s ; reseed : %s\n", a == b ? "OK" : "ECHEC", a == c ? "OK" : "ECHEC");
    }

    printf("4. donnees reelles\n");
    {
        std::ifstream f("out/l3_real.txt");
        if (!f) {
            printf("   out/l3_real.txt absent : partie sautee (python3 test_level3.py export input_1000.h5)\n");
        } else {
            int nt = 0;
            f >> nt;
            std::vector<double> z;
            double zs = 0, zs2 = 0, worst = 0;
            Processor P(cfg, 2024);
            for (int t = 0; t < nt; ++t) {
                long m = 0;
                f >> m;
                std::vector<double> dE(m);
                double sne = 0;
                for (long i = 0; i < m; ++i) { f >> dE[i]; sne += dE[i] / W; }
                std::vector<double> n;
                run(P, dE, n);
                double sn = 0;
                for (size_t i = 0; i < n.size(); ++i) sn += n[i];
                check((long)n.size() == m, "nombre de voxels different de l'entree");
                const double zz = (sn - sne) / std::sqrt(Fano * sne);
                const double rel = (sn - sne) / sne;
                if (std::fabs(rel) > worst) worst = std::fabs(rel);
                zs += zz; zs2 += zz * zz;
            }
            const double mean = zs / nt, sd = std::sqrt(zs2 / nt - mean * mean);
            printf("   %d traces ; somme(n)/somme(dE/W) - 1 : max %.2e ; (somme n - somme ne)/sqrt(Fano somme ne) : moyenne %.3f (attendu 0 +- %.3f), ecart-type %.3f (attendu 1 +- %.3f)\n",
                   nt, worst, mean, 1.0 / std::sqrt((double)nt), sd, 1.0 / std::sqrt(2.0 * nt));
            check(std::fabs(mean) < 4.0 / std::sqrt((double)nt), "ecart moyen non nul");
            check(std::fabs(sd - 1.0) < 4.0 / std::sqrt(2.0 * nt), "ecart-type different de 1");
        }
    }

    printf("\nBILAN NIVEAU 3 : %s\n", nfail == 0 ? "tout est OK" : "ATTENTION : un critere echoue");
    return nfail == 0 ? 0 : 1;
}
