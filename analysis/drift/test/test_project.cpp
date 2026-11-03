// Tests de PadPlane::project (induction par la table). Unités : cm, ns, électrons.
#include "pad_geometry.hpp"
#include "detector.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <thread>

static int n_fail = 0;
static void check(const char* nom, double calcule, double attendu, double tol) {
    double ecart = std::fabs(calcule - attendu);
    bool ok = ecart <= tol;
    if (!ok) n_fail++;
    printf("[%s] %-52s calcule = %.6g  attendu = %.6g  ecart = %.2e\n",
           ok ? "OK " : "ECHEC", nom, calcule, attendu, ecart);
}

static double gcdf(double t) { return 0.5 * (1.0 + erf(t / sqrt(2.0))); }

// charge cumulée d'un pad à la fin du bin k (les bins non gardés ont qt = 0 dans ce test)
static double Q_at(const PadCharge& pc, int k, double dt) {
    double Q = 0.0;
    for (size_t m = 0; m < pc.t.size(); ++m) {
        if ((int)std::floor(pc.t[m] / dt + 0.5) <= k) Q = pc.Q[m]; else break;
    }
    return Q;
}

static std::vector<PadCharge> un_voxel(const PadPlane& plane, const DetectorConfig& cfg,
                                       double x, double y, double d, double N,
                                       std::vector<std::vector<double> >& buf) {
    std::vector<double> X(1, x), Y(1, y), Z(1, cfg.dkg - d), NE(1, N);
    std::vector<double> sT(1, cfg.sigma_T(cfg.dkg - d)), sL(1, cfg.sigma_L(cfg.dkg - d));
    return plane.project(X, Y, Z, NE, sT, sL, cfg, buf);
}

int main() {
    DetectorConfig cfg;
    cfg.seuil_qt_pos = 0.0;      // garde tout bin non nul (positif ou négatif)
    cfg.seuil_qt_neg = 0.0;
    PadPlaneParams pp;
    const double PITCH = pp.pad_size + pp.inter_pad;

    clock_t c0 = clock();
    PadPlane plane(pp, cfg);
    printf("construction : %.2f s, n_bins = %d, pads = %d\n",
           (double)(clock() - c0) / CLOCKS_PER_SEC, cfg.n_bins(), (int)plane.pads().size());
    const int NB = cfg.n_bins();
    std::vector<std::vector<double> > buf(plane.pads().size(), std::vector<double>(NB, 0.0));
    const double N = 1.0e5;
    const double Tgap = cfg.T_gap();

    printf("\n=== 1. Conservation de la charge et regle de somme (1 voxel) ===\n");
    double pos[10][3] = {
        {0.0, 0.4725, 3.0}, {0.0, 0.0, 10.0}, {0.2, 0.9, 19.0}, {4.1, 3.3, 10.0},
        {-5.5, -6.1, 3.0}, {7.9, 1.2, 10.0}, {8.15, 0.3, 10.0}, {-6.0, 5.8, 15.0},
        {9.0, 2.0, 10.0}, {0.0, 8.0, 5.0}};
    for (int q = 0; q < 10; ++q) {
        double x = pos[q][0], y = pos[q][1], d = pos[q][2];
        std::vector<PadCharge> res = un_voxel(plane, cfg, x, y, d, N, buf);
        double tot = 0.0;
        for (size_t m = 0; m < res.size(); ++m) tot += res[m].q;
        // charge perdue au-delà de r_active : fraction de la gaussienne hors du disque (approx.)
        double r = std::sqrt(x * x + y * y), sig = cfg.sigma_T(cfg.dkg - d);
        double perdu = 1.0 - gcdf((pp.r_active - r) / sig);
        char nom[96];
        snprintf(nom, sizeof(nom), "charge finale/N, voxel (%.2f,%.2f) d=%g", x, y, d);
        check(nom, tot / N, 1.0 - perdu, 2.0e-3);

        // somme des Q(t) à mi-gap : doit valoir S(tau). Près du bord actif, une partie de
        // la charge induite tombe au-delà de r_active (pas de pad) : contrôle sauté.
        if (r > pp.r_active - 1.5) {
            printf("   (voxel (%.2f,%.2f) a moins de 1,5 cm du bord actif : somme a mi-gap non testee)\n", x, y);
            continue;
        }
        double t_g = cfg.t_grid(cfg.dkg - d);
        int kmid = (int)((t_g + 0.5 * Tgap) / cfg.delta_t);
        double somme = 0.0;
        for (size_t m = 0; m < res.size(); ++m) somme += Q_at(res[m], kmid, cfg.delta_t);
        double tau = (kmid + 1) * cfg.delta_t - t_g;            // fin du bin
        double st = plane.table().sigma_t(d), attendu = 0.0, pas = 0.05;
        for (double dl = -6.0 * st; dl <= 6.0 * st; dl += pas) {
            double qq = (tau - dl) / Tgap;
            double S = (qq <= 0.0) ? 0.0 : ((qq >= 1.0) ? 1.0 : qq);
            attendu += S * exp(-0.5 * dl * dl / (st * st)) / (st * sqrt(2.0 * M_PI)) * pas;
        }
        snprintf(nom, sizeof(nom), "somme Q(t)/N a mi-gap, voxel (%.2f,%.2f) d=%g", x, y, d);
        check(nom, somme / N, attendu * (1.0 - perdu), 1.5e-3);
    }

    printf("\n=== 2. Plateau du pad central = fx * fy ===\n");
    {
        double d = 10.0, x = 0.0, y = 0.4725;      // centre d'un pad de la zone quinconce
        std::vector<PadCharge> res = un_voxel(plane, cfg, x, y, d, N, buf);
        double sig = std::sqrt(plane.table().sigma_T(d) * plane.table().sigma_T(d)
                               + cfg.DT_ga * cfg.DT_ga * cfg.dga);
        double a = 0.5 * PITCH;
        double f = gcdf(a / sig) - gcdf(-a / sig);
        double qc = 0.0;
        for (size_t m = 0; m < res.size(); ++m)
            if (std::fabs(res[m].x - x) < 1e-9 && std::fabs(res[m].y - y) < 1e-9) qc = res[m].q;
        check("pad central : q / N", qc / N, f * f, 2.0e-4);
    }

    printf("\n=== 3. Seuils signes ===\n");
    {
        double d = 5.0, x = 0.0, y = 0.4725;
        // lecture de tout : référence
        std::vector<PadCharge> ref = un_voxel(plane, cfg, x, y, d, 5.0e4, buf);
        DetectorConfig c2 = cfg;
        c2.seuil_qt_pos = 200.0;  c2.seuil_qt_neg = -200.0;
        std::vector<PadCharge> deux = un_voxel(plane, c2, x, y, d, 5.0e4, buf);
        c2.seuil_qt_pos = 1.0e300; c2.seuil_qt_neg = -20.0;
        std::vector<PadCharge> neg = un_voxel(plane, c2, x, y, d, 5.0e4, buf);
        int nb_ref = 0, nb_deux = 0, nb_neg = 0, mauvais = 0;
        double pire_q = 0.0;
        for (size_t m = 0; m < ref.size(); ++m) nb_ref += (int)ref[m].t.size();
        for (size_t m = 0; m < deux.size(); ++m) {
            nb_deux += (int)deux[m].t.size();
            for (size_t k = 0; k < deux[m].qt.size(); ++k)
                if (!(deux[m].qt[k] > 200.0 || deux[m].qt[k] < -200.0)) mauvais++;
            for (size_t r = 0; r < ref.size(); ++r)       // la charge finale ne depend pas du seuil
                if (ref[r].pad_id == deux[m].pad_id)
                    pire_q = std::max(pire_q, std::fabs(ref[r].q - deux[m].q));
        }
        for (size_t m = 0; m < neg.size(); ++m) {
            nb_neg += (int)neg[m].t.size();
            for (size_t k = 0; k < neg[m].qt.size(); ++k)
                if (!(neg[m].qt[k] < -20.0)) mauvais++;
        }
        printf("   bins gardes : tout = %d, +-200 = %d, lobe negatif < -20 seul = %d\n",
               nb_ref, nb_deux, nb_neg);
        check("aucun bin garde ne viole le seuil", mauvais, 0, 0);
        check("charge finale independante du seuil", pire_q, 0.0, 1.0e-6);
        check("lobe negatif seul : des bins existent", nb_neg > 0 ? 1.0 : 0.0, 1.0, 0.0);
        check("lobe negatif seul : moins de bins que tout", nb_neg < nb_ref ? 1.0 : 0.0, 1.0, 0.0);
    }

    printf("\n=== 4. Tampon remis a zero, repetabilite ===\n");
    {
        std::vector<PadCharge> r1 = un_voxel(plane, cfg, 1.0, 2.0, 8.0, N, buf);
        double reste = 0.0;
        for (size_t i = 0; i < buf.size(); ++i)
            for (size_t k = 0; k < buf[i].size(); ++k) reste += std::fabs(buf[i][k]);
        check("qt_buf nul apres project", reste, 0.0, 0.0);
        std::vector<PadCharge> r2 = un_voxel(plane, cfg, 1.0, 2.0, 8.0, N, buf);
        double ecart = 0.0;
        for (size_t m = 0; m < r1.size(); ++m) ecart += std::fabs(r1[m].q - r2[m].q);
        check("deux appels identiques", ecart, 0.0, 0.0);
    }

    printf("\n=== 5. Plusieurs threads (table partagee) ===\n");
    {
        const int NTH = 4;
        std::vector<double> sortie(NTH, 0.0);
        std::vector<std::thread> th;
        struct Job {
            static void run(const PadPlane* pl, const DetectorConfig* c, double* out, int id) {
                std::vector<std::vector<double> > b(pl->pads().size(),
                                                    std::vector<double>(c->n_bins(), 0.0));
                double s = 0.0;
                for (int rep = 0; rep < 20; ++rep) {
                    std::vector<double> X(1, 0.5), Y(1, 1.0), Z(1, 10.0), NE(1, 1.0e5);
                    std::vector<double> sT(1, c->sigma_T(10.0)), sL(1, 0.0);
                    std::vector<PadCharge> r = pl->project(X, Y, Z, NE, sT, sL, *c, b);
                    for (size_t m = 0; m < r.size(); ++m) s += r[m].q;
                }
                out[id] = s;
            }
        };
        for (int t = 0; t < NTH; ++t) th.push_back(std::thread(Job::run, &plane, &cfg, &sortie[0], t));
        for (int t = 0; t < NTH; ++t) th[t].join();
        double ecart = 0.0;
        for (int t = 1; t < NTH; ++t) ecart = std::max(ecart, std::fabs(sortie[t] - sortie[0]));
        check("resultats identiques sur 4 threads", ecart, 0.0, 0.0);
    }

    printf("\n=== 6. Temps de calcul : trace de 160 voxels ===\n");
    {
        std::vector<double> X, Y, Z, NE, sT, sL;
        for (int i = 0; i < 160; ++i) {          // y de 0 à 8 cm : reste dans le plan actif
            X.push_back(0.3);  Y.push_back(0.05 * i);  Z.push_back(10.0 + 0.02 * i);
            NE.push_back(4000.0);
            sT.push_back(cfg.sigma_T(Z.back()));  sL.push_back(cfg.sigma_L(Z.back()));
        }
        clock_t t1 = clock();
        std::vector<PadCharge> r = plane.project(X, Y, Z, NE, sT, sL, cfg, buf);
        double dt = (double)(clock() - t1) / CLOCKS_PER_SEC;
        printf("   %.1f ms pour 160 voxels (%d pads lus, seuils = 0)\n", dt * 1000.0, (int)r.size());
        double tot = 0.0;
        for (size_t m = 0; m < r.size(); ++m) tot += r[m].q;
        check("trace : charge finale / charge injectee", tot / (160 * 4000.0), 1.0, 2.0e-3);
    }

    printf("\n%s (%d echec(s))\n", n_fail == 0 ? "TOUS LES TESTS PASSENT" : "ECHECS", n_fail);
    return n_fail == 0 ? 0 : 1;
}
