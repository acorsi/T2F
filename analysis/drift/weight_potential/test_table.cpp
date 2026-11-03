// Tests de PsiTable : exactitude aux noeuds, erreur d'interpolation contre un calcul
// direct (hors noeuds en u, v, tau et d), règle de somme, coût d'un appel. Unités : cm, ns.
#include "psi_table.hpp"
#include "psi_builder.hpp"
#include "detector.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>

static int n_fail = 0;

static void check(const char* nom, double calcule, double attendu, double tol) {
    double ecart = std::fabs(calcule - attendu);
    const char* etat = (ecart <= tol) ? "OK " : "ECHEC";
    if (ecart > tol) n_fail++;
    printf("[%s] %-46s calcule = %.6f  attendu = %.6f  ecart = %.2e\n",
           etat, nom, calcule, attendu, ecart);
}

static double alea() { return (double)rand() / ((double)RAND_MAX + 1.0); }

static double gcdf(double t) { return 0.5 * (1.0 + erf(t / sqrt(2.0))); }

int main() {
    const double PITCH = 0.315;
    DetectorConfig cfg;                          // valeurs par défaut de detector.hpp
    TableParams tp = default_table_params(cfg, PITCH);

    printf("=== Construction de la table ===\n");
    clock_t t0 = clock();
    PsiTable table(tp);
    printf("   %.2f s, %.1f Mo, tau de %.1f a %.1f ns (%d points, pas %.2f ns), u_max = %.2f mm\n",
           (double)(clock() - t0) / CLOCKS_PER_SEC, table.memory_bytes() / 1.0e6,
           table.tau_min(), table.tau_max(), table.n_tau(), table.dtau(), table.u_max() * 10.0);
    const int NT = table.n_tau();
    std::vector<double> vec(NT);

    // ------------------------------------------------------------------
    printf("\n=== 1. Exactitude aux noeuds ===\n");
    {
        // en un noeud (u, v, d, tau), psi_vector puis psi_at doivent redonner le calcul direct
        // (ici : plateau collecté, connu exactement). d = d_max : noeud en d.
        double d = tp.d_max;
        double sig = sqrt(table.sigma_T(d) * table.sigma_T(d) + tp.DT_ga * tp.DT_ga * tp.g);
        double a = tp.a;
        double u = 5.0 * tp.step_out, v = 11.0 * tp.step_out;   // noeuds
        double fx = gcdf((a - u) / sig) - gcdf((-a - u) / sig);
        double fy = gcdf((a - v) / sig) - gcdf((-a - v) / sig);
        table.psi_vector(u, v, d, &vec[0]);
        check("plateau au noeud (d_max, u=5, v=11)", table.psi_at(&vec[0], table.tau_max()), fx * fy, 2.0e-6);
        // symétrie
        std::vector<double> vec2(NT);
        table.psi_vector(-u, -v, d, &vec2[0]);
        check("symetrie (u,v) -> (-u,-v)", vec2[NT / 2], vec[NT / 2], 1.0e-14);
    }

    // ------------------------------------------------------------------
    printf("\n=== 2. Erreur d'interpolation contre calcul direct (grille 4x plus fine) ===\n");
    {
        // Référence : mêmes formules, mais pas en u,v divisé par 4 et pas en tau divisé par 4
        // (donc 4 fois plus de niveaux de hauteur : n_h = 400).
        const int REF = 4;
        const int NOUT_REF = 100;                        // 25 pas de la table = 4,9 mm
        PhiSource src_ref(tp.a, tp.g, tp.n_img, tp.m_cells, tp.r_src, tp.n_h * REF);
        double dtau_ref = table.dtau() / REF;
        int m_min_ref = table.m_min_index() * REF;
        int m_max_ref = table.m_max_index() * REF;
        int nt_ref = m_max_ref - m_min_ref + 1;

        double d_test[8] = {0.0, 0.05, 0.3, 1.7, 5.2, 12.9, 19.4, 20.0};
        srand(12345);
        for (int q = 0; q < 8; ++q) {
            double d = d_test[q];
            PsiParams P;
            P.sigma0 = table.sigma_T(d);  P.DT_ga = tp.DT_ga;  P.sigma_t = table.sigma_t(d);
            P.dtau = dtau_ref;  P.ve_ga = tp.ve_ga;
            P.m_min = m_min_ref;  P.m_max = m_max_ref;
            P.n_out = NOUT_REF;  P.step_out = tp.step_out / REF;
            std::vector<double> ref;
            build_psi(src_ref, P, ref);

            double err_d = 0.0, err_all = 0.0;
            // (a) uniquement l'erreur en d : (u,v,tau) sur des noeuds de la table
            for (int essai = 0; essai < 300; ++essai) {
                int i = REF * (int)(alea() * (NOUT_REF / REF));
                int j = REF * (int)(alea() * (NOUT_REF / REF));
                int n = REF * (int)(alea() * (nt_ref / REF));
                table.psi_vector(i * P.step_out, j * P.step_out, d, &vec[0]);
                double tau = (m_min_ref + n) * dtau_ref;
                double e = fabs(table.psi_at(&vec[0], tau) - ref[((size_t)i * NOUT_REF + j) * nt_ref + n]);
                if (e > err_d) err_d = e;
            }
            // (b) erreur totale : (u,v,tau) quelconques
            for (int essai = 0; essai < 600; ++essai) {
                int i = (int)(alea() * NOUT_REF);
                int j = (int)(alea() * NOUT_REF);
                int n = (int)(alea() * nt_ref);
                table.psi_vector(i * P.step_out, j * P.step_out, d, &vec[0]);
                double tau = (m_min_ref + n) * dtau_ref;
                double e = fabs(table.psi_at(&vec[0], tau) - ref[((size_t)i * NOUT_REF + j) * nt_ref + n]);
                if (e > err_all) err_all = e;
            }
            printf("   d = %5.2f cm (sigma_T = %.2f mm) : erreur max = %.2e (en d seul : %.2e)\n",
                   d, table.sigma_T(d) * 10.0, err_all, err_d);
            char nom[64];
            snprintf(nom, sizeof(nom), "erreur < 1e-3 pour d = %.2f cm", d);
            check(nom, err_all, 0.0, 1.0e-3);
        }
    }

    // ------------------------------------------------------------------
    printf("\n=== 3. Regle de somme via la table (voxel quelconque, pavage carre) ===\n");
    {
        srand(777);
        double pire = 0.0;
        for (int essai = 0; essai < 40; ++essai) {
            double d   = 20.0 * alea();
            double x0  = 0.5 * PITCH * alea();
            double y0  = 0.5 * PITCH * alea();
            double tau = table.tau_min() + (table.tau_max() - table.tau_min()) * alea();

            double somme = 0.0;
            for (int ip = -7; ip <= 7; ++ip) {
                for (int jp = -7; jp <= 7; ++jp) {
                    double u = x0 - ip * PITCH, v = y0 - jp * PITCH;
                    if (fabs(u) > table.u_max() || fabs(v) > table.u_max()) continue;
                    table.psi_vector(u, v, d, &vec[0]);
                    somme += table.psi_at(&vec[0], tau);
                }
            }
            // attendu : S(q) = clip(tau/T_gap) convolué par la gaussienne sigma_t(d)
            double st = table.sigma_t(d);
            double Tgap = tp.g / tp.ve_ga;
            double attendu = 0.0, pas = 0.05;
            for (double dl = -6.0 * st; dl <= 6.0 * st; dl += pas) {
                double q = (tau - dl) / Tgap;
                double S = (q <= 0.0) ? 0.0 : ((q >= 1.0) ? 1.0 : q);
                attendu += S * exp(-0.5 * dl * dl / (st * st)) / (st * sqrt(2.0 * M_PI)) * pas;
            }
            if (fabs(somme - attendu) > pire) pire = fabs(somme - attendu);
        }
        // Le déficit vient uniquement des pads au-delà de u_max (3,9 pitchs), dont la charge
        // induite n'est pas dans la table : mesuré à 3e-5 avec une table de 6 pitchs.
        // Il est le plus grand (~3e-3) quand le voxel est près d'un pad et que le pavage
        // perd une rangée à 4 pitchs. Par pad, la contribution perdue reste < 1e-4.
        printf("   ecart maximal sur 40 tirages : %.2e\n", pire);
        check("regle de somme < 1e-4 (portee de la table)", pire, 0.0, 1.0e-4);
    }

    // ------------------------------------------------------------------
    printf("\n=== 4. Cout d'un appel ===\n");
    {
        srand(99);
        const int NAPPELS = 20000;
        double garde = 0.0;
        clock_t t1 = clock();
        for (int c = 0; c < NAPPELS; ++c) {
            table.psi_vector(0.8 * alea(), 0.8 * alea(), 20.0 * alea(), &vec[0]);
            garde += vec[NT / 2];
        }
        double dt = (double)(clock() - t1) / CLOCKS_PER_SEC;
        printf("   psi_vector : %.2f microsecondes par appel (%d valeurs en tau) [%.3f]\n",
               dt / NAPPELS * 1.0e6, NT, garde);
    }

    printf("\n%s (%d echec(s))\n", n_fail == 0 ? "TOUS LES TESTS PASSENT" : "ECHECS", n_fail);
    return n_fail == 0 ? 0 : 1;
}
