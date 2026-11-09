// Tests de psi_builder (lissage latéral + convolution temporelle). Unités : cm, ns.
#include "psi_builder.hpp"
#include "weighting_potential.hpp"
#include <cstdio>
#include <cmath>
#include <ctime>
#include <vector>

static int n_fail = 0;

static void check(const char* nom, double calcule, double attendu, double tol) {
    double ecart = std::fabs(calcule - attendu);
    const char* etat = (ecart <= tol) ? "OK " : "ECHEC";
    if (ecart > tol) n_fail++;
    printf("[%s] %-40s calcule = %.6f  attendu = %.6f  ecart = %.2e\n",
           etat, nom, calcule, attendu, ecart);
}

// --- Paramètres communs ------------------------------------------------------
static const double PITCH  = 0.315;          // 3,15 mm
static const double A      = 0.5 * PITCH;    // demi-côté de la tuile
static const double G      = 0.5;            // gap
static const double VE_GA  = 0.002;          // cm/ns
static const double DTAU   = 2.5;            // ns
static const int    NH     = 100;            // G / NH = VE_GA * DTAU
static const int    NIMG   = 10;
static const int    MCELLS = 8;              // pas source = PITCH/16
static const double R_SRC  = 2.2;            // cm
static const int    NOUT   = 112;            // noeuds en u et v
static const double STEP_OUT = PITCH / 16.0; // = pas source : les décalages du pavage sont des noeuds

static double gcdf(double t) { return 0.5 * (1.0 + erf(t / sqrt(2.0))); }

// fraction exacte d'un nuage gaussien (sigma s), centré en (u,v) du centre du pad, tombant sur la tuile
static double collecte(double u, double v, double s) {
    double fx = gcdf((A - u) / s) - gcdf((-A - u) / s);
    double fy = gcdf((A - v) / s) - gcdf((-A - v) / s);
    return fx * fy;
}

static double psi_at(const std::vector<double>& psi, int n_tau, int i, int j, int n) {
    return psi[((size_t)i * NOUT + j) * n_tau + n];
}

static int run_psi(const PhiSource& src, double sigma0, double DT_ga, double sigma_t,
                   int m_min, int m_max, std::vector<double>& psi)
{
    PsiParams P;
    P.sigma0 = sigma0;  P.DT_ga = DT_ga;  P.sigma_t = sigma_t;
    P.dtau = DTAU;  P.ve_ga = VE_GA;
    P.m_min = m_min;  P.m_max = m_max;
    P.n_out = NOUT;  P.step_out = STEP_OUT;
    clock_t t0 = clock();
    build_psi(src, P, psi);
    printf("   (build_psi : %.2f s)\n", (double)(clock() - t0) / CLOCKS_PER_SEC);
    return psi_n_tau(P);
}

// Lissage de phi à (u,v,h) par quadrature directe, sur des cellules 8 fois plus petites
// que la source, avec phi_point évalué à chaque cellule : référence indépendante.
static double lisse_direct(double u, double v, double h, double sigma, int m_ref) {
    double step = A / m_ref;
    int jlo = (int)floor((u - 5.0 * sigma) / step), jhi = (int)floor((u + 5.0 * sigma) / step);
    int klo = (int)floor((v - 5.0 * sigma) / step), khi = (int)floor((v + 5.0 * sigma) / step);
    std::vector<double> wx, wy;
    for (int j = jlo; j <= jhi; ++j) wx.push_back(gcdf(((j + 1) * step - u) / sigma) - gcdf((j * step - u) / sigma));
    for (int k = klo; k <= khi; ++k) wy.push_back(gcdf(((k + 1) * step - v) / sigma) - gcdf((k * step - v) / sigma));
    double s = 0.0;
    for (int j = jlo; j <= jhi; ++j) {
        for (int k = klo; k <= khi; ++k) {
            double w = wx[j - jlo] * wy[k - klo];
            if (w < 1.0e-12) continue;
            s += w * phi_point((j + 0.5) * step, (k + 0.5) * step, h, A, G, NIMG);
        }
    }
    return s;
}

int main() {
    printf("=== Construction de la source phi ===\n");
    clock_t t0 = clock();
    PhiSource src(A, G, NIMG, MCELLS, R_SRC, NH);
    printf("   (PhiSource : %.2f s, %d cellules par axe, pas = %.4f cm)\n",
           (double)(clock() - t0) / CLOCKS_PER_SEC, src.n_cells(), src.step());

    std::vector<double> psi;
    int nt;

    // ------------------------------------------------------------------
    printf("\n=== 1. Plateau final = fraction collectee exacte (sigma constant) ===\n");
    {
        double sig[3] = {0.042, 0.1, 0.268};
        int pts[6][2] = {{0,0}, {16,0}, {8,16}, {24,16}, {0,32}, {16,16}};
        for (int s = 0; s < 3; ++s) {
            nt = run_psi(src, sig[s], 0.0, 0.0, 0, 160, psi);
            for (int p = 0; p < 6; ++p) {
                int i = pts[p][0], j = pts[p][1];
                char nom[64];
                snprintf(nom, sizeof(nom), "plateau sigma=%.3f (%d,%d)", sig[s], i, j);
                check(nom, psi_at(psi, nt, i, j, nt - 1),
                      collecte(i * STEP_OUT, j * STEP_OUT, sig[s]), 2.0e-6);
            }
        }
    }

    // ------------------------------------------------------------------
    printf("\n=== 2. Valeurs du resume (nuage au-dessus du centre, sigma constant) ===\n");
    {
        double sig[3] = {0.042, 0.1, 0.268};
        for (int s = 0; s < 3; ++s) {
            nt = run_psi(src, sig[s], 0.0, 0.0, 0, 160, psi);
            double fin_c = psi_at(psi, nt, 0, 0, nt - 1);
            double fin_v = psi_at(psi, nt, 16, 0, nt - 1);
            double max_v = 0.0;
            for (int n = 0; n < nt; ++n) {
                double v = psi_at(psi, nt, 16, 0, n);
                if (v > max_v) max_v = v;
            }
            printf("   sigma = %.2f mm : centre final = %.4f, voisin (p,0) final = %.4f, maximum transitoire = %.4f\n",
                   sig[s] * 10.0, fin_c, fin_v, max_v);
        }
        printf("   resume : 0,42 mm -> 0,997 ; 1,0 mm -> 0,780 / 0,051 / 0,076 ; 2,68 mm -> 0,196 / 0,106\n");
    }

    // ------------------------------------------------------------------
    printf("\n=== 3. Limite ponctuelle (sigma -> plancher) contre phi_point ===\n");
    {
        nt = run_psi(src, 0.0, 0.0, 0.0, 0, 160, psi);
        int taus[6] = {50, 100, 150, 200, 225, 240};
        for (int q = 0; q < 6; ++q) {
            double tau = taus[q];
            double h = G - VE_GA * tau;
            char nom[64];
            snprintf(nom, sizeof(nom), "centre, tau = %d ns (h = %.2f mm)", taus[q], h * 10.0);
            check(nom, psi_at(psi, nt, 0, 0, (int)(tau / DTAU)), phi_point(0, 0, h, A, G, NIMG), 3.0e-3);
        }
    }

    // ------------------------------------------------------------------
    printf("\n=== 4. Lissage : table contre quadrature directe (cellules 8x plus fines) ===\n");
    {
        // sigma(h)^2 = DT_ga^2 (g - h), pour d = 0
        nt = run_psi(src, 0.0, 0.06, 0.0, 0, 160, psi);
        int niveaux[4] = {2, 5, 20, 50};
        int pts[5][2] = {{0,0}, {8,0}, {16,0}, {8,16}, {8,8}};
        double pire = 0.0;
        for (int q = 0; q < 4; ++q) {
            int l = niveaux[q];
            double h = l * G / NH;
            double sigma = 0.06 * sqrt(G - h);
            if (sigma < 0.5 * src.step()) sigma = 0.5 * src.step();
            for (int p = 0; p < 5; ++p) {
                int i = pts[p][0], j = pts[p][1];
                double t = psi_at(psi, nt, i, j, NH - l);
                double d = lisse_direct(i * STEP_OUT, j * STEP_OUT, h, sigma, 64);
                double e = fabs(t - d);
                if (e > pire) pire = e;
            }
        }
        printf("   erreur maximale sur 20 points (h = 0,1 a 2,5 mm) : %.2e\n", pire);
        check("erreur de lissage < 1e-3", pire, 0.0, 1.0e-3);
    }

    // ------------------------------------------------------------------
    printf("\n=== 5. Regle de somme sur un pavage carre (niveaux isoles) ===\n");
    {
        nt = run_psi(src, 0.1, 0.06, 0.0, 0, 160, psi);
        int niveaux[3] = {20, 50, 80};
        for (int q = 0; q < 3; ++q) {
            int l = niveaux[q];
            double somme = 0.0;
            for (int ip = -5; ip <= 5; ++ip) {
                for (int jp = -5; jp <= 5; ++jp) {
                    int i = abs(3 - 16 * ip), j = abs(5 - 16 * jp);
                    somme += psi_at(psi, nt, i, j, NH - l);
                }
            }
            char nom[64];
            snprintf(nom, sizeof(nom), "somme des pads, h = %.1f mm", l * G / NH * 10.0);
            check(nom, somme, 1.0 - (double)l / NH, 1.0e-4);
        }
    }

    // ------------------------------------------------------------------
    printf("\n=== 6. Avec etalement temporel sigma_t = 28.5 ns (d = 20 cm) ===\n");
    {
        double sigma_t = 28.5;
        double sigma0 = 0.06 * sqrt(20.0);          // 2,68 mm
        int m_min = -40, m_max = 200;
        nt = run_psi(src, sigma0, 0.06, sigma_t, m_min, m_max, psi);

        // plateau = fx*fy avec sigma_anode
        double sig_anode = sqrt(sigma0 * sigma0 + 0.06 * 0.06 * G);
        check("plateau centre", psi_at(psi, nt, 0, 0, nt - 1), collecte(0, 0, sig_anode), 2.0e-6);
        check("plateau voisin (p,0)", psi_at(psi, nt, 16, 0, nt - 1), collecte(PITCH, 0, sig_anode), 2.0e-6);

        // valeur au bord inferieur de la plage en tau (-100 ns) et en tau = 0
        printf("   Psi(centre) a tau = -100 ns : %.2e ; a tau = 0 : %.2e\n",
               psi_at(psi, nt, 0, 0, 0), psi_at(psi, nt, 0, 0, -m_min));
        check("Psi(centre, tau = -100 ns) ~ 0", psi_at(psi, nt, 0, 0, 0), 0.0, 1.0e-4);

        // somme sur le pavage, au cours du temps : doit valoir la somme clippee de 1-h/g
        // convoluee par les memes poids temporels.
        std::vector<double> wt;
        int jmax = (int)ceil(5.0 * sigma_t / DTAU);
        for (int j = -jmax; j <= jmax; ++j)
            wt.push_back(gcdf((j + 0.5) * DTAU / sigma_t) - gcdf((j - 0.5) * DTAU / sigma_t));
        double pire = 0.0;
        for (int n = 0; n < nt; n += 7) {
            int m = m_min + n;
            double attendu = 0.0;
            for (int jj = -jmax; jj <= jmax; ++jj) {
                int q = m - jj;
                double S = (q <= 0) ? 0.0 : ((q >= NH) ? 1.0 : (double)q / NH);
                attendu += wt[jj + jmax] * S;
            }
            double somme = 0.0;
            for (int ip = -5; ip <= 5; ++ip)
                for (int jp = -5; jp <= 5; ++jp)
                    somme += psi_at(psi, nt, abs(3 - 16 * ip), abs(5 - 16 * jp), n);
            if (fabs(somme - attendu) > pire) pire = fabs(somme - attendu);
        }
        printf("   ecart maximal de la regle de somme au cours du temps : %.2e\n", pire);
        check("regle de somme en temps < 3e-4", pire, 0.0, 3.0e-4);
    }

    printf("\n%s (%d echec(s))\n", n_fail == 0 ? "TOUS LES TESTS PASSENT" : "ECHECS", n_fail);
    return n_fail == 0 ? 0 : 1;
}
