// test_nh.cpp : qualité de l'interpolation en tau selon n_h (pas en tau = g / (ve_ga * n_h)).
//
// Construit la table de référence (n_h = 100), puis des tables n_h = 200, 50, 25, 20.
// Pour des tirages aléatoires de (u, v, d, tau), compare Psi (psi_vector + psi_at)
// à la référence, ainsi que la charge d'un bin de 10 ns : Psi(tau+10) - Psi(tau).
// Les erreurs sont absolues, par électron du nuage. Multiplier par n_e pour des électrons.
//
// Usage : test_nh [N_par_groupe] [n_e]     (défauts : 12000 et 1000)

#include "psi_table.hpp"
#include "detector.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>

struct Sample {
    double u, v, d, tau;
    int group;
};

static double unif(double a, double b) {
    return a + (b - a) * (rand() / (double)RAND_MAX);
}

static const int N_D = 4;
static const int N_T = 4;
static const double D_EDGES[N_D + 1]   = { 0.0, 0.5, 2.0, 8.0, 20.0 };
static const double TAU_EDGES[N_T + 1] = { -100.0, 40.0, 200.0, 330.0, 385.0 };

static PsiTable* make_table(const DetectorConfig& cfg, double pitch, int n_h) {
    TableParams tp = default_table_params(cfg, pitch);
    tp.n_h = n_h;
    return new PsiTable(tp);
}

static void evaluate(const PsiTable& T, const std::vector<Sample>& S,
                     std::vector<double>& psi, std::vector<double>& dq) {
    std::vector<double> vec(T.n_tau());
    psi.resize(S.size());
    dq.resize(S.size());
    for (size_t i = 0; i < S.size(); ++i) {
        T.psi_vector(S[i].u, S[i].v, S[i].d, &vec[0]);
        double p0 = T.psi_at(&vec[0], S[i].tau);
        double p1 = T.psi_at(&vec[0], S[i].tau + 10.0);
        psi[i] = p0;
        dq[i]  = p1 - p0;
    }
}

struct Stat {
    long   n;
    double ref_max;                 // max |Psi| de la référence
    double e_max, e_sum2;           // erreur sur Psi
    double q_max, q_sum2;           // erreur sur la charge d'un bin de 10 ns
    Stat() : n(0), ref_max(0), e_max(0), e_sum2(0), q_max(0), q_sum2(0) {}
};

static void compare(const char* nom, const std::vector<Sample>& S,
                    const std::vector<double>& psi_r, const std::vector<double>& dq_r,
                    const std::vector<double>& psi_t, const std::vector<double>& dq_t,
                    double n_e)
{
    Stat st[N_D * N_T];
    for (size_t i = 0; i < S.size(); ++i) {
        Stat& s = st[S[i].group];
        double e = psi_t[i] - psi_r[i];
        double q = dq_t[i]  - dq_r[i];
        s.n++;
        if (std::fabs(psi_r[i]) > s.ref_max) s.ref_max = std::fabs(psi_r[i]);
        if (std::fabs(e) > s.e_max) s.e_max = std::fabs(e);
        if (std::fabs(q) > s.q_max) s.q_max = std::fabs(q);
        s.e_sum2 += e * e;
        s.q_sum2 += q * q;
    }
    printf("\n=== %s ===\n", nom);
    printf("  d (cm)       tau (ns)        max|Psi|   Psi: max      rms     bin10: max      rms\n");
    double gmax_q = 0.0, gmax_e = 0.0;
    for (int id = 0; id < N_D; ++id) {
        for (int it = 0; it < N_T; ++it) {
            const Stat& s = st[id * N_T + it];
            if (s.n == 0) continue;
            printf("  %4.1f-%4.1f  %6.0f..%4.0f   %9.3e  %9.2e %9.2e   %9.2e %9.2e\n",
                   D_EDGES[id], D_EDGES[id + 1], TAU_EDGES[it], TAU_EDGES[it + 1],
                   s.ref_max, s.e_max, sqrt(s.e_sum2 / s.n), s.q_max, sqrt(s.q_sum2 / s.n));
            if (s.e_max > gmax_e) gmax_e = s.e_max;
            if (s.q_max > gmax_q) gmax_q = s.q_max;
        }
    }
    printf("  -> erreur max sur Psi : %.2e   sur la charge d'un bin : %.2e\n", gmax_e, gmax_q);
    printf("  -> pour n_e = %.0f : %.3f e- (Psi), %.3f e- (bin de 10 ns)\n",
           n_e, gmax_e * n_e, gmax_q * n_e);
}

int main(int argc, char* argv[]) {
    int    n_par = (argc > 1) ? atoi(argv[1]) : 12000;
    double n_e   = (argc > 2) ? atof(argv[2]) : 1000.0;

    DetectorConfig cfg;
    const double pitch = 0.3 + 0.015;          // pad_size + inter_pad
    const double reach = 3.0 * pitch;          // zone où le signal est notable

    // tirages : n_par par groupe (d, tau)
    srand(12345);
    std::vector<Sample> S;
    for (int id = 0; id < N_D; ++id) {
        for (int it = 0; it < N_T; ++it) {
            for (int k = 0; k < n_par; ++k) {
                Sample s;
                s.u   = unif(0.0, reach);
                s.v   = unif(0.0, reach);
                s.d   = unif(D_EDGES[id], D_EDGES[id + 1]);
                s.tau = unif(TAU_EDGES[it], TAU_EDGES[it + 1]);
                s.group = id * N_T + it;
                S.push_back(s);
            }
        }
    }
    printf("%d tirages, reach = %.2f cm, n_e = %.0f\n", (int)S.size(), reach, n_e);

    // référence n_h = 100
    clock_t c0 = clock();
    PsiTable* ref = make_table(cfg, pitch, 100);
    printf("table n_h=100 : %.1f s, %.1f Mo, n_tau=%d, dtau=%.2f ns\n",
           (double)(clock() - c0) / CLOCKS_PER_SEC, ref->memory_bytes() / 1.0e6,
           ref->n_tau(), ref->dtau());
    std::vector<double> psi_r, dq_r;
    evaluate(*ref, S, psi_r, dq_r);

    const int NH[4] = { 200, 50, 25, 20 };
    for (int m = 0; m < 4; ++m) {
        c0 = clock();
        PsiTable* T = make_table(cfg, pitch, NH[m]);
        printf("table n_h=%d : %.1f s, %.1f Mo, n_tau=%d, dtau=%.2f ns\n",
               NH[m], (double)(clock() - c0) / CLOCKS_PER_SEC, T->memory_bytes() / 1.0e6,
               T->n_tau(), T->dtau());
        std::vector<double> psi_t, dq_t;
        evaluate(*T, S, psi_t, dq_t);
        char nom[64];
        snprintf(nom, sizeof(nom), "n_h = %d  contre reference n_h = 100", NH[m]);
        compare(nom, S, psi_r, dq_r, psi_t, dq_t, n_e);
        delete T;
    }
    delete ref;
    return 0;
}
