// ---------------------------------------------------------------------------
// Tests de coherence de la chaine de lecture (readout.hpp / readout.cpp) : fenetre ecrite, conservation
// de la charge, linearite signal / bruit. Complete test_readout.cpp (qui teste chaque brique) par des
// familles de cas systematiques. Sans HDF5.
//
// Usage :  ./test_chain
//
//   C1  fenetre ecrite sur une famille de signaux (k0 de 0 a la fin de la grille, nq de 1 a 30 bins,
//       plusieurs phases, avec et sans bruit) : bornes exactes [k0 - marge, k0 + nq - 1 + (n_taps - 1) + marge],
//       jamais tronquee en fin de grille ; au debut de la grille (k0 < marge) la fenetre est coupee a
//       l'echantillon 0 : LIMITE CONNUE, le test verifie que c'est bien ce comportement defini
//   C2  conservation : la somme de la sortie vraie = (somme de qt) x (somme de h) ; sur la grille de
//       lecture, sur les echantillons ecrits (adc_ratio = 1, exact), et avec adc_ratio = 2
//   C3  linearite : (avec bruit) - (sans bruit) ne depend pas de l'amplitude du signal (meme graine) ;
//       la sortie sans bruit est lineaire en signal ; une autre graine donne un autre bruit
//
// Gaussienne de test, pas de reponse MINOS. Code retour : 0 si tout passe. C++ classique.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <string>
#include <vector>

#include "noise.hpp"
#include "preamp_response.hpp"
#include "readout.hpp"

static int g_fail = 0, g_ok = 0;

static void check(bool cond, const std::string& what, double value, double expected)
{
    if (cond) { ++g_ok;   std::printf("  ok   %-72s %12.5g (attendu %.5g)\n", what.c_str(), value, expected); }
    else      { ++g_fail; std::printf("  ECHEC %-71s %12.5g (attendu %.5g)\n", what.c_str(), value, expected); }
}

static PadCharge make_pad(int id, int k0, const std::vector<double>& qt)
{
    PadCharge p;
    p.pad_id = id;
    p.x = 0.1 * id; p.y = -0.2 * id;
    p.q = 0.0;
    for (size_t i = 0; i < qt.size(); ++i) p.q += qt[i];
    p.k0 = k0;
    p.qt = qt;
    return p;
}

static const ReadoutPad* find_pad(const ReadoutEvent& e, int id)
{
    for (size_t i = 0; i < e.pads.size(); ++i) if (e.pads[i].pad_id == id) return &e.pads[i];
    return 0;
}

// premier echantillon j >= 0 tel que phase + j * R >= bin ; dernier j tel que phase + j * R <= bin
static int first_sample_from(int bin, int phase, int R)
{
    int j = 0;
    while (phase + j * R < bin) ++j;
    return j;
}
static int last_sample_to(int bin, int phase, int R)
{
    int j = 0;
    while (phase + (j + 1) * R <= bin) ++j;
    return j;
}

// signal de test de nq bins, amplitude 'amp' (forme quelconque mais non constante)
static std::vector<double> make_signal(int nq, double amp)
{
    std::vector<double> q(nq);
    for (int i = 0; i < nq; ++i) q[i] = amp * (1.0 + 0.3 * std::sin(0.7 * i) + 0.01 * i);
    return q;
}

static double sum_of(const std::vector<double>& v)
{
    double s = 0.0;
    for (size_t i = 0; i < v.size(); ++i) s += v[i];
    return s;
}

int main()
{
    const int n_bins = 216;
    const double dt = 10.0;

    GaussTestResponse resp(190.0);
    std::vector<double> h;
    resp.sample(dt, h);
    NoiseParams np; np.dt = dt; np.adc_ratio = 2; np.enc_total = 2000.0;
    NoiseModel model(np, h);

    ReadoutParams rp_off;
    ReadoutParams rp_on; rp_on.noise_on = true;
    Readout R_off(rp_off, model, n_bins), R_on(rp_on, model, n_bins);
    const int R = np.adc_ratio;
    const int tail = R_off.tail_bins(), margin = R_off.margin_bins(), grid = R_off.grid_bins();

    std::printf("test_chain : n_bins = %d, grille de lecture = %d, queue de h = %d bins (n_taps = %d), marge = %d bins, adc_ratio = %d\n",
                n_bins, grid, tail, (int)h.size(), margin, R);
    check(tail == (int)h.size() - 1, "la queue ecrite est la queue complete de h : n_taps - 1", (double)tail, (double)(h.size() - 1));
    check(grid == n_bins + tail + margin, "grille de lecture = n_bins + queue + marge", (double)grid, (double)(n_bins + tail + margin));

    // =====================================================================
    std::printf("\nC1 fenetre ecrite sur une famille de signaux\n");
    {
        const int k0s[] = { 0, 3, margin - 1, margin, margin + 1, 60, 100, 150 };
        const int nqs[] = { 1, 4, 30 };
        const int n_seeds = 10;
        int n_phase[2] = { 0, 0 };
        for (size_t ik = 0; ik < sizeof(k0s) / sizeof(k0s[0]); ++ik) {
            for (size_t in = 0; in < sizeof(nqs) / sizeof(nqs[0]); ++in) {
                int k0 = k0s[ik], nq = nqs[in];
                if (k0 + nq > n_bins) continue;
                int bad_start = 0, bad_end = 0, bad_off_on = 0;
                for (int s = 0; s < n_seeds; ++s) {
                    std::vector<PadCharge> H, L;
                    H.push_back(make_pad(3, k0, make_signal(nq, 2.0e5)));
                    ReadoutEvent a, b;
                    R_off.process(H, L, 1000 + s, a);
                    R_on.process(H, L, 1000 + s, b);
                    ++n_phase[a.phase];
                    const ReadoutPad* pa = find_pad(a, 3);
                    const ReadoutPad* pb = find_pad(b, 3);
                    if (pa == 0 || pb == 0) { ++bad_start; continue; }   // 2e5 e- : toujours lu
                    int first = k0 - margin; if (first < 0) first = 0;
                    const int last = k0 + nq - 1 + tail + margin;
                    const int j0 = first_sample_from(first, a.phase, R);
                    const int j1 = last_sample_to(last, a.phase, R);
                    if (pa->j0 != j0) ++bad_start;
                    if (pa->j0 + (int)pa->samples.size() - 1 != j1) ++bad_end;
                    if (pa->j0 != pb->j0 || pa->samples.size() != pb->samples.size()) ++bad_off_on;
                }
                char what[160];
                std::snprintf(what, sizeof what, "k0 = %3d nq = %2d : debut ecrit = max(k0 - marge, 0)%s", k0, nq,
                              k0 < margin ? " [debut de grille : coupe a 0, limite connue]" : "");
                check(bad_start == 0, what, (double)bad_start, 0.0);
                std::snprintf(what, sizeof what, "k0 = %3d nq = %2d : fin = k0 + nq - 1 + queue + marge (jamais tronquee)", k0, nq);
                check(bad_end == 0, what, (double)bad_end, 0.0);
                std::snprintf(what, sizeof what, "k0 = %3d nq = %2d : meme fenetre avec et sans bruit", k0, nq);
                check(bad_off_on == 0, what, (double)bad_off_on, 0.0);
            }
        }
        check(n_phase[0] > 0 && n_phase[1] > 0, "C1 les deux phases du CAN ont ete rencontrees", (double)(n_phase[0] + n_phase[1]), 2.0);
        // limite connue : signal des les premiers bins -> debut coupe a l'echantillon 0, mais valeurs exactes
        std::vector<PadCharge> H, L;
        H.push_back(make_pad(3, 0, make_signal(4, 2.0e5)));
        ReadoutEvent a; R_off.process(H, L, 5, a);
        const ReadoutPad* pa = find_pad(a, 3);
        std::vector<double> sig(grid, 0.0), shaped;
        std::vector<double> q4 = make_signal(4, 2.0e5);
        for (int k = 0; k < 4; ++k) sig[k] = q4[k];
        model.shape(sig, grid, shaped);
        double dmax = 0.0;
        if (pa) for (size_t i = 0; i < pa->samples.size(); ++i)
            dmax = std::max(dmax, std::fabs(pa->samples[i] - shaped[a.phase + (pa->j0 + (int)i) * R]));
        check(pa != 0 && pa->j0 == 0 && dmax < 1.0e-9, "C1 k0 = 0 : j0 = 0 et valeurs exactes malgre le debut coupe", dmax, 0.0);
    }

    // =====================================================================
    std::printf("\nC2 conservation : integrale de la sortie vraie = (somme de qt) x (somme de h)\n");
    {
        const double sum_h = sum_of(h);
        const int k0s[] = { 0, 40, 150, 186 };          // 186 + 30 = 216 = fin de la grille physique
        // (a) sur la grille de lecture entiere
        for (size_t ik = 0; ik < sizeof(k0s) / sizeof(k0s[0]); ++ik) {
            std::vector<double> q = make_signal(30, 1.0e5), sig(grid, 0.0), shaped;
            for (int i = 0; i < 30; ++i) sig[k0s[ik] + i] = q[i];
            model.shape(sig, grid, shaped);
            const double want = sum_of(q) * sum_h;
            const double got = sum_of(shaped);
            char what[160];
            std::snprintf(what, sizeof what, "(a) grille de lecture, k0 = %3d : somme(sortie) / (somme qt x somme h) - 1", k0s[ik]);
            check(std::fabs(got / want - 1.0) < 1.0e-12, what, got / want - 1.0, 0.0);
        }
        // (b) echantillons ecrits, adc_ratio = 1 : toute la sortie non nulle est dans la fenetre -> exact
        {
            NoiseParams n1 = np; n1.adc_ratio = 1;
            NoiseModel m1(n1, h);
            Readout R1(rp_off, m1, n_bins);
            for (size_t ik = 0; ik < sizeof(k0s) / sizeof(k0s[0]); ++ik) {
                std::vector<PadCharge> H, L;
                std::vector<double> q = make_signal(30, 1.0e5);
                H.push_back(make_pad(3, k0s[ik], q));
                ReadoutEvent e; R1.process(H, L, 9, e);
                const ReadoutPad* p = find_pad(e, 3);
                const double want = sum_of(q) * sum_h;
                const double got = p ? sum_of(p->samples) : 0.0;
                char what[160];
                std::snprintf(what, sizeof what, "(b) adc_ratio = 1, k0 = %3d : somme des echantillons ecrits / attendu - 1", k0s[ik]);
                check(p != 0 && std::fabs(got / want - 1.0) < 1.0e-12, what, got / want - 1.0, 0.0);
                check(p != 0 && std::fabs(p->q - sum_of(q)) < 1.0e-9, "(b) la charge vraie pq du pad = somme de qt", p ? p->q : -1.0, sum_of(q));
            }
        }
        // (c) adc_ratio = 2 : R x somme des echantillons ~ somme de la sortie (sortie lisse, gaussienne de 190 ns)
        for (size_t ik = 0; ik < sizeof(k0s) / sizeof(k0s[0]); ++ik) {
            std::vector<double> rel;
            for (int s = 0; s < 4; ++s) {
                std::vector<PadCharge> H, L;
                std::vector<double> q = make_signal(30, 1.0e5);
                H.push_back(make_pad(3, k0s[ik], q));
                ReadoutEvent e; R_off.process(H, L, 20 + s, e);
                const ReadoutPad* p = find_pad(e, 3);
                if (p == 0) { rel.push_back(1.0); continue; }
                rel.push_back((double)R * sum_of(p->samples) / (sum_of(q) * sum_h) - 1.0);
            }
            double w = 0.0;
            for (size_t i = 0; i < rel.size(); ++i) w = std::max(w, std::fabs(rel[i]));
            char what[160];
            std::snprintf(what, sizeof what, "(c) adc_ratio = %d, k0 = %3d : R x somme des echantillons / attendu - 1 (pire des phases)", R, k0s[ik]);
            check(w < 1.0e-3, what, w, 0.0);
        }
        // (d) evenement a plusieurs pads : chaque pad conserve sa propre charge
        {
            std::vector<PadCharge> H, L;
            H.push_back(make_pad(2, 20, make_signal(5, 3.0e4)));
            H.push_back(make_pad(8, 170, make_signal(40, 8.0e4)));
            L.push_back(make_pad(8, 170, make_signal(40, 8.0e4)));
            L.push_back(make_pad(4, 90, make_signal(12, 1.5e5)));
            NoiseParams n1 = np; n1.adc_ratio = 1;
            NoiseModel m1(n1, h);
            Readout R1(rp_off, m1, n_bins);
            ReadoutEvent e; R1.process(H, L, 11, e);
            double worst = 0.0;
            for (size_t i = 0; i < e.pads.size(); ++i) {
                const ReadoutPad& p = e.pads[i];
                worst = std::max(worst, std::fabs(sum_of(p.samples) / (p.q * sum_h) - 1.0));
            }
            check(e.pads.size() == 3 && worst < 1.0e-12, "(d) 3 pads (un H, un L, un H+L) : somme(sortie) = pq x somme h pour chacun", worst, 0.0);
        }
    }

    // =====================================================================
    std::printf("\nC3 linearite signal / bruit\n");
    {
        std::vector<PadCharge> H1, L1, H3, L3, Hh, Lh;
        const int ids[] = { 2, 5, 9 };
        const int k0s[] = { 30, 100, 180 };
        const int nqs[] = { 6, 20, 36 };
        for (int i = 0; i < 3; ++i) {
            H1.push_back(make_pad(ids[i], k0s[i], make_signal(nqs[i], 1.0e5)));
            H3.push_back(make_pad(ids[i], k0s[i], make_signal(nqs[i], 3.0e5)));
            Hh.push_back(make_pad(ids[i], k0s[i], make_signal(nqs[i], 0.5e5)));
        }
        const uint64_t seed = 777;
        ReadoutEvent c1, n1, c3, n3, ch, nh, nx;
        R_off.process(H1, L1, seed, c1);  R_on.process(H1, L1, seed, n1);
        R_off.process(H3, L3, seed, c3);  R_on.process(H3, L3, seed, n3);
        R_off.process(Hh, Lh, seed, ch);  R_on.process(Hh, Lh, seed, nh);
        R_on.process(H1, L1, seed + 1, nx);

        bool all_read = (c1.pads.size() == 3 && n1.pads.size() == 3 && n3.pads.size() == 3 && nh.pads.size() == 3 && nx.pads.size() == 3);
        check(all_read, "C3 les 3 pads sont lus dans tous les cas (pas d'effet de seuil)", (double)n1.pads.size(), 3.0);

        double d_amp = 0.0, d_half = 0.0, d_seed = 0.0, d_clean = 0.0;
        if (all_read) {
            for (int i = 0; i < 3; ++i) {
                const ReadoutPad &a1 = c1.pads[i], &b1 = n1.pads[i], &a3 = c3.pads[i], &b3 = n3.pads[i];
                const ReadoutPad &ah = ch.pads[i], &bh = nh.pads[i], &bx = nx.pads[i];
                for (size_t j = 0; j < a1.samples.size(); ++j) {
                    const double r1 = b1.samples[j] - a1.samples[j];           // bruit pur de l'evenement 1x
                    const double r3 = b3.samples[j] - a3.samples[j];           // bruit pur de l'evenement 3x
                    const double rh = bh.samples[j] - ah.samples[j];
                    d_amp  = std::max(d_amp,  std::fabs(r3 - r1));
                    d_half = std::max(d_half, std::fabs(rh - r1));
                    d_seed = std::max(d_seed, std::fabs((bx.samples[j] - a1.samples[j]) - r1));
                    d_clean = std::max(d_clean, std::fabs(a3.samples[j] - 3.0 * a1.samples[j]) / 3.0e5);
                }
            }
        }
        check(d_amp  < 1.0e-6, "C3 (bruite - propre) de l'evenement x3 = celui de l'evenement x1 (max, e-)", d_amp, 0.0);
        check(d_half < 1.0e-6, "C3 idem pour l'evenement x0,5", d_half, 0.0);
        check(d_clean < 1.0e-12, "C3 sortie sans bruit lineaire : propre(x3) = 3 x propre(x1)", d_clean, 0.0);
        check(d_seed > 100.0, "C3 une autre graine donne un autre bruit (ecart max > 100 e-)", d_seed, 100.0);
    }

    std::printf("\n%d verifications reussies, %d echecs\n", g_ok, g_fail);
    return g_fail == 0 ? 0 : 1;
}
