// ---------------------------------------------------------------------------
// Tests des pads fantomes (ghost.hpp / ghost.cpp, et leur emploi dans readout.cpp).
//
// Usage :  ./test_ghost [facteur]      (defaut : 1 ; 0.3 pour un essai rapide, 3 pour plus de statistique)
//
//   G1  covariance analytique c(k) des echantillons du CAN = covariance empirique ; c(0) = enc_total^2
//   G2  gaussienne tronquee : moyenne, minimum
//   G3  probabilite p (estimateur d'importance) = frequence brute-force d'un pad de bruit lu
//   G4  forme d'un fantome (tiree sous condition) = forme brute-force "bruit pur, garde si lu"
//   G5  dans Readout : nombre de fantomes binomial, identifiants uniformes, flag 0, fenetre,
//       pads du signal inchange, reproductibilite, pas de fantome sans bruit
//   G6  erreurs
//
// Aucune reponse MINOS : gaussienne de test. Code retour : 0 si tout passe. C++ classique.
// ---------------------------------------------------------------------------

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "ghost.hpp"
#include "noise.hpp"
#include "preamp_response.hpp"
#include "readout.hpp"

static int g_fail = 0, g_ok = 0;

static void check(bool cond, const std::string& what, double value, double expected)
{
    if (cond) { ++g_ok;   std::printf("  ok   %-70s %12.5g (attendu %.5g)\n", what.c_str(), value, expected); }
    else      { ++g_fail; std::printf("  ECHEC %-69s %12.5g (attendu %.5g)\n", what.c_str(), value, expected); }
}

// bruit pur d'un pad sans signal : meme chaine que les pads du signal
static void brute_noise(const NoiseModel& M, GaussRng& rng, int n_bins, int phase, std::vector<double>& smp)
{
    std::vector<double> bins(n_bins, 0.0);
    M.add_filtered_noise(rng, bins);
    M.decimate(bins, phase, smp);
    M.add_adc_noise(rng, smp);
}

// moyenne et erreur sur la moyenne d'une suite
struct Acc {
    double s1, s2; long n;
    Acc() : s1(0), s2(0), n(0) {}
    void add(double x) { s1 += x; s2 += x * x; ++n; }
    double mean() const { return n ? s1 / n : 0.0; }
    double se() const {
        if (n < 2) return 0.0;
        const double m = s1 / n, v = s2 / n - m * m;
        return std::sqrt((v > 0 ? v : 0.0) / (n - 1));
    }
};

static std::string fmt(const std::string& f, double a)
{
    char b[200]; std::snprintf(b, sizeof(b), f.c_str(), a); return std::string(b);
}

// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    const double fac = (argc > 1) ? std::atof(argv[1]) : 1.0;
    const int n_bins = 216;
    const double dt = 10.0;

    GaussTestResponse resp(190.0);
    std::vector<double> h;
    resp.sample(dt, h);
    NoiseParams np; np.dt = dt; np.adc_ratio = 2; np.enc_total = 2000.0;
    NoiseModel model(np, h);
    const double enc = np.enc_total;

    std::printf("test_ghost : facteur de statistique %g ; n_bins = %d, adc_ratio = %d, enc_total = %g e-\n\n",
                fac, n_bins, np.adc_ratio, enc);

    // =======================================================================
    std::printf("=== G1  covariance des echantillons du CAN ===\n");
    {
        GhostParams gp; gp.window_ns = 940.0;
        GhostModel G(model, gp, 4.0 * enc, 1, n_bins);

        // c(0) = sigma_par^2 sum h^2 + sigma_ser^2 sum dh^2 + enc_can^2 = enc_total^2
        const double c0f = model.sigma_par() * model.sigma_par() * model.sum_h2()
                         + model.sigma_ser() * model.sigma_ser() * model.sum_dh2()
                         + model.enc_can() * model.enc_can();
        check(std::fabs(c0f / (enc * enc) - 1.0) < 1.0e-10, "c(0) = enc_total^2 (somme des trois composantes)", c0f / (enc * enc), 1.0);

        // covariance empirique, k = 0 .. 4
        GaussRng rng(2024);
        const int N = (int)(20000 * fac);
        const int nk = 5;
        std::vector<double> acc(nk, 0.0);
        std::vector<long> cnt(nk, 0);
        std::vector<double> smp;
        for (int n = 0; n < N; ++n) {
            brute_noise(model, rng, n_bins, n % np.adc_ratio, smp);
            for (int k = 0; k < nk; ++k)
                for (size_t j = 0; j + k < smp.size(); ++j) { acc[k] += smp[j] * smp[j + k]; ++cnt[k]; }
        }
        const double c0 = enc * enc;
        for (int k = 0; k < nk; ++k) {
            const double emp = acc[k] / (double)cnt[k];
            const double tol = 4.0 * std::sqrt(2.0 / (double)cnt[k]) * c0 * (k == 0 ? 1.0 : 1.6) + 1.0e-3 * c0;
            check(std::fabs(emp - G.cov(k)) < tol, fmt("c(%.0f) empirique = analytique", (double)k), emp / c0, G.cov(k) / c0);
        }
        check(G.cov(0) > G.cov(1) && G.cov(1) > 0.0, "c(0) > c(1) > 0 (bruit filtre correle)", G.cov(1) / G.cov(0), 0.5);
        std::printf("       lags non nuls : %d ; c(k)/c(0) = ", G.n_cov());
        for (int k = 0; k < G.n_cov() && k < 8; ++k) std::printf("%.4f ", G.cov(k) / G.cov(0));
        std::printf("\n");
    }

    // =======================================================================
    std::printf("\n=== G2  gaussienne tronquee ===\n");
    {
        GaussRng rng(7);
        const double ts[5] = { -0.5, 0.0, 1.0, 2.5, 4.0 };
        const int N = (int)(200000 * fac);
        for (int it = 0; it < 5; ++it) {
            const double t = ts[it];
            Acc a; double zmin = 1.0e9;
            for (int n = 0; n < N; ++n) { const double z = GhostModel::normal_above(rng, t); a.add(z); if (z < zmin) zmin = z; }
            const double phi = std::exp(-0.5 * t * t) / std::sqrt(2.0 * 3.14159265358979323846);
            const double Q = 0.5 * std::erfc(t / std::sqrt(2.0));
            const double mexp = phi / Q;
            check(std::fabs(a.mean() - mexp) < 4.5 * a.se(), fmt("moyenne de z | z > t,  t = %.1f", t) , a.mean(), mexp);
            check(zmin > t, "minimum > t", zmin, t);
        }
    }

    // =======================================================================
    std::printf("\n=== G3  probabilite p : importance vs brute-force ===\n");
    {
        const double thr_sig[3] = { 3.0, 3.5, 2.5 };
        const int    ncons[3]   = { 1, 1, 2 };
        for (int it = 0; it < 3; ++it) {
            GhostParams gp; gp.calib_trials = (int)(20000 * fac);
            GhostModel G(model, gp, thr_sig[it] * enc, ncons[it], n_bins);
            GaussRng rc(100 + it);
            G.calibrate(rc);

            GaussRng rb(200 + it);
            const int Nb = (int)(40000 * fac);
            long hit = 0;
            std::vector<double> smp;
            for (int n = 0; n < Nb; ++n) {
                brute_noise(model, rb, n_bins, 0, smp);
                if (GhostModel::first_run(smp, thr_sig[it] * enc, ncons[it]) >= 0) ++hit;
            }
            const double pb = (double)hit / Nb;
            const double seb = std::sqrt(pb * (1.0 - pb) / Nb);
            const double tol = 4.0 * std::sqrt(seb * seb + G.prob_err() * G.prob_err());
            char lab[160];
            std::snprintf(lab, sizeof(lab), "p (seuil %.1f sigma, %d echantillon(s)) : importance %.4f +- %.4f", thr_sig[it], ncons[it], G.prob(), G.prob_err());
            check(std::fabs(G.prob() - pb) < tol, lab, G.prob(), pb);
        }
        // seuil de production : bornes et valeur affichee
        GhostParams gp; gp.calib_trials = (int)(20000 * fac);
        GhostModel G(model, gp, 4.0 * enc, 1, n_bins);
        GaussRng rc(77);
        G.calibrate(rc);
        std::printf("       seuil de production 4 sigma, 1 echantillon : p = %.3e +- %.1e  (Ns q = %.3e)\n",
                    G.prob(), G.prob_err(), G.n_samples() * G.q_tail());
        check(G.prob() > G.q_tail() && G.prob() < G.n_samples() * G.q_tail() * 1.0000001,
              "q < p < Ns q (bornes)", G.prob(), G.n_samples() * G.q_tail());
        // p imposee
        GhostParams gq; gq.prob = 0.0123;
        GhostModel Gq(model, gq, 4.0 * enc, 1, n_bins);
        GaussRng rq(1);
        Gq.calibrate(rq);
        check(Gq.prob() == 0.0123 && Gq.prob_err() == 0.0, "p imposee (ghost_prob >= 0) : reprise telle quelle", Gq.prob(), 0.0123);
    }

    // =======================================================================
    std::printf("\n=== G4  forme d'un fantome : tirage conditionne vs brute-force (bruit pur garde si lu) ===\n");
    {
        const double thr_sig[2] = { 3.0, 2.5 };
        const int    ncons[2]   = { 1, 2 };
        for (int it = 0; it < 2; ++it) {
            const double thr = thr_sig[it] * enc;
            GhostParams gp;
            GhostModel G(model, gp, thr, ncons[it], n_bins);

            const int NO = 6;
            const char* names[NO] = { "valeur au premier franchissement", "echantillon suivant", "echantillon j+2",
                                       "maximum", "nombre d'echantillons > seuil", "position du premier franchissement" };
            Acc A[NO], B[NO];

            // brute-force
            GaussRng rb(300 + it);
            const int Nb = (int)(60000 * fac);
            std::vector<double> smp;
            for (int n = 0; n < Nb; ++n) {
                brute_noise(model, rb, n_bins, 0, smp);
                const int jf = GhostModel::first_run(smp, thr, ncons[it]);
                if (jf < 0) continue;
                const int ns = (int)smp.size();
                double mx = -1.0e30; int K = 0;
                for (int j = 0; j < ns; ++j) { if (smp[j] > mx) mx = smp[j]; if (smp[j] > thr) ++K; }
                if (jf + 2 < ns) { B[0].add(smp[jf] / enc); B[1].add(smp[jf + 1] / enc); B[2].add(smp[jf + 2] / enc); }
                B[3].add(mx / enc); B[4].add((double)K); B[5].add((double)jf / ns);
            }
            // tirage conditionne
            GaussRng rg(400 + it);
            const int Ng = (int)(6000 * fac);
            for (int n = 0; n < Ng; ++n) {
                int jf = 0;
                G.draw(rg, 0, smp, jf);
                const int ns = (int)smp.size();
                double mx = -1.0e30; int K = 0;
                for (int j = 0; j < ns; ++j) { if (smp[j] > mx) mx = smp[j]; if (smp[j] > thr) ++K; }
                if (jf + 2 < ns) { A[0].add(smp[jf] / enc); A[1].add(smp[jf + 1] / enc); A[2].add(smp[jf + 2] / enc); }
                A[3].add(mx / enc); A[4].add((double)K); A[5].add((double)jf / ns);
            }
            std::printf("  seuil %.1f sigma, %d echantillon(s) consecutif(s) : %ld fantomes brute-force, %d tires\n", thr_sig[it], ncons[it], B[3].n, Ng);
            for (int o = 0; o < NO; ++o) {
                const double d = A[o].mean() - B[o].mean();
                const double se = std::sqrt(A[o].se() * A[o].se() + B[o].se() * B[o].se());
                char lab[200];
                std::snprintf(lab, sizeof(lab), "%s : %.4f vs %.4f (ecart %.1f se)", names[o], A[o].mean(), B[o].mean(), se > 0 ? d / se : 0.0);
                check(std::fabs(d) < 4.5 * se, lab, A[o].mean(), B[o].mean());
            }
        }
    }

    // =======================================================================
    std::printf("\n=== G5  fantomes dans Readout ===\n");
    {
        const int n_sites = 1114;
        std::vector<PadSite> sites(n_sites);
        for (int i = 0; i < n_sites; ++i) { sites[i].pad_id = i; sites[i].x = 0.1 * i; sites[i].y = -0.05 * i; }

        // signal : pads 5, 6 (H) et 6, 7 (L), gros signaux (passent toujours le seuil)
        std::vector<PadCharge> H, L;
        {
            std::vector<double> qt(30, 0.0);
            for (int k = 0; k < 30; ++k) qt[k] = 40000.0;     // 1.2 Me-
            PadCharge a; a.pad_id = 5; a.x = 0.5; a.y = -0.25; a.q = 1.2e6; a.k0 = 80; a.qt = qt;
            PadCharge b = a; b.pad_id = 6; b.x = 0.6; b.y = -0.3; b.k0 = 85;
            PadCharge c = a; c.pad_id = 7; c.x = 0.7; c.y = -0.35; c.k0 = 90;
            H.push_back(a); H.push_back(b); L.push_back(b); L.push_back(c);
        }
        const int n_true = 3;

        ReadoutParams rp_on; rp_on.noise_on = true;
        Readout R_on(rp_on, model, n_bins);                 // sans fantomes
        Readout R_gh(rp_on, model, n_bins);                 // avec fantomes
        GhostParams gp; gp.calib_trials = (int)(8000 * fac); gp.window_ns = 940.0;
        GhostModel G(model, gp, R_gh.threshold(), rp_on.n_consec, R_gh.n_bins());
        GaussRng rc(555);
        G.calibrate(rc);
        R_gh.set_ghosts(&G, sites);
        std::printf("       p = %.3e +- %.1e, fenetre = %d echantillons, seuil = %g e-\n", G.prob(), G.prob_err(), G.window_samples(), R_gh.threshold());

        const int NE = (int)(1500 * fac);
        const int ncand = n_sites - n_true;
        Acc ng; long total_ghost = 0;
        bool ok_flag = true, ok_q = true, ok_sorted = true, ok_win = true, ok_cross = true, ok_distinct = true, ok_true_same = true, ok_nontrue = true;
        long dec[10] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        for (int e = 0; e < NE; ++e) {
            const uint64_t seed = 9000 + 17 * (uint64_t)e;
            ReadoutEvent a, b;
            R_on.process(H, L, seed, a);
            R_gh.process(H, L, seed, b);
            ng.add((double)b.n_ghost);
            total_ghost += b.n_ghost;

            // pads du signal inchanges
            int nghost_seen = 0, ntrue_seen = 0;
            for (size_t i = 0; i < b.pads.size(); ++i) {
                if (i > 0 && b.pads[i - 1].pad_id >= b.pads[i].pad_id) ok_sorted = false;
                const ReadoutPad& p = b.pads[i];
                if (p.flag == 0) {
                    ++nghost_seen;
                    if (p.q != 0.0) ok_q = false;
                    if (p.pad_id >= 5 && p.pad_id <= 7) ok_nontrue = false;
                    if ((int)p.samples.size() != G.window_samples()) ok_win = false;
                    double mx = -1.0e30;
                    for (size_t j = 0; j < p.samples.size(); ++j) if (p.samples[j] > mx) mx = p.samples[j];
                    if (!(mx > R_gh.threshold())) ok_cross = false;
                    int d = (int)(10.0 * p.pad_id / n_sites); if (d > 9) d = 9; ++dec[d];
                } else {
                    ++ntrue_seen;
                    if (p.flag < 1 || p.flag > 3) ok_flag = false;
                    const ReadoutPad* pa = 0;
                    for (size_t k = 0; k < a.pads.size(); ++k) if (a.pads[k].pad_id == p.pad_id) pa = &a.pads[k];
                    if (!pa || pa->samples.size() != p.samples.size() || pa->j0 != p.j0 || pa->flag != p.flag) ok_true_same = false;
                    else for (size_t j = 0; j < p.samples.size(); ++j) if (pa->samples[j] != p.samples[j]) ok_true_same = false;
                }
            }
            if (ntrue_seen != (int)a.pads.size()) ok_true_same = false;
            if (nghost_seen != b.n_ghost) ok_flag = false;
            for (size_t i = 1; i < b.pads.size(); ++i) if (b.pads[i].pad_id == b.pads[i - 1].pad_id) ok_distinct = false;
            if (a.n_ghost != 0) ok_flag = false;
        }
        const double mean_exp = G.prob() * ncand;
        const double var_exp  = ncand * G.prob() * (1.0 - G.prob());
        const double tol = 4.0 * std::sqrt(var_exp / NE + ncand * ncand * G.prob_err() * G.prob_err());
        check(std::fabs(ng.mean() - mean_exp) < tol, fmt("nombre moyen de fantomes par evenement = p x candidats (%.3f)", mean_exp), ng.mean(), mean_exp);
        const double vr = (ng.s2 / ng.n - ng.mean() * ng.mean()) * ng.n / (ng.n - 1.0);
        check(std::fabs(vr / var_exp - 1.0) < 6.0 * std::sqrt(2.0 / NE) + 0.05, "variance du nombre de fantomes = binomiale", vr, var_exp);
        check(ok_flag, "flags : 1..3 pour le signal, 0 pour les fantomes, n_ghost coherent", 1, 1);
        check(ok_q, "charge vraie d'un fantome = 0", 1, 1);
        check(ok_nontrue, "un fantome n'est jamais un pad de la liste vraie", 1, 1);
        check(ok_sorted && ok_distinct, "liste triee par pad_id, identifiants distincts", 1, 1);
        check(ok_win, "fenetre d'un fantome = ghost_window (fixe)", 1, 1);
        check(ok_cross, "la fenetre ecrite d'un fantome contient le franchissement du seuil", 1, 1);
        check(ok_true_same, "pads du signal identiques avec / sans fantomes (meme graine)", 1, 1);
        // identifiants uniformes : chaque dixieme recoit ~ 1/10 (sous reserve des 3 pads du signal)
        bool ok_uni = true; double worst = 0.0;
        for (int d = 0; d < 10; ++d) {
            const double ex = (double)total_ghost / 10.0;
            const double z = (dec[d] - ex) / std::sqrt(ex * 0.9);
            if (std::fabs(z) > worst) worst = std::fabs(z);
            if (std::fabs(z) > 4.5) ok_uni = false;
        }
        check(ok_uni, "identifiants des fantomes uniformes sur les pads (pire ecart, en sigma)", worst, 0.0);

        // reproductibilite
        ReadoutEvent r1, r2;
        R_gh.process(H, L, 31337, r1);
        R_gh.process(H, L, 31337, r2);
        bool same = (r1.pads.size() == r2.pads.size()) && r1.n_ghost == r2.n_ghost && r1.phase == r2.phase;
        for (size_t i = 0; same && i < r1.pads.size(); ++i) {
            same = r1.pads[i].pad_id == r2.pads[i].pad_id && r1.pads[i].j0 == r2.pads[i].j0 && r1.pads[i].samples == r2.pads[i].samples;
        }
        check(same, "reproductibilite : meme graine, meme evenement", 1, 1);

        // sans bruit : pas de fantomes
        ReadoutParams rp_off;                   // noise_on = false
        Readout R_off(rp_off, model, n_bins);
        R_off.set_ghosts(&G, sites);
        ReadoutEvent r0;
        R_off.process(H, L, 31337, r0);
        check(r0.n_ghost == 0 && (int)r0.pads.size() == n_true, "sans bruit : pas de fantome", r0.n_ghost, 0);

        // p imposee = 0 : aucun ; p imposee = 1 : tous les candidats
        GhostParams g0; g0.prob = 0.0;
        GhostModel G0(model, g0, R_gh.threshold(), 1, n_bins);
        GaussRng r(1); G0.calibrate(r);
        Readout R0(rp_on, model, n_bins); R0.set_ghosts(&G0, sites);
        ReadoutEvent e0; R0.process(H, L, 5, e0);
        check(e0.n_ghost == 0, "ghost_prob = 0 : aucun fantome", e0.n_ghost, 0);
        GhostParams g1; g1.prob = 1.0;
        GhostModel G1(model, g1, R_gh.threshold(), 1, n_bins);
        G1.calibrate(r);
        Readout R1(rp_on, model, n_bins); R1.set_ghosts(&G1, sites);
        ReadoutEvent e1; R1.process(H, L, 5, e1);
        check(e1.n_ghost == ncand && (int)e1.pads.size() == ncand + n_true, "ghost_prob = 1 : tous les pads hors liste sont des fantomes", e1.n_ghost, ncand);
    }

    // =======================================================================
    std::printf("\n=== G6  erreurs ===\n");
    {
        GhostParams gp;
        bool t1 = false, t2 = false, t3 = false, t4 = false;
        try { GhostParams b = gp; b.window_ns = -1.0; GhostModel g(model, b, 8000.0, 1, n_bins); } catch (const std::exception&) { t1 = true; }
        try { GhostParams b = gp; b.calib_trials = 10; GhostModel g(model, b, 8000.0, 1, n_bins); } catch (const std::exception&) { t2 = true; }
        try { GhostModel g(model, gp, 8000.0, 0, n_bins); } catch (const std::exception&) { t3 = true; }
        try { GhostModel g(model, gp, -1.0, 1, n_bins); } catch (const std::exception&) { t4 = true; }
        check(t1, "ghost_window_ns < 0 refuse", 1, 1);
        check(t2, "ghost_calib_trials < 100 refuse", 1, 1);
        check(t3, "n_consec < 1 refuse", 1, 1);
        check(t4, "seuil < 0 refuse", 1, 1);
    }

    std::printf("\n%d verifications reussies, %d echecs\n", g_ok, g_fail);
    return g_fail == 0 ? 0 : 1;
}
