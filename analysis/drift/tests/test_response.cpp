// Test de la reponse MINOS (MinosResponse), independant des autres tests.
//
// Compilation et lancement (depuis le dossier tests/) :
//   g++ -O2 -std=c++17 -Wall -I../inc test_response.cpp ../src/noise.cpp -o test_response && ./test_response

#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "noise.hpp"
#include "preamp_response.hpp"

static int g_ok = 0, g_fail = 0;

static void check(bool cond, const char* what, double value, double expected) {
    if (cond) { ++g_ok; }
    else { ++g_fail; std::printf("ECHEC : %s  (obtenu %.6g, attendu %.6g)\n", what, value, expected); }
}

// rms de la sortie (parallele + serie filtres, puis CAN) pour des fractions donnees
static double rms_output(const std::vector<double>& h, double fp, double fs, double fc,
                         bool with_adc, double enc_total) {
    NoiseParams p;
    p.enc_total = enc_total; p.f_par = fp; p.f_ser = fs; p.f_can = fc;
    p.dt = 10.0; p.adc_ratio = 2;
    NoiseModel nm(p, h);
    GaussRng rng(12345);
    double sum2 = 0.0; long n = 0;
    for (int rep = 0; rep < 100; ++rep) {
        std::vector<double> out(2000, 0.0);
        nm.add_filtered_noise(rng, out);
        if (with_adc) {
            std::vector<double> s;
            nm.decimate(out, 0, s);
            nm.add_adc_noise(rng, s);
            for (size_t i = 0; i < s.size(); ++i) { sum2 += s[i] * s[i]; ++n; }
        } else {
            for (size_t i = 0; i < out.size(); ++i) { sum2 += out[i] * out[i]; ++n; }
        }
    }
    return std::sqrt(sum2 / n);
}

int main() {
    // ---- 1. position du maximum : u = 1.1664 ----
    const double up = MinosResponse::u_peak();
    check(std::fabs(up - 1.1664) < 1.0e-4, "u_peak = 1.1664", up, 1.1664);

    // ---- 2. forme, pour plusieurs temps de pic, avec dt = 1 ns (precision) ----
    const double peaks[] = {76.0, 123.0, 180.0, 388.0, 578.0, 919.0};
    for (int ip = 0; ip < 6; ++ip) {
        const double peak = peaks[ip];
        MinosResponse r(peak);
        std::vector<double> h;
        r.sample(1.0, h);
        const double tau = peak / up;

        check(std::fabs(r.tau() - tau) < 1.0e-9, "tau = peak / 1.1664", r.tau(), tau);

        double hmax = 0.0, hmin = 0.0, area = 0.0; int kmax = 0;
        for (size_t k = 0; k < h.size(); ++k) {
            if (h[k] > hmax) { hmax = h[k]; kmax = (int)k; }
            if (h[k] < hmin) hmin = h[k];
            area += h[k] * 1.0;
        }
        check(std::fabs(hmax - 1.0) < 1.0e-12, "max(h) = 1", hmax, 1.0);
        check(h[0] == 0.0, "causale : h[0] = 0", h[0], 0.0);
        check(std::fabs(kmax * 1.0 - peak) <= 0.5 + 1.0e-9, "temps du maximum = peak (dt = 1 ns)", kmax, peak);
        check(std::fabs(area / (1.306 * tau) - 1.0) < 2.0e-3, "integrale = 1.306 tau", area / tau, 1.306);
        check(std::fabs(hmin - (-0.00957)) < 3.0e-4, "sous-oscillation = -0.957 %", hmin, -0.00957);
        check(std::fabs(h[h.size() - 1]) < 2.0e-4, "queue : |h_dernier| < 1e-4", h[h.size() - 1], 1.0e-4);
        // la coupure n'est pas trop precoce : l'echantillon n-5 (en ns, n grand) depasse encore ~1e-4 en enveloppe
        // (verifie que le tableau n'est pas excessivement long : moins de 6.5 tau)
        check((double)h.size() < 6.5 * tau, "longueur du tableau < 6.5 tau", (double)h.size(), 6.5 * tau);
    }

    // ---- 3. nombre de taps a dt = 10 ns (coupure sur l enveloppe, calcul independant en Python : 90) ----
    {
        MinosResponse r(175.0);
        std::vector<double> h;
        r.sample(10.0, h);
        check(h.size() == 90, "175 ns, dt = 10 ns : 90 echantillons", (double)h.size(), 90.0);
    }

    // ---- 4. NoiseModel accepte la reponse, dt = 10 ns, pour tous les reglages ----
    for (int ip = 0; ip < 6; ++ip) {
        MinosResponse r(peaks[ip]);
        std::vector<double> h;
        r.sample(10.0, h);
        bool ok = true;
        try {
            NoiseParams p; p.dt = 10.0; p.adc_ratio = 2;
            NoiseModel nm(p, h);
            ok = (nm.n_taps() == (int)h.size()) && std::fabs(nm.sum_dh()) < 1.0e-9;
        } catch (const std::exception& e) { ok = false; std::printf("exception : %s\n", e.what()); }
        check(ok, "NoiseModel accepte la reponse MINOS (n_taps, sum_dh = 0)", ok ? 1.0 : 0.0, 1.0);
    }

    // ---- 5. rms de sortie = ENC pour chaque composante (tp = 388 ns) puis pour le total ----
    {
        MinosResponse r(388.0);
        std::vector<double> h;
        r.sample(10.0, h);
        const double enc = 2000.0;
        double v;
        v = rms_output(h, 1.0, 0.0, 0.0, false, enc);
        check(std::fabs(v / enc - 1.0) < 0.015, "bruit parallele seul : rms = ENC", v, enc);
        v = rms_output(h, 0.0, 1.0, 0.0, false, enc);
        check(std::fabs(v / enc - 1.0) < 0.015, "bruit serie seul : rms = ENC", v, enc);
        v = rms_output(h, 0.0, 0.0, 1.0, true, enc);
        check(std::fabs(v / enc - 1.0) < 0.015, "bruit CAN seul : rms = ENC", v, enc);
        v = rms_output(h, 0.475, 0.475, 0.05, true, enc);
        check(std::fabs(v / enc - 1.0) < 0.015, "total apres CAN : rms = ENC", v, enc);
    }

    // ---- 6. parametres invalides ----
    {
        bool thrown = false;
        try { MinosResponse r(0.0); } catch (const std::exception&) { thrown = true; }
        check(thrown, "peak = 0 refuse", 0.0, 1.0);
        thrown = false;
        try { MinosResponse r(-50.0); } catch (const std::exception&) { thrown = true; }
        check(thrown, "peak < 0 refuse", 0.0, 1.0);
        thrown = false;
        try { MinosResponse r(300.0); std::vector<double> h; r.sample(0.0, h); } catch (const std::exception&) { thrown = true; }
        check(thrown, "dt = 0 refuse", 0.0, 1.0);
    }

    std::printf("%d verifications, %d echec(s)\n", g_ok + g_fail, g_fail);
    return g_fail == 0 ? 0 : 1;
}
