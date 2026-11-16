// Test du modele ENC(tp, C) (enc_model.hpp) et de son branchement sur CapNoiseModel / NoiseModel
// avec la reponse MINOS. Independant des autres tests.
//
// Compilation et lancement (depuis le dossier tests/) :
//   g++ -O2 -std=c++17 -Wall -I../inc test_encmodel.cpp ../src/noise.cpp -o test_encmodel && ./test_encmodel

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "noise.hpp"
#include "preamp_response.hpp"
#include "cap_noise.hpp"
#include "enc_model.hpp"

static int g_ok = 0, g_fail = 0;

static void check(bool cond, const char* what, double value, double expected) {
    if (cond) { ++g_ok; }
    else { ++g_fail; std::printf("ECHEC : %s  (obtenu %.6g, attendu %.6g)\n", what, value, expected); }
}

// rms de sortie de tout le bruit (parallele + serie filtres, puis CAN) d'un NoiseModel
static double rms_total(const NoiseModel& nm, double ser_scale) {
    GaussRng rng(777);
    double sum2 = 0.0; long n = 0;
    for (int rep = 0; rep < 150; ++rep) {
        std::vector<double> out(2000, 0.0);
        nm.add_filtered_noise(rng, out, ser_scale);
        std::vector<double> s;
        nm.decimate(out, 0, s);
        nm.add_adc_noise(rng, s);
        for (size_t i = 0; i < s.size(); ++i) { sum2 += s[i] * s[i]; ++n; }
    }
    return std::sqrt(sum2 / n);
}

int main() {
    // ---- 1. les 30 points de la figure (C arrondis a 0, 50, 100, 160, 220 pF) ----
    const double C[5] = {0.0, 50.0, 100.0, 160.0, 220.0};
    const double tps[6] = {76.0, 123.0, 180.0, 388.0, 578.0, 919.0};
    const double pts[6][5] = {
        {1369.5, 1746.3, 2381.8, 3046.8, 3970.4},
        {1170.0, 1436.0, 1849.8, 2344.8, 3054.2},
        {1118.2, 1317.7, 1642.9, 1997.5, 2551.7},
        {1199.5, 1273.4, 1487.7, 1679.8, 2027.1},
        {1273.4, 1339.9, 1472.9, 1628.1, 1908.9},
        {1480.3, 1524.6, 1591.1, 1694.6, 1886.7}};
    double s2 = 0.0, emax = 0.0;
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 5; ++j) {
            const double d = EncTpModel::enc(tps[i], C[j]) - pts[i][j];
            s2 += d * d; if (std::fabs(d) > emax) emax = std::fabs(d);
        }
    const double rms = std::sqrt(s2 / 30.0);
    check(rms < 65.0,  "ecart rms avec les 30 points < 65 e-",  rms, 65.0);
    check(emax < 175.0, "ecart max avec les 30 points < 175 e-", emax, 175.0);

    // ---- 2. composantes : somme = ENC^2, valeurs de reference (tp = 388 ns) ----
    {
        const EncComponents e = EncTpModel::components(388.0);
        check(std::fabs(std::sqrt(e.par2) - 905.0) < 3.0, "tp = 388 : parallele ~ 905 e-", std::sqrt(e.par2), 905.0);
        check(std::fabs(std::sqrt(e.ser02) - 582.0) < 3.0, "tp = 388 : serie(C=0) ~ 582 e-", std::sqrt(e.ser02), 582.0);
        check(std::fabs(std::sqrt(e.can2) - 510.0) < 3.0, "tp = 388 : CAN ~ 510 e-", std::sqrt(e.can2), 510.0);
        check(std::fabs(e.b - 7.33) < 0.02, "tp = 388 : b ~ 7.33 e-/pF", e.b, 7.33);
        const double c = 103.0;
        const double sum = std::sqrt(e.par2 + e.ser02 + e.can2 + e.b * e.b * c * c);
        check(std::fabs(sum - EncTpModel::enc(388.0, c)) < 1.0e-9, "somme des composantes = ENC(tp, C)", sum, EncTpModel::enc(388.0, c));
    }

    // ---- 3. plage : hors de [76 ; 919] refuse, bornes acceptees ----
    {
        bool t1 = false, t2 = false, t3 = false, t4 = false;
        try { EncTpModel::components(75.9); }  catch (const std::exception&) { t1 = true; }
        try { EncTpModel::components(919.1); } catch (const std::exception&) { t2 = true; }
        try { EncTpModel::components(50.0); }  catch (const std::exception&) { t3 = true; }
        try { EncTpModel::components(76.0); EncTpModel::components(919.0); } catch (const std::exception&) { t4 = true; }
        check(t1 && t2 && t3, "tp hors de 76-919 ns refuse", 0.0, 1.0);
        check(!t4, "tp = 76 et 919 ns acceptes", t4 ? 1.0 : 0.0, 0.0);
    }

    // ---- 4. CapNoiseModel construit depuis les composantes ----
    const double c_ref = 93.0;
    for (int i = 0; i < 6; ++i) {
        const double tp = tps[i];
        const EncComponents e = EncTpModel::components(tp);
        CapNoiseModel cm(e.par2, e.ser02, e.can2, e.b, c_ref);
        check(std::fabs(cm.enc_total(c_ref) - EncTpModel::enc(tp, c_ref)) < 1.0e-9,
              "CapNoiseModel : enc_total(c_ref) = ENC(tp, c_ref)", cm.enc_total(c_ref), EncTpModel::enc(tp, c_ref));
        check(std::fabs(cm.params().enc_ref - EncTpModel::enc(tp, c_ref)) < 1.0e-9,
              "CapNoiseModel : params().enc_ref = ENC(tp, c_ref)", cm.params().enc_ref, EncTpModel::enc(tp, c_ref));
        check(std::fabs(cm.enc_total(103.0) - EncTpModel::enc(tp, 103.0)) < 1.0e-9,
              "CapNoiseModel : enc_total(103 pF) = ENC(tp, 103 pF)", cm.enc_total(103.0), EncTpModel::enc(tp, 103.0));
        check(std::fabs(cm.enc_can() - std::sqrt(e.can2)) < 1.0e-9, "CapNoiseModel : enc_can", cm.enc_can(), std::sqrt(e.can2));
        check(std::fabs(cm.ser_scale(c_ref) - 1.0) < 1.0e-12, "ser_scale(c_ref) = 1", cm.ser_scale(c_ref), 1.0);

        // ---- 5. NoiseModel avec h MINOS : rms de sortie = ENC(tp, C) pour c_ref et pour 103 pF ----
        MinosResponse r(tp);
        std::vector<double> h;
        r.sample(10.0, h);
        NoiseParams np; np.dt = 10.0; np.adc_ratio = 2;
        cm.fill_noise_params(np);
        NoiseModel nm(np, h);
        const double v_ref = rms_total(nm, 1.0);
        check(std::fabs(v_ref / EncTpModel::enc(tp, c_ref) - 1.0) < 0.02, "rms de sortie = ENC(tp, c_ref)",
              v_ref, EncTpModel::enc(tp, c_ref));
        const double v103 = rms_total(nm, cm.ser_scale(103.0));
        check(std::fabs(v103 / EncTpModel::enc(tp, 103.0) - 1.0) < 0.02, "rms de sortie = ENC(tp, 103 pF)",
              v103, EncTpModel::enc(tp, 103.0));
    }

    // ---- 6. le chemin d'origine de CapNoiseModel est inchange (enc0 = 1150, enc_ref = 1400, c_ref = 93) ----
    {
        CapNoiseParams cp;
        CapNoiseModel cm(cp);
        check(std::fabs(cm.enc_total(93.0) - 1400.0) < 1.0e-9, "ancien constructeur : enc_total(93 pF) = 1400", cm.enc_total(93.0), 1400.0);
        check(std::fabs(cm.enc_total(0.0) - 1150.0) < 1.0e-9,  "ancien constructeur : enc_total(0) = 1150", cm.enc_total(0.0), 1150.0);
    }

    std::printf("%d verifications, %d echec(s)\n", g_ok + g_fail, g_fail);
    return g_fail == 0 ? 0 : 1;
}
