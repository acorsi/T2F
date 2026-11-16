// ---------------------------------------------------------------------------
// Tests du module de bruit (noise.hpp / noise.cpp) et de l'interface preamp_response.hpp.
//
// Usage :  ./test_noise [n_essais]          (defaut : 2000 realisations par cas)
//
// Les tests valent pour N'IMPORTE QUEL h : ils sont faits avec la gaussienne de test et avec
// deux autres formes sans rapport avec le papier MINOS (une reponse CR-RC^2 et une reponse a
// sous-oscillation). Aucune reponse MINOS dans ce fichier.
//
//   T0  generateur : moyenne, variance, kurtosis, reproductibilite
//   T1  impulsion : une charge Q injectee d'un coup donne un pic de sortie Q (convention ENC), et shape(delta) = h
//   T2  bruit de sortie rms = ENC pour : parallele seul, serie seul, CAN seul, et le melange (total)
//   T3  le bruit serie n'a pas de composante continue : somme(h[j] - h[j-1]) = 0
//   T4  autocorrelation du bruit de sortie = theorie (parallele, serie)
//   T5  reproductibilite a graine egale
//   T6  mise a l'echelle avec dt : sigma_par ~ sqrt(dt), sigma_ser ~ 1/sqrt(dt), et sortie inchangee
//   T7  decimation (phases)
//   T8  refus d'une reponse mal normalisee ou tronquee
//
// Code retour : 0 si tout passe, 1 sinon.  C++ classique.
// ---------------------------------------------------------------------------

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "noise.hpp"
#include "preamp_response.hpp"

static int g_fail = 0;
static int g_ok   = 0;

static void check(bool cond, const std::string& what, double value, double expected, double tol)
{
    if (cond) {
        ++g_ok;
        std::printf("  ok   %-62s %12.5g (attendu %.5g, tol %.3g)\n", what.c_str(), value, expected, tol);
    } else {
        ++g_fail;
        std::printf("  ECHEC %-61s %12.5g (attendu %.5g, tol %.3g)\n", what.c_str(), value, expected, tol);
    }
}

static void check_rel(const std::string& what, double value, double expected, double tol_rel)
{
    check(std::fabs(value - expected) <= tol_rel * std::fabs(expected), what, value, expected, tol_rel);
}

static void check_abs(const std::string& what, double value, double expected, double tol_abs)
{
    check(std::fabs(value - expected) <= tol_abs, what, value, expected, tol_abs);
}

// ---------------------------------------------------------------------------
// Reponses de test (aucune ne ressemble a la reponse MINOS)
// ---------------------------------------------------------------------------

// CR-RC^2 : h ~ x^2 exp(-2x), x = t / tau
class RcRcTestResponse : public PreampResponse {
public:
    explicit RcRcTestResponse(double tau) : m_tau(tau) {}
    std::string name() const { return "test_rcrc2"; }
    std::string describe() const { return "x^2 exp(-2x), test seulement"; }
    void sample(double dt, std::vector<double>& h) const {
        const int n = (int)std::ceil(14.0 * m_tau / dt) + 1;
        h.assign(n, 0.0);
        double m = 0.0;
        for (int k = 0; k < n; ++k) {
            const double x = k * dt / m_tau;
            h[k] = x * x * std::exp(-2.0 * x);
            if (h[k] > m) m = h[k];
        }
        for (int k = 0; k < n; ++k) h[k] /= m;
    }
private:
    double m_tau;
};

// forme a sous-oscillation : h ~ x exp(-x) (1 - 0.3 x)
class BipolarTestResponse : public PreampResponse {
public:
    explicit BipolarTestResponse(double tau) : m_tau(tau) {}
    std::string name() const { return "test_bipolar"; }
    std::string describe() const { return "x exp(-x) (1 - 0.3 x), test seulement"; }
    void sample(double dt, std::vector<double>& h) const {
        const int n = (int)std::ceil(16.0 * m_tau / dt) + 1;
        h.assign(n, 0.0);
        double m = 0.0;
        for (int k = 0; k < n; ++k) {
            const double x = k * dt / m_tau;
            h[k] = x * std::exp(-x) * (1.0 - 0.3 * x);
            if (h[k] > m) m = h[k];
        }
        for (int k = 0; k < n; ++k) h[k] /= m;
    }
private:
    double m_tau;
};

// ---------------------------------------------------------------------------
// utilitaires
// ---------------------------------------------------------------------------

static double rms_of(const std::vector<double>& v)
{
    double s = 0.0;
    for (size_t i = 0; i < v.size(); ++i) s += v[i] * v[i];
    return std::sqrt(s / (double)v.size());
}

// autocorrelation empirique normalisee a lag m, sur n_try realisations ; add_noise ajoute le bruit voulu
static void mc_autocorr(const NoiseModel& M, int n_try, int n_bins, const int* lags, int n_lags,
                        std::vector<double>& rho, double& rms)
{
    GaussRng rng(12345);
    std::vector<double> num(n_lags, 0.0);
    double den = 0.0;
    long   cnt0 = 0;
    std::vector<long> cnt(n_lags, 0);
    for (int t = 0; t < n_try; ++t) {
        std::vector<double> out(n_bins, 0.0);
        M.add_filtered_noise(rng, out);
        for (int k = 0; k < n_bins; ++k) den += out[k] * out[k];
        cnt0 += n_bins;
        for (int a = 0; a < n_lags; ++a) {
            const int m = lags[a];
            for (int k = 0; k + m < n_bins; ++k) { num[a] += out[k] * out[k + m]; ++cnt[a]; }
        }
    }
    const double r0 = den / (double)cnt0;
    rms = std::sqrt(r0);
    rho.assign(n_lags, 0.0);
    for (int a = 0; a < n_lags; ++a) rho[a] = (num[a] / (double)cnt[a]) / r0;
}

static double theory_rho(const std::vector<double>& a, int m)       // sum a_j a_(j+m) / sum a_j^2
{
    double s = 0.0, s0 = 0.0;
    for (size_t j = 0; j < a.size(); ++j) s0 += a[j] * a[j];
    for (size_t j = 0; j + (size_t)m < a.size(); ++j) s += a[j] * a[j + m];
    return s / s0;
}

// ---------------------------------------------------------------------------
// T0 : generateur
// ---------------------------------------------------------------------------
static void test_rng()
{
    std::printf("T0 generateur (Box-Muller sur mt19937_64)\n");
    GaussRng r(7);
    const int N = 2000000;
    double s1 = 0.0, s2 = 0.0, s4 = 0.0, umin = 1.0, umax = 0.0;
    for (int i = 0; i < N; ++i) {
        const double z = r.normal();
        s1 += z; s2 += z * z; s4 += z * z * z * z;
    }
    for (int i = 0; i < N; ++i) {
        const double u = r.uniform();
        if (u < umin) umin = u;
        if (u > umax) umax = u;
    }
    const double mean = s1 / N, var = s2 / N - mean * mean, kurt = (s4 / N) / (var * var);
    check_abs("normale : moyenne", mean, 0.0, 5.0e-3);
    check_rel("normale : variance", var, 1.0, 5.0e-3);
    check_rel("normale : kurtosis (3 pour une gaussienne)", kurt, 3.0, 2.0e-2);
    check(umin > 0.0 && umax < 1.0, "uniforme : strictement dans ]0,1[", umin, umax, 0.0);

    GaussRng a(99), b(99), c(100);
    bool same = true, differ = false;
    for (int i = 0; i < 1000; ++i) {
        const double x = a.normal(), y = b.normal(), z = c.normal();
        if (x != y) same = false;
        if (x != z) differ = true;
    }
    check(same, "meme graine : memes tirages", same ? 1.0 : 0.0, 1.0, 0.0);
    check(differ, "graines differentes : tirages differents", differ ? 1.0 : 0.0, 1.0, 0.0);

    int cnt[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 100000; ++i) ++cnt[r.uniform_int(10)];
    bool uni = true;
    for (int i = 0; i < 10; ++i) if (std::fabs(cnt[i] - 10000.0) > 400.0) uni = false;
    check(uni, "uniform_int(10) : 10 classes equilibrees a 4 %", uni ? 1.0 : 0.0, 1.0, 0.0);
}

// ---------------------------------------------------------------------------
// tests pour une reponse donnee, a un pas dt donne
// ---------------------------------------------------------------------------
static void test_response(const PreampResponse& R, double dt, int adc_ratio, int n_try)
{
    std::printf("\n=== reponse : %s (%s), dt = %g ns, CAN = %g ns ===\n",
                R.name().c_str(), R.describe().c_str(), dt, adc_ratio * dt);
    std::vector<double> h;
    R.sample(dt, h);
    const int n_bins = 400;

    // ---- T1 : impulsion ----
    {
        NoiseParams p; p.dt = dt; p.adc_ratio = adc_ratio;
        NoiseModel M(p, h);
        std::vector<double> sig(n_bins, 0.0);
        const double Q = 12345.0;
        sig[3] = Q;
        std::vector<double> out;
        M.shape(sig, n_bins, out);
        double pk = 0.0, dmax = 0.0;
        for (int k = 0; k < n_bins; ++k) {
            if (out[k] > pk) pk = out[k];
            const int j = k - 3;
            const double ref = (j >= 0 && j < (int)h.size()) ? Q * h[j] : 0.0;
            if (std::fabs(out[k] - ref) > dmax) dmax = std::fabs(out[k] - ref);
        }
        check_rel("T1 pic de sortie pour une charge Q injectee d'un coup", pk, Q, 1.0e-9);
        check_abs("T1 shape(delta) = Q h (ecart max)", dmax, 0.0, 1.0e-9);
        // impulsion en fin de fenetre : pas de depassement de tableau, sortie tronquee
        std::vector<double> sig2(n_bins, 0.0);
        sig2[n_bins - 2] = 1.0;
        M.shape(sig2, n_bins, out);
        check_abs("T1 impulsion en fin de fenetre : out[n-2] = h[0]", out[n_bins - 2], h[0], 1.0e-12);
        check_abs("T1 impulsion en fin de fenetre : out[n-1] = h[1]", out[n_bins - 1], h[1], 1.0e-12);
    }

    // ---- T2 : bruit de sortie rms = ENC ----
    const double enc = 2000.0;
    struct Case { const char* name; double fp, fs, fc; };
    const Case cases[4] = { {"parallele seul", 1, 0, 0}, {"serie seul", 0, 1, 0},
                            {"CAN seul", 0, 0, 1},       {"melange 0.475/0.475/0.05", 0.475, 0.475, 0.05} };
    for (int c = 0; c < 4; ++c) {
        NoiseParams p; p.dt = dt; p.adc_ratio = adc_ratio; p.enc_total = enc;
        p.f_par = cases[c].fp; p.f_ser = cases[c].fs; p.f_can = cases[c].fc;
        NoiseModel M(p, h);
        GaussRng rng(2024 + c);
        std::vector<double> all;                         // echantillons CAN de toutes les realisations
        std::vector<double> allbin;                      // sortie preampli a dt (avant CAN)
        for (int t = 0; t < n_try; ++t) {
            std::vector<double> out(n_bins, 0.0);
            M.add_filtered_noise(rng, out);
            for (int k = 0; k < n_bins; ++k) allbin.push_back(out[k]);
            std::vector<double> smp;
            M.decimate(out, rng.uniform_int(adc_ratio), smp);
            M.add_adc_noise(rng, smp);
            for (size_t k = 0; k < smp.size(); ++k) all.push_back(smp[k]);
        }
        std::string lab = std::string("T2 rms de sortie, ") + cases[c].name;
        if (c == 2) {
            // le CAN seul : pas de bruit avant l'echantillonnage
            check_rel(lab + " (apres CAN)", rms_of(all), enc, 0.01);
        } else {
            check_rel(lab + " (apres CAN)", rms_of(all), enc, 0.02);
            // sortie du preampli seule : ENC_par et ENC_ser, c'est-a-dire sans la part CAN
            const double fsum = cases[c].fp + cases[c].fs + cases[c].fc;
            const double exp_pre = enc * std::sqrt((cases[c].fp + cases[c].fs) / fsum);
            check_rel(lab + " (sortie preampli, sans CAN)", rms_of(allbin), exp_pre, 0.02);
        }
    }

    // ---- T3 : pas de composante continue pour le bruit serie ----
    {
        NoiseParams p; p.dt = dt; p.adc_ratio = adc_ratio;
        NoiseModel M(p, h);
        check_abs("T3 somme (h[j]-h[j-1]) = 0", M.sum_dh(), 0.0, 1.0e-12);
    }

    // ---- T4 : autocorrelation ----
    {
        const int lags[5] = {1, 3, 6, 10, 20};
        for (int comp = 0; comp < 2; ++comp) {
            NoiseParams p; p.dt = dt; p.adc_ratio = adc_ratio; p.enc_total = enc;
            p.f_par = (comp == 0) ? 1.0 : 0.0;
            p.f_ser = (comp == 1) ? 1.0 : 0.0;
            p.f_can = 0.0;
            NoiseModel M(p, h);
            std::vector<double> rho; double rms = 0.0;
            mc_autocorr(M, n_try, n_bins, lags, 5, rho, rms);
            // coefficients de la theorie : h pour le parallele, h[j] - h[j-1] pour le serie
            std::vector<double> a;
            if (comp == 0) a = h;
            else { a.assign(h.size() + 1, 0.0); for (size_t j = 0; j <= h.size(); ++j) a[j] = (j < h.size() ? h[j] : 0.0) - (j > 0 ? h[j - 1] : 0.0); }
            for (int i = 0; i < 5; ++i) {
                char lab[128];
                std::snprintf(lab, sizeof lab, "T4 autocorrelation %s, decalage %d bins", comp == 0 ? "parallele" : "serie", lags[i]);
                check_abs(lab, rho[i], theory_rho(a, lags[i]), 0.02);
            }
        }
    }

    // ---- T5 : reproductibilite ----
    {
        NoiseParams p; p.dt = dt; p.adc_ratio = adc_ratio;
        NoiseModel M(p, h);
        GaussRng r1(555), r2(555), r3(556);
        std::vector<double> o1(n_bins, 0.0), o2(n_bins, 0.0), o3(n_bins, 0.0);
        M.add_filtered_noise(r1, o1); M.add_filtered_noise(r2, o2); M.add_filtered_noise(r3, o3);
        bool same = true, differ = false;
        for (int k = 0; k < n_bins; ++k) { if (o1[k] != o2[k]) same = false; if (o1[k] != o3[k]) differ = true; }
        check(same, "T5 meme graine : meme bruit filtre", same ? 1.0 : 0.0, 1.0, 0.0);
        check(differ, "T5 graine differente : bruit different", differ ? 1.0 : 0.0, 1.0, 0.0);
    }
}

// ---------------------------------------------------------------------------
// T6 : le meme h physique a deux pas dt
// ---------------------------------------------------------------------------
static void test_dt_scaling(int n_try)
{
    std::printf("\n=== T6 : meme reponse physique a dt = 10 ns et dt = 5 ns ===\n");
    GaussTestResponse R(190.0);
    std::vector<double> h10, h5;
    R.sample(10.0, h10); R.sample(5.0, h5);
    NoiseParams p10; p10.dt = 10.0; p10.adc_ratio = 2;
    NoiseParams p5;  p5.dt = 5.0;   p5.adc_ratio = 4;
    NoiseModel M10(p10, h10), M5(p5, h5);
    check_rel("T6 sigma_par(5)/sigma_par(10) = 1/sqrt(2)", M5.sigma_par() / M10.sigma_par(), 1.0 / std::sqrt(2.0), 0.02);
    check_rel("T6 sigma_ser(5)/sigma_ser(10) = sqrt(2)",   M5.sigma_ser() / M10.sigma_ser(), std::sqrt(2.0), 0.02);

    // le bruit de sortie (apres CAN de 20 ns) est le meme : rms et autocorrelation a 20 ns
    std::vector<double> rms(2, 0.0), rho20(2, 0.0);
    const NoiseModel* Ms[2] = { &M10, &M5 };
    const int nb[2] = { 400, 800 };
    for (int i = 0; i < 2; ++i) {
        GaussRng rng(31 + i);
        double s = 0.0, s20 = 0.0; long n = 0, n20 = 0;
        for (int t = 0; t < n_try; ++t) {
            std::vector<double> out(nb[i], 0.0), smp;
            Ms[i]->add_filtered_noise(rng, out);
            Ms[i]->decimate(out, 0, smp);
            for (size_t k = 0; k < smp.size(); ++k) { s += smp[k] * smp[k]; ++n; }
            for (size_t k = 0; k + 1 < smp.size(); ++k) { s20 += smp[k] * smp[k + 1]; ++n20; }
        }
        rms[i]   = std::sqrt(s / n);
        rho20[i] = (s20 / n20) / (s / n);
    }
    check_rel("T6 rms apres CAN 20 ns : dt = 5 ns / dt = 10 ns", rms[1] / rms[0], 1.0, 0.02);
    check_abs("T6 autocorrelation a un echantillon de 20 ns : dt = 5 ns - dt = 10 ns", rho20[1] - rho20[0], 0.0, 0.02);
}

// ---------------------------------------------------------------------------
// T7 : decimation ; T8 : refus
// ---------------------------------------------------------------------------
static void test_decimate_and_errors()
{
    std::printf("\n=== T7 decimation, T8 refus ===\n");
    GaussTestResponse R(190.0);
    std::vector<double> h;
    R.sample(10.0, h);
    NoiseParams p; p.dt = 10.0; p.adc_ratio = 2;
    NoiseModel M(p, h);
    std::vector<double> in(7), out;
    for (int i = 0; i < 7; ++i) in[i] = (double)i;
    M.decimate(in, 0, out);
    check(out.size() == 4 && out[0] == 0 && out[1] == 2 && out[3] == 6, "T7 phase 0 : 0 2 4 6", (double)out.size(), 4.0, 0.0);
    M.decimate(in, 1, out);
    check(out.size() == 3 && out[0] == 1 && out[2] == 5, "T7 phase 1 : 1 3 5", (double)out.size(), 3.0, 0.0);

    bool thrown = false;
    std::vector<double> bad = h;
    for (size_t i = 0; i < bad.size(); ++i) bad[i] *= 0.9;           // max = 0.9 : mal normalisee
    try { NoiseModel Mb(p, bad); } catch (const std::exception&) { thrown = true; }
    check(thrown, "T8 reponse non normalisee a 1 : refusee", thrown ? 1.0 : 0.0, 1.0, 0.0);

    thrown = false;
    std::vector<double> cut(h.begin(), h.begin() + h.size() / 2);     // tronquee en plein pic de decroissance
    double m = 0.0; for (size_t i = 0; i < cut.size(); ++i) if (cut[i] > m) m = cut[i];
    for (size_t i = 0; i < cut.size(); ++i) cut[i] /= m;
    try { NoiseModel Mc(p, cut); } catch (const std::exception&) { thrown = true; }
    check(thrown, "T8 reponse tronquee : refusee", thrown ? 1.0 : 0.0, 1.0, 0.0);

    thrown = false;
    NoiseParams q = p; q.f_par = -1.0;
    try { NoiseModel Md(q, h); } catch (const std::exception&) { thrown = true; }
    check(thrown, "T8 fraction negative : refusee", thrown ? 1.0 : 0.0, 1.0, 0.0);

    // enc_total = 0 : aucun bruit, aucune erreur
    NoiseParams z = p; z.enc_total = 0.0;
    NoiseModel Mz(z, h);
    GaussRng rng(1);
    std::vector<double> o(50, 0.0);
    Mz.add_filtered_noise(rng, o);
    check_abs("T8 enc_total = 0 : sortie nulle", rms_of(o), 0.0, 0.0);

    // normalisation des fractions : (2,2,0.2) equivaut a (0.476,0.476,0.048)
    NoiseParams a = p; a.f_par = 2.0; a.f_ser = 2.0; a.f_can = 0.2;
    NoiseParams b = p; b.f_par = 0.4761904761904762; b.f_ser = 0.4761904761904762; b.f_can = 0.047619047619047616;
    NoiseModel Ma(a, h), Mb(b, h);
    check_rel("T8 fractions normalisees : sigma_par", Ma.sigma_par(), Mb.sigma_par(), 1.0e-12);
    check_rel("T8 fractions normalisees : sigma_can", Ma.sigma_can(), Mb.sigma_can(), 1.0e-12);
}

int main(int argc, char* argv[])
{
    const int n_try = (argc > 1) ? std::atoi(argv[1]) : 2000;
    std::printf("test_noise : %d realisations par cas\n\n", n_try);

    test_rng();

    GaussTestResponse g190(190.0);
    test_response(g190, 10.0, 2, n_try);

    GaussTestResponse g100(100.0);
    test_response(g100, 5.0, 4, n_try);

    RcRcTestResponse rc(120.0);
    test_response(rc, 10.0, 2, n_try);

    BipolarTestResponse bp(100.0);
    test_response(bp, 10.0, 2, n_try);

    test_dt_scaling(n_try);
    test_decimate_and_errors();

    std::printf("\n%d verifications reussies, %d echecs\n", g_ok, g_fail);
    return g_fail == 0 ? 0 : 1;
}
