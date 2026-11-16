#include "ghost.hpp"

#include <cmath>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Gaussienne tronquee z > t
// ---------------------------------------------------------------------------
double GhostModel::normal_above(GaussRng& rng, double t)
{
    if (t < 0.5) {
        // acceptation >= 30 % : rejet simple
        for (;;) {
            const double z = rng.normal();
            if (z > t) return z;
        }
    }
    // t >= 0.5 : exponentielle translatee avec rejet (Robert, 1995)
    const double lambda = 0.5 * (t + std::sqrt(t * t + 4.0));
    for (;;) {
        const double z = t - std::log(rng.uniform()) / lambda;
        const double d = z - lambda;
        if (rng.uniform() <= std::exp(-0.5 * d * d)) return z;
    }
}

int GhostModel::first_run(const std::vector<double>& smp, double thr, int n_consec)
{
    int run = 0;
    const int ns = (int)smp.size();
    for (int j = 0; j < ns; ++j) {
        if (smp[j] > thr) { if (++run >= n_consec) return j - n_consec + 1; }
        else run = 0;
    }
    return -1;
}

// ---------------------------------------------------------------------------
GhostModel::GhostModel(const NoiseModel& noise, const GhostParams& gp, double thr, int n_consec, int n_bins)
    : m_noise(&noise), m_gp(gp), m_thr(thr), m_n_consec(n_consec), m_n_bins(n_bins),
      m_ns(0), m_win(1), m_c0(0.0), m_q(0.0), m_p(0.0), m_p_err(0.0)
{
    if (n_bins < 1)          throw std::runtime_error("GhostModel : n_bins doit etre >= 1");
    if (n_consec < 1)        throw std::runtime_error("GhostModel : n_consec doit etre >= 1");
    if (thr < 0.0)           throw std::runtime_error("GhostModel : seuil < 0");
    if (gp.window_ns < 0.0)  throw std::runtime_error("GhostModel : ghost_window_ns doit etre >= 0");
    if (gp.calib_trials < 100)
                             throw std::runtime_error("GhostModel : ghost_calib_trials doit etre >= 100");

    const NoiseParams& np = noise.params();
    const int R = np.adc_ratio;
    const std::vector<double>& h = noise.h();
    const int L = (int)h.size();

    // h[j] - h[j-1], j = 0 .. L
    std::vector<double> hd(L + 1, 0.0);
    for (int j = 0; j <= L; ++j) {
        const double cur  = (j < L) ? h[j] : 0.0;
        const double prev = (j > 0) ? h[j - 1] : 0.0;
        hd[j] = cur - prev;
    }

    m_c0 = np.enc_total * np.enc_total;
    const double sp2 = noise.sigma_par() * noise.sigma_par();
    const double ss2 = noise.sigma_ser() * noise.sigma_ser();
    m_c.clear();
    m_c.push_back(m_c0);
    for (int k = 1; k * R <= L; ++k) {
        double a = 0.0, b = 0.0;
        for (int j = 0; j + k * R < L; ++j)      a += h[j] * h[j + k * R];
        for (int j = 0; j + k * R <= L; ++j)     b += hd[j] * hd[j + k * R];
        m_c.push_back(sp2 * a + ss2 * b);
    }

    m_ns  = (n_bins + R - 1) / R;
    m_win = (int)std::floor(gp.window_ns / (R * np.dt) + 0.5);
    if (m_win < 1) m_win = 1;

    if (m_c0 > 0.0) {
        const double t = thr / std::sqrt(m_c0);
        m_q = 0.5 * std::erfc(t / std::sqrt(2.0));
    }
}

// ---------------------------------------------------------------------------
int GhostModel::propose(GaussRng& rng, int phase, std::vector<double>& smp, bool& pass, int& j_first,
                         double scale) const
{
    if (!(scale > 0.0)) throw std::runtime_error("GhostModel::propose : scale doit etre > 0");
    const double thr = m_thr / scale;           // seuil effectif, dans l'echelle du bruit de reference
    if (m_c0 <= 0.0) throw std::runtime_error("GhostModel::propose : bruit nul, pas de fantome possible");

    // 1. bruit non conditionne : meme chaine que les pads du signal
    std::vector<double> bins(m_n_bins, 0.0);
    m_noise->add_filtered_noise(rng, bins);
    m_noise->decimate(bins, phase, smp);
    m_noise->add_adc_noise(rng, smp);
    const int ns = (int)smp.size();
    if (ns < 1) throw std::runtime_error("GhostModel::propose : fenetre vide");

    // 2. echantillon j* impose a a > seuil ; les autres suivent par conditionnement gaussien
    const double s0 = std::sqrt(m_c0);
    const int    js = rng.uniform_int(ns);
    const double a  = s0 * normal_above(rng, thr / s0);
    const double delta = a - smp[js];
    const int    K  = (int)m_c.size() - 1;
    for (int j = js - K; j <= js + K; ++j) {
        if (j < 0 || j >= ns) continue;
        const int lag = (j > js) ? (j - js) : (js - j);
        smp[j] += delta * m_c[lag] / m_c0;
    }
    smp[js] = a;                         // evite tout arrondi : a > seuil exactement

    int nabove = 0;
    for (int j = 0; j < ns; ++j) if (smp[j] > thr) ++nabove;
    j_first = first_run(smp, thr, m_n_consec);
    pass = (j_first >= 0);
    return nabove;
}

int GhostModel::draw(GaussRng& rng, int phase, std::vector<double>& smp, int& j_first, double scale) const
{
    for (int it = 1; it <= 1000000; ++it) {
        bool pass = false;
        const int K = propose(rng, phase, smp, pass, j_first, scale);
        if (pass && K >= 1 && rng.uniform() * K < 1.0) {               // acceptation avec la probabilite 1/K
            if (scale != 1.0) for (size_t j = 0; j < smp.size(); ++j) smp[j] *= scale;
            return it;
        }
    }
    throw std::runtime_error("GhostModel::draw : acceptation introuvable (seuil ou n_consec trop severes ?)");
}

// ---------------------------------------------------------------------------
double GhostModel::prob_for_scale(double scale) const
{
    if (!(scale > 0.0)) throw std::runtime_error("GhostModel::prob_for_scale : scale doit etre > 0");
    if (m_gp.prob >= 0.0 || scale == 1.0 || m_c0 <= 0.0 || m_q <= 0.0) return m_p;
    if (!m_ps.empty()) {
        const std::map<long, double>::const_iterator it = m_ps.find((long)std::floor(scale * 1.0e4 + 0.5));
        if (it != m_ps.end()) return it->second;
    }
    const double t = m_thr / (scale * std::sqrt(m_c0));
    const double q = 0.5 * std::erfc(t / std::sqrt(2.0));
    double p = m_p * q / m_q;
    if (p > 1.0) p = 1.0;
    return p;
}

// ---------------------------------------------------------------------------
void GhostModel::calibrate_scales(GaussRng& rng, const std::vector<double>& scales)
{
    m_ps.clear();
    if (!m_gp.enable || m_gp.prob >= 0.0 || m_c0 <= 0.0) return;
    std::map<long, double> todo;
    for (size_t i = 0; i < scales.size(); ++i) {
        if (!(scales[i] > 0.0)) throw std::runtime_error("GhostModel::calibrate_scales : scale doit etre > 0");
        todo[(long)std::floor(scales[i] * 1.0e4 + 0.5)] = 0.0;
    }
    const int N = m_gp.calib_trials;
    std::vector<double> smp;
    for (std::map<long, double>::iterator it = todo.begin(); it != todo.end(); ++it) {
        const double s = (double)it->first * 1.0e-4;
        if (it->first == 10000) { it->second = m_p; continue; }
        const double t = m_thr / (s * std::sqrt(m_c0));
        const double q = 0.5 * std::erfc(t / std::sqrt(2.0));
        double s1 = 0.0;
        for (int n = 0; n < N; ++n) {
            bool pass = false; int jf = 0;
            const int K = propose(rng, 0, smp, pass, jf, s);
            if (pass && K >= 1) s1 += 1.0 / (double)K;
        }
        double p = (double)m_ns * q * s1 / (double)N;
        if (p > 1.0) p = 1.0;
        it->second = p;
    }
    m_ps = todo;
}

// ---------------------------------------------------------------------------
void GhostModel::calibrate(GaussRng& rng)
{
    m_p = 0.0; m_p_err = 0.0;
    if (!m_gp.enable || m_c0 <= 0.0) return;
    if (m_gp.prob >= 0.0) { m_p = m_gp.prob; return; }

    const int N = m_gp.calib_trials;
    double s1 = 0.0, s2 = 0.0;
    std::vector<double> smp;
    for (int n = 0; n < N; ++n) {
        bool pass = false; int jf = 0;
        const int K = propose(rng, 0, smp, pass, jf);
        const double w = (pass && K >= 1) ? 1.0 / (double)K : 0.0;
        s1 += w; s2 += w * w;
    }
    const double mean = s1 / N;
    double var = s2 / N - mean * mean;
    if (var < 0.0) var = 0.0;
    const double f = (double)m_ns * m_q;
    m_p     = f * mean;
    m_p_err = f * std::sqrt(var / N);
}
