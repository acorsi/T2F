#include "noise.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// GaussRng
// ---------------------------------------------------------------------------

double GaussRng::uniform() {
    // 53 bits de mantisse, decales d'une demi-unite : resultat dans ]0,1[
    const uint64_t u = m_eng() >> 11;
    return ((double)u + 0.5) * (1.0 / 9007199254740992.0);
}

double GaussRng::normal() {
    if (m_has) {
        m_has = false;
        return m_spare;
    }
    const double u1 = uniform();
    const double u2 = uniform();
    const double r  = std::sqrt(-2.0 * std::log(u1));
    const double a  = 6.283185307179586476925287 * u2;
    m_spare = r * std::sin(a);
    m_has   = true;
    return r * std::cos(a);
}

int GaussRng::uniform_int(int n) {
    int k = (int)std::floor(uniform() * n);
    if (k >= n) k = n - 1;
    return k;
}

// ---------------------------------------------------------------------------
// NoiseModel
// ---------------------------------------------------------------------------

NoiseModel::NoiseModel(const NoiseParams& p, const std::vector<double>& h)
    : m_p(p), m_h(h)
{
    if (p.dt <= 0.0)            throw std::runtime_error("NoiseModel : dt doit etre > 0");
    if (p.adc_ratio < 1)        throw std::runtime_error("NoiseModel : adc_ratio doit etre >= 1");
    if (p.enc_total < 0.0)      throw std::runtime_error("NoiseModel : enc_total doit etre >= 0");
    if (p.f_par < 0.0 || p.f_ser < 0.0 || p.f_can < 0.0)
                                throw std::runtime_error("NoiseModel : les fractions f_par, f_ser, f_can doivent etre >= 0");
    const double fsum = p.f_par + p.f_ser + p.f_can;
    if (p.enc_total > 0.0 && fsum <= 0.0)
                                throw std::runtime_error("NoiseModel : f_par + f_ser + f_can doit etre > 0");
    if (m_h.size() < 2)         throw std::runtime_error("NoiseModel : la reponse h doit avoir au moins 2 echantillons");

    // verification de la normalisation et de la troncature de h
    double hmax = 0.0;
    for (size_t j = 0; j < m_h.size(); ++j) if (m_h[j] > hmax) hmax = m_h[j];
    if (std::fabs(hmax - 1.0) > 1.0e-6)
        throw std::runtime_error("NoiseModel : la reponse h doit etre normalisee a 1 au maximum");
    if (std::fabs(m_h[m_h.size() - 1]) > 1.0e-3)
        throw std::runtime_error("NoiseModel : la reponse h est tronquee (|h| en fin de tableau > 1e-3)");

    // h[j] - h[j-1] pour j = 0 .. L (h[-1] = h[L] = 0)
    const int L = (int)m_h.size();
    m_hd.assign(L + 1, 0.0);
    for (int j = 0; j <= L; ++j) {
        const double cur  = (j < L) ? m_h[j] : 0.0;
        const double prev = (j > 0) ? m_h[j - 1] : 0.0;
        m_hd[j] = cur - prev;
    }

    m_sum_h2 = 0.0; m_sum_dh2 = 0.0; m_sum_dh = 0.0;
    for (int j = 0; j < L; ++j)  m_sum_h2  += m_h[j] * m_h[j];
    for (int j = 0; j <= L; ++j) { m_sum_dh2 += m_hd[j] * m_hd[j]; m_sum_dh += m_hd[j]; }

    // ENC par composante et amplitudes par bin
    const double e2 = p.enc_total * p.enc_total;
    if (fsum > 0.0) {
        m_enc_par = std::sqrt(e2 * p.f_par / fsum);
        m_enc_ser = std::sqrt(e2 * p.f_ser / fsum);
        m_enc_can = std::sqrt(e2 * p.f_can / fsum);
    } else {
        m_enc_par = m_enc_ser = m_enc_can = 0.0;
    }
    m_sig_par = m_enc_par / std::sqrt(m_sum_h2);
    m_sig_ser = m_enc_ser / std::sqrt(m_sum_dh2);

    m_th.resize(L);
    m_thd.resize(L + 1);
    for (int j = 0; j < L; ++j)  m_th[j]  = m_sig_par * m_h[j];
    for (int j = 0; j <= L; ++j) m_thd[j] = m_sig_ser * m_hd[j];
}

void NoiseModel::shape(const std::vector<double>& sig, int n_out, std::vector<double>& out) const
{
    const int L    = (int)m_h.size();
    const int n_in = (int)sig.size();
    out.assign(n_out > 0 ? n_out : 0, 0.0);
    for (int i = 0; i < n_in; ++i) {
        const double s = sig[i];
        if (s == 0.0) continue;
        int jmax = n_out - i;           // j < n_out - i
        if (jmax > L) jmax = L;
        for (int j = 0; j < jmax; ++j) out[i + j] += s * m_h[j];
    }
}

void NoiseModel::add_filtered_noise(GaussRng& rng, std::vector<double>& out) const
{
    add_filtered_noise(rng, out, 1.0);
}

void NoiseModel::add_filtered_noise(GaussRng& rng, std::vector<double>& out, double ser_scale) const
{
    if (!(ser_scale >= 0.0)) throw std::runtime_error("NoiseModel::add_filtered_noise : ser_scale doit etre >= 0");
    const int n = (int)out.size();
    const int L = (int)m_h.size();
    if (n <= 0) return;
    const int ng = n + L;                // tirages aux indices i = -L .. n-1, stockes en i + L

    // parallele : bruit blanc par bin, filtre par h
    if (m_sig_par > 0.0) {
        std::vector<double> g(ng);
        for (int i = 0; i < ng; ++i) g[i] = rng.normal();
        for (int k = 0; k < n; ++k) {
            double s = 0.0;
            for (int j = 0; j < L; ++j) s += m_th[j] * g[k - j + L];
            out[k] += s;
        }
    }
    // serie : bruit blanc sur la charge cumulee c[i] ; courant = c[i] - c[i-1] ;
    // sortie = somme_j h[j] (c[k-j] - c[k-j-1]) = somme_j (h[j] - h[j-1]) c[k-j]
    if (m_sig_ser > 0.0) {
        std::vector<double> g(ng);
        for (int i = 0; i < ng; ++i) g[i] = rng.normal();
        for (int k = 0; k < n; ++k) {
            double s = 0.0;
            for (int j = 0; j <= L; ++j) s += m_thd[j] * g[k - j + L];
            out[k] += ser_scale * s;
        }
    }
}

void NoiseModel::decimate(const std::vector<double>& in, int phase, std::vector<double>& out) const
{
    out.clear();
    if (phase < 0) throw std::runtime_error("NoiseModel::decimate : phase < 0");
    for (size_t i = (size_t)phase; i < in.size(); i += (size_t)m_p.adc_ratio) out.push_back(in[i]);
}

void NoiseModel::add_adc_noise(GaussRng& rng, std::vector<double>& samples) const
{
    if (m_enc_can <= 0.0) return;
    for (size_t j = 0; j < samples.size(); ++j) samples[j] += m_enc_can * rng.normal();
}

std::string NoiseModel::summary() const
{
    std::ostringstream s;
    s << "bruit : enc_total = " << m_p.enc_total << " e-  (par " << m_enc_par << ", ser " << m_enc_ser
      << ", CAN " << m_enc_can << ")\n"
      << "  h : " << m_h.size() << " echantillons de " << m_p.dt << " ns,  sum h^2 = " << m_sum_h2
      << ",  sum (dh)^2 = " << m_sum_dh2 << "\n"
      << "  sigma_par = " << m_sig_par << " e- par bin (charge par bin),  sigma_ser = " << m_sig_ser
      << " e- par bin (charge cumulee),  sigma_can = " << m_enc_can << " e- par echantillon CAN (periode "
      << m_p.adc_ratio * m_p.dt << " ns)";
    return s.str();
}
