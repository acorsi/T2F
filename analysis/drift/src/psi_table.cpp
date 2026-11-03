#include "psi_table.hpp"
#include "psi_builder.hpp"
#include <cmath>
#include <stdexcept>

namespace {

// Poids de l'interpolation de Lagrange à 4 points, pour les noeuds
// -1, 0, 1, 2 et la position t dans [0, 1].
void lagrange4(double t, double* w) {
    w[0] = -t * (t - 1.0) * (t - 2.0) / 6.0;
    w[1] =  (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0;
    w[2] = -(t + 1.0) * t * (t - 2.0) / 2.0;
    w[3] =  (t + 1.0) * t * (t - 1.0) / 6.0;
}

} // namespace

TableParams default_table_params(const DetectorConfig& cfg, double pitch) {
    TableParams tp;
    tp.a        = 0.5 * pitch;
    tp.g        = cfg.dga;
    tp.ve_kg    = cfg.ve_kg;
    tp.ve_ga    = cfg.ve_ga;
    tp.DT_kg    = cfg.DT_kg;
    tp.DT_ga    = cfg.DT_ga;
    tp.DL_kg    = cfg.DL_kg;
    tp.DL_ga    = cfg.DL_ga;
    tp.d_max    = cfg.dkg;
    tp.n_d      = 10;
    tp.n_out    = 100;
    tp.step_out = pitch / 16.0;
    tp.n_h      = 100;
    tp.tau_max_ns = 400.0;
    tp.n_img    = 10;
    tp.m_cells  = 8;
    tp.r_src    = 2.2;
    return tp;
}

double PsiTable::sigma_T(double d) const {
    return m_tp.DT_kg * std::sqrt(d);
}

// Etalement temporel : diffusion longitudinale dans la zone de dérive, plus
// (en quadrature) celle du gap, évaluée sur toute l'épaisseur g.
double PsiTable::sigma_t(double d) const {
    double s_kg = m_tp.DL_kg * std::sqrt(d) / m_tp.ve_kg;
    double s_ga = m_tp.DL_ga * std::sqrt(m_tp.g) / m_tp.ve_ga;
    return std::sqrt(s_kg * s_kg + s_ga * s_ga);
}

double PsiTable::d_node(int k) const {
    double x  = m_x0 + k * m_dx;
    double d  = (std::exp(2.0 * x) - m_c0) / (m_tp.DT_kg * m_tp.DT_kg);
    return (d > 0.0) ? d : 0.0;
}

PsiTable::PsiTable(const TableParams& tp)
    : m_tp(tp), m_dtau(0.0), m_min(0), m_max(0), m_c0(0.0), m_x0(0.0), m_dx(0.0)
{
    if (tp.n_d < 2) throw std::runtime_error("PsiTable : n_d doit etre >= 2");

    m_dtau = tp.g / (tp.ve_ga * tp.n_h);

    // Plage en tau : on descend jusqu'à -3.5 sigma_t (là Psi < 1e-6) et on monte
    // jusqu'à tau_max_ns, ou plus loin si le plateau n'est pas atteint.
    double st_max = sigma_t(tp.d_max);
    int jmax = (int)std::ceil(5.0 * st_max / m_dtau);
    m_min = -(int)std::ceil(3.5 * st_max / m_dtau);
    m_max = (int)std::ceil(tp.tau_max_ns / m_dtau);
    if (m_max < tp.n_h + jmax + 2) m_max = tp.n_h + jmax + 2;

    // Noeuds en d : équidistants en x = ln(sigma_anode(d)).
    // Le plancher (une demi-cellule de la source) évite ln(0) si DT_ga = 0.
    double plancher = 0.5 * tp.a / tp.m_cells;
    m_c0 = tp.DT_ga * tp.DT_ga * tp.g + plancher * plancher;
    m_x0 = 0.5 * std::log(m_c0);
    double x_max = 0.5 * std::log(tp.DT_kg * tp.DT_kg * tp.d_max + m_c0);
    m_dx = (x_max - m_x0) / (tp.n_d - 1);

    const int n_tau = m_max - m_min + 1;
    const size_t taille_d = (size_t)tp.n_out * tp.n_out * n_tau;
    m_data.assign((size_t)tp.n_d * taille_d, 0.0f);

    PhiSource src(tp.a, tp.g, tp.n_img, tp.m_cells, tp.r_src, tp.n_h);

    std::vector<double> psi;
    for (int k = 0; k < tp.n_d; ++k) {
        double d = d_node(k);

        PsiParams P;
        P.sigma0   = sigma_T(d);
        P.DT_ga    = tp.DT_ga;
        P.sigma_t  = sigma_t(d);
        P.dtau     = m_dtau;
        P.ve_ga    = tp.ve_ga;
        P.m_min    = m_min;
        P.m_max    = m_max;
        P.n_out    = tp.n_out;
        P.step_out = tp.step_out;
        build_psi(src, P, psi);

        float* dest = &m_data[(size_t)k * taille_d];
        for (size_t q = 0; q < taille_d; ++q) dest[q] = (float)psi[q];
    }
}

void PsiTable::psi_vector(double u, double v, double d, double* out) const {
    const int n_tau = m_max - m_min + 1;
    const int n_out = m_tp.n_out;
    for (int n = 0; n < n_tau; ++n) out[n] = 0.0;

    // --- position en d : cubique en x = ln(sigma_anode) ---
    if (d < 0.0) d = 0.0;
    if (d > m_tp.d_max) d = m_tp.d_max;         // au-delà de d_max : dernière valeur
    double xs = (0.5 * std::log(m_tp.DT_kg * m_tp.DT_kg * d + m_c0) - m_x0) / m_dx;
    // Stencil de 4 noeuds consécutifs b-1 ... b+2. Aux deux extrémités on décale le
    // stencil (b entre 1 et n_d-3) : on interpole toujours, on n'extrapole jamais.
    int nd = m_tp.n_d;
    int b = (int)xs;
    if (b < 1) b = 1;
    if (b > nd - 3) b = nd - 3;
    double wd[4];
    lagrange4(xs - b, wd);                      // t peut sortir de [0,1] de façon contrôlée

    // --- position en u et v (symétrie : on prend les valeurs absolues) ---
    double xu = std::fabs(u) / m_tp.step_out;
    double xv = std::fabs(v) / m_tp.step_out;
    int i0 = (int)xu;
    int j0 = (int)xv;
    if (i0 >= n_out || j0 >= n_out) return;     // hors table : Psi = 0
    double wu[4], wv[4];
    lagrange4(xu - i0, wu);
    lagrange4(xv - j0, wv);

    const size_t taille_d = (size_t)n_out * n_out * n_tau;

    for (int kk = 0; kk < 4; ++kk) {
        const float* base = &m_data[(size_t)(b - 1 + kk) * taille_d];
        for (int a = 0; a < 4; ++a) {
            int i = i0 - 1 + a;
            if (i < 0) i = -i;                  // miroir : noeud -1 = noeud 1
            if (i >= n_out) continue;           // au-delà de la table : valeur nulle
            for (int b = 0; b < 4; ++b) {
                int j = j0 - 1 + b;
                if (j < 0) j = -j;
                if (j >= n_out) continue;
                double w = wd[kk] * wu[a] * wv[b];
                const float* p = base + ((size_t)i * n_out + j) * n_tau;
                for (int n = 0; n < n_tau; ++n) out[n] += w * p[n];
            }
        }
    }
}

double PsiTable::psi_at(const double* vec, double tau) const {
    const int n_tau = m_max - m_min + 1;
    double x = tau / m_dtau - m_min;             // position en indices de noeuds
    if (x <= 0.0)         return vec[0];         // avant la plage : ~ 0
    if (x >= n_tau - 1)   return vec[n_tau - 1]; // après la plage : plateau

    int n0 = (int)x;
    double w[4];
    lagrange4(x - n0, w);
    double s = 0.0;
    for (int a = 0; a < 4; ++a) {
        int n = n0 - 1 + a;
        if (n < 0) n = 0;                        // bords : on répète le dernier noeud
        if (n > n_tau - 1) n = n_tau - 1;
        s += w[a] * vec[n];
    }
    return s;
}
