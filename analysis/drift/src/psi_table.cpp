#include "psi_table.hpp"
#include "psi_builder.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace {

// Poids de l'interpolation de Lagrange à 4 points, pour les noeuds
// -1, 0, 1, 2 et la position t dans [0, 1].
void lagrange4(double t, double* w) {
    w[0] = -t * (t - 1.0) * (t - 2.0) / 6.0;
    w[1] =  (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0;
    w[2] = -(t + 1.0) * t * (t - 2.0) / 2.0;
    w[3] =  (t + 1.0) * t * (t - 1.0) / 6.0;
}

// Accumulation flux par flux, taille fixe (le compilateur peut dérouler et garder acc en registres)
template<int NT>
void accumulate_fixed(const float* const* ptr, const float* wgt, int ns, float* out) {
    float acc[NT];
    for (int n = 0; n < NT; ++n) acc[n] = 0.0f;
    for (int s = 0; s < ns; ++s) {
        const float* p = ptr[s];
        const float  w = wgt[s];
        for (int n = 0; n < NT; ++n) acc[n] += w * p[n];
    }
    for (int n = 0; n < NT; ++n) out[n] = acc[n];
}

// Même chose pour un n_tau quelconque (<= 256)
void accumulate_generic(const float* const* ptr, const float* wgt, int ns, int n_tau, float* out) {
    float acc[256];
    for (int n = 0; n < n_tau; ++n) acc[n] = 0.0f;
    for (int s = 0; s < ns; ++s) {
        const float* p = ptr[s];
        const float  w = wgt[s];
        for (int n = 0; n < n_tau; ++n) acc[n] += w * p[n];
    }
    for (int n = 0; n < n_tau; ++n) out[n] = acc[n];
}

} // namespace

void table_time_range(const TableParams& tp, double dtau, int& m_min, int& m_max)
{
    // sigma_t a d_max : diffusion longitudinale cathode-grille et gap, en quadrature
    double s_kg = tp.DL_kg * std::sqrt(tp.d_max) / tp.ve_kg;
    double s_ga = tp.DL_ga * std::sqrt(tp.g) / tp.ve_ga;
    double st_max = std::sqrt(s_kg * s_kg + s_ga * s_ga);
    int jmax = (int)std::ceil(5.0 * st_max / dtau);
    m_min = -(int)std::ceil(3.5 * st_max / dtau);
    m_max = (int)std::ceil(tp.tau_max_ns / dtau);
    if (m_max < tp.n_h + jmax + 2) m_max = tp.n_h + jmax + 2;
}

void table_grid(const TableParams& tp, std::vector<double>& step, std::vector<int>& nk)
{
    step.assign(tp.n_d, tp.step_out);
    nk.assign(tp.n_d, tp.n_out);
    if (!tp.adaptive) return;

    // memes noeuds en d que PsiTable : equidistants en x = ln(sigma_anode)
    const double plancher = 0.5 * tp.a / tp.m_cells;
    const double c0 = tp.DT_ga * tp.DT_ga * tp.g + plancher * plancher;
    const double x0 = 0.5 * std::log(c0);
    const double x_max = 0.5 * std::log(tp.DT_kg * tp.DT_kg * tp.d_max + c0);
    const double dx = (tp.n_d > 1) ? (x_max - x0) / (tp.n_d - 1) : 0.0;
    const double U = (tp.n_out - 1) * tp.step_out;
    for (int k = 0; k < tp.n_d; ++k) {
        double s = tp.step_out * std::exp(x0 + k * dx - x0);       // step_out * sigma_k / sigma_0
        if (tp.step_cap > 0.0 && s > tp.step_cap) s = tp.step_cap;
        if (s < tp.step_out) s = tp.step_out;
        int n = (int)std::ceil(U / s - 1.0e-9) + 1;
        if (n < 4) n = 4;
        if (n > tp.n_out) n = tp.n_out;
        nk[k] = n;
        step[k] = U / (n - 1);
    }
}

double table_cells(const TableParams& tp)
{
    std::vector<double> step;
    std::vector<int> nk;
    table_grid(tp, step, nk);
    double s = 0.0;
    for (int k = 0; k < tp.n_d; ++k) s += (double)nk[k] * nk[k];
    return s;
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
    : m_tp(tp), m_dtau(0.0), m_min(0), m_max(0), m_c0(0.0), m_x0(0.0), m_dx(0.0), m_phi_edge(0.0)
{
    if (tp.n_d < 2) throw std::runtime_error("PsiTable : n_d doit etre >= 2");
    if (tp.R < 1)   throw std::runtime_error("PsiTable : R doit etre >= 1");
    if (tp.dtau <= 0.0) throw std::runtime_error("PsiTable : dtau doit etre > 0");

    m_dtau = tp.dtau;

    // Plage en tau : on descend jusqu'à -3.5 sigma_t (là Psi < 1e-6) et on monte
    // jusqu'à tau_max_ns, ou plus loin si le plateau n'est pas atteint.
    table_time_range(tp, m_dtau, m_min, m_max);

    // Noeuds en d : équidistants en x = ln(sigma_anode(d)).
    // Le plancher (une demi-cellule de la source) évite ln(0) si DT_ga = 0.
    double plancher = 0.5 * tp.a / tp.m_cells;
    m_c0 = tp.DT_ga * tp.DT_ga * tp.g + plancher * plancher;
    m_x0 = 0.5 * std::log(m_c0);
    double x_max = 0.5 * std::log(tp.DT_kg * tp.DT_kg * tp.d_max + m_c0);
    m_dx = (x_max - m_x0) / (tp.n_d - 1);

    const int n_tau = m_max - m_min + 1;

    // grille (u,v) de chaque noeud en d
    table_grid(tp, m_step, m_nk);
    m_off.assign(tp.n_d + 1, 0);
    m_offd.assign(tp.n_d + 1, 0);
    for (int k = 0; k < tp.n_d; ++k) {
        size_t cells = (size_t)m_nk[k] * m_nk[k];
        m_off[k + 1]  = m_off[k]  + cells * n_tau;
        m_offd[k + 1] = m_offd[k] + cells;
    }
    m_data.assign(m_off[tp.n_d], 0.0f);

    PhiSource src(tp.a, tp.g, tp.n_img, tp.m_cells, tp.r_src, tp.n_h, tp.ve_ga * m_dtau);
    m_phi_edge = src.edge_max();

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
        P.n_out    = m_nk[k];
        P.step_out = m_step[k];
        build_psi(src, P, psi);

        const size_t taille_d = (size_t)m_nk[k] * m_nk[k] * n_tau;
        float* dest = &m_data[m_off[k]];
        for (size_t q = 0; q < taille_d; ++q) dest[q] = (float)psi[q];
    }

    // carte des maxima : pour chaque noeud (k, i, j), max sur tau de |Psi(tau+dtau) - Psi(tau)|
    m_dmax.assign(m_offd[tp.n_d], 0.0f);
    for (int k = 0; k < tp.n_d; ++k) {
        const int ncell = m_nk[k] * m_nk[k];
        for (int q = 0; q < ncell; ++q) {
            const float* p = &m_data[m_off[k] + (size_t)q * n_tau];
            float mx = 0.0f;
            for (int n = 0; n + 1 < n_tau; ++n) {
                float a = p[n + 1] - p[n];
                if (a < 0.0f) a = -a;
                if (a > mx) mx = a;
            }
            m_dmax[m_offd[k] + q] = mx;
        }
    }
}

int PsiTable::d_cell(double d) const {
    if (d < 0.0) d = 0.0;
    if (d > m_tp.d_max) d = m_tp.d_max;
    double xs = (0.5 * std::log(m_tp.DT_kg * m_tp.DT_kg * d + m_c0) - m_x0) / m_dx;
    int k = (int)xs;
    if (k < 0) k = 0;
    if (k > m_tp.n_d - 2) k = m_tp.n_d - 2;
    return k;
}


double PsiTable::dpsi_max(double u, double v, int kd) const {
    float m = 0.0f;
    for (int kk = kd; kk <= kd + 1; ++kk) {
        const int n_out = m_nk[kk];
        int i0 = (int)(std::fabs(u) / m_step[kk]);
        int j0 = (int)(std::fabs(v) / m_step[kk]);
        for (int a = 0; a <= 1; ++a) {
            int i = i0 + a;
            if (i >= n_out) continue;
            for (int b = 0; b <= 1; ++b) {
                int j = j0 + b;
                if (j >= n_out) continue;
                float w = m_dmax[m_offd[kk] + (size_t)i * n_out + j];
                if (w > m) m = w;
            }
        }
    }
    // m = variation maximale sur UN pas de table ; un bin de lecture en couvre R
    return m * (float)m_tp.R;
}

// Liste des flux a combiner : pour chacun des 4 noeuds en d du stencil, 4 x 4 noeuds en (u,v)
// de la grille PROPRE a ce noeud en d (symetrie : valeurs absolues de u et v).
int PsiTable::flux_list(double u, double v, double d, const float** ptr, double* wgt) const {
    const int n_tau = m_max - m_min + 1;

    // --- position en d : cubique en x = ln(sigma_anode) ---
    if (d < 0.0) d = 0.0;
    if (d > m_tp.d_max) d = m_tp.d_max;         // au-dela de d_max : derniere valeur
    double xs = (0.5 * std::log(m_tp.DT_kg * m_tp.DT_kg * d + m_c0) - m_x0) / m_dx;
    // Stencil de 4 noeuds consecutifs b-1 ... b+2. Aux deux extremites on decale le
    // stencil (b entre 1 et n_d-3) : on interpole toujours, on n'extrapole jamais.
    int nd = m_tp.n_d;
    int b = (int)xs;
    if (b < 1) b = 1;
    if (b > nd - 3) b = nd - 3;
    double wd[4];
    lagrange4(xs - b, wd);                      // t peut sortir de [0,1] de facon controlee

    int ns = 0;
    for (int kk = 0; kk < 4; ++kk) {
        const int node  = b - 1 + kk;
        const int n_out = m_nk[node];
        const double xu = std::fabs(u) / m_step[node];
        const double xv = std::fabs(v) / m_step[node];
        const int i0 = (int)xu;
        const int j0 = (int)xv;
        if (i0 >= n_out || j0 >= n_out) continue;      // hors table : Psi = 0
        double wu[4], wv[4];
        lagrange4(xu - i0, wu);
        lagrange4(xv - j0, wv);
        const float* base = &m_data[m_off[node]];
        for (int a = 0; a < 4; ++a) {
            int i = i0 - 1 + a;
            if (i < 0) i = -i;                  // miroir : noeud -1 = noeud 1
            if (i >= n_out) continue;           // au-dela de la table : valeur nulle
            for (int c = 0; c < 4; ++c) {
                int j = j0 - 1 + c;
                if (j < 0) j = -j;
                if (j >= n_out) continue;
                ptr[ns] = base + ((size_t)i * n_out + j) * n_tau;
                wgt[ns] = wd[kk] * wu[a] * wv[c];
                ++ns;
            }
        }
    }
    return ns;
}

void PsiTable::psi_vector(double u, double v, double d, double* out) const {
    const int n_tau = m_max - m_min + 1;
    for (int n = 0; n < n_tau; ++n) out[n] = 0.0;
    const float* ptr[64];
    double       wgt[64];
    const int ns = flux_list(u, v, d, ptr, wgt);
    for (int s = 0; s < ns; ++s) {
        const float* p = ptr[s];
        const double w = wgt[s];
        for (int n = 0; n < n_tau; ++n) out[n] += w * p[n];
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


bool PsiTable::edges_prepare(double tau0, double step, int nedges, int* idx, double* w) const {
    double ratio = step / m_dtau;
    int R = (int)(ratio + 0.5);
    if (R < 1 || std::fabs(ratio - R) > 1.0e-9) return false;
    const int n_tau = m_max - m_min + 1;
    double x0 = tau0 / m_dtau - m_min;
    double fl = std::floor(x0);
    double f  = x0 - fl;
    lagrange4(f, w);
    for (int j = 0; j < nedges; ++j) {
        double xi = fl + (double)(j * R);      // partie entière de la position du bord j
        double x  = xi + f;
        if (x <= 0.0)              idx[j] = -1;
        else if (x >= n_tau - 1)   idx[j] = -2;
        else                       idx[j] = (int)xi;
    }
    return true;
}

void PsiTable::edges_values(const double* p, const int* idx, const double* w,
                            int nedges, double* out) const {
    const int n_tau = m_max - m_min + 1;
    const double w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
    for (int j = 0; j < nedges; ++j) {
        int n = idx[j];
        if (n >= 0)        out[j] = w0 * p[n] + w1 * p[n + 1] + w2 * p[n + 2] + w3 * p[n + 3];
        else if (n == -1)  out[j] = p[1];          // vec[0]
        else               out[j] = p[n_tau];      // vec[n_tau - 1]
    }
}

void PsiTable::psi_vector_f(double u, double v, double d, float* out) const {
    const int n_tau = m_max - m_min + 1;
    const float* dptr[64];
    double       dwgt[64];
    const int ns = flux_list(u, v, d, dptr, dwgt);
    if (ns == 0) {
        for (int n = 0; n < n_tau; ++n) out[n] = 0.0f;
        return;
    }
    const float* ptr[64];
    float        wgt[64];
    for (int s = 0; s < ns; ++s) { ptr[s] = dptr[s]; wgt[s] = (float)dwgt[s]; }

    // accumulation par blocs de 8 valeurs, gardees en registres
    const int B = 8;
    int n = 0;
    while (n < n_tau) {
        if (n + B > n_tau) n = n_tau - B;          // dernier bloc decale (recouvrement)
        float acc[B];
        for (int q = 0; q < B; ++q) acc[q] = 0.0f;
        for (int s = 0; s < ns; ++s) {
            const float* p = ptr[s] + n;
            const float  w = wgt[s];
            for (int q = 0; q < B; ++q) acc[q] += w * p[q];
        }
        for (int q = 0; q < B; ++q) out[n + q] = acc[q];
        if (n + B >= n_tau) break;
        n += B;
    }
}

void PsiTable::psi_vector_g(double u, double v, double d, float* out) const {
    const int n_tau = m_max - m_min + 1;
    if (n_tau > 256) { psi_vector_f(u, v, d, out); return; }
    const float* dptr[64];
    double       dwgt[64];
    const int ns = flux_list(u, v, d, dptr, dwgt);
    if (ns == 0) {
        for (int n = 0; n < n_tau; ++n) out[n] = 0.0f;
        return;
    }
    const float* ptr[64];
    float        wgt[64];
    for (int s = 0; s < ns; ++s) { ptr[s] = dptr[s]; wgt[s] = (float)dwgt[s]; }

    switch (n_tau) {
        case 54: accumulate_fixed<54>(ptr, wgt, ns, out); break;   // dga = 5 mm
        case 44: accumulate_fixed<44>(ptr, wgt, ns, out); break;   // dga = 3 mm
        default: accumulate_generic(ptr, wgt, ns, n_tau, out); break;
    }
}

void PsiTable::edges_values_f(const float* p, const int* idx, const double* w,
                              int nedges, double* out) const {
    const int n_tau = m_max - m_min + 1;
    const double w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
    for (int j = 0; j < nedges; ++j) {
        int n = idx[j];
        if (n >= 0)        out[j] = w0 * p[n] + w1 * p[n + 1] + w2 * p[n + 2] + w3 * p[n + 3];
        else if (n == -1)  out[j] = p[1];
        else               out[j] = p[n_tau];
    }
}
