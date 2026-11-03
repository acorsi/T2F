#include "pad_geometry.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace {

inline double fast_erf(double x) {
    const double a1 =  0.254829592, a2 = -0.284496736;
    const double a3 =  1.421413741, a4 = -1.453152027, a5 = 1.061405429;
    const double p  =  0.3275911;
    double sign = (x < 0) ? -1.0 : 1.0;
    x = std::abs(x);
    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t * std::exp(-x*x);
    return sign * y;
}

inline double gaussian_cdf(double t) {
    static constexpr double inv_sqrt2 = 0.7071067811865475;
    return 0.5 * (1.0 + fast_erf(t * inv_sqrt2));
}

// Une cellule du pavage est un pad réel (zone quinconce) si son coin le plus éloigné
// est à l'intérieur de r_quinconce. Même critère dans build_anode et build_cells.
inline bool cell_is_real(double x, double y, double half_pad, double r_quinc) {
    double cx = std::abs(x) + half_pad;
    double cy = std::abs(y) + half_pad;
    return std::sqrt(cx * cx + cy * cy) < r_quinc;
}

// Indice du quadrant : 0:(+,+) 1:(+,-) 2:(-,+) 3:(-,-)  (ordre de création des anneaux)
inline int quadrant_index(double x, double y) {
    return ((x >= 0.0) ? 0 : 2) + ((y >= 0.0) ? 0 : 1);
}

} // namespace

// ---------------------------------------------------------------------------
// Constructeur
// ---------------------------------------------------------------------------

PadPlane::PadPlane(const PadPlaneParams& params, const DetectorConfig& cfg)
    : m_params(params),
      m_pitch(params.pad_size + params.inter_pad),
      m_reach(params.reach_pitch * (params.pad_size + params.inter_pad)),
      m_table(default_table_params(cfg, params.pad_size + params.inter_pad)),
      m_nrows(0), m_ncols(0)
{
    if (std::fabs(params.y_offset) > 1.0e-12)
        throw std::runtime_error("PadPlane : y_offset doit etre 0 (la table suppose des "
                                 "tuiles jointives en y = 0)");
    if (m_reach > m_table.u_max())
        throw std::runtime_error("PadPlane : reach_pitch depasse la portee de la table");
    build_anode();
    build_cells();
}

// ---------------------------------------------------------------------------
// neighbors — O(N)
// ---------------------------------------------------------------------------

std::vector<const Pad*> PadPlane::neighbors(double x0, double y0, double r) const {
    std::vector<const Pad*> result;
    double r2 = r * r;
    for (const auto& p : m_pads) {
      double dx = p.x - x0, dy = p.y - y0;
      if (dx*dx + dy*dy <= r2)
	result.push_back(&p);
    }
    return result;
}

// ---------------------------------------------------------------------------
// build_anode — quinconce centrale + anneau transition + demi-anneaux
// ---------------------------------------------------------------------------

void PadPlane::build_anode() {
    const double pad_size    = m_params.pad_size;
    const double pitch       = m_pitch;
    const double r_quinc     = m_params.r_quinconce;
    const double r_trans     = m_params.r_transition;
    const double r_active    = m_params.r_active;
    const int    n_rings     = m_params.n_rings;
    const double half_pad    = pad_size / 2.0;
    const double ring_width  = (r_active - r_trans) / n_rings;
    int id = 0;

    // --- 1. Zone quinconce ---
    // Les rangées sont en y = +-(k - 1/2) pitch : les deux rangées voisines de y = 0 se
    // touchent en y = 0 et ont le même décalage en x (rangées d'indice +1 et -1).
    int n_rows = static_cast<int>(std::ceil(r_active / pitch)) + 1;
    int n_cols = static_cast<int>(std::ceil(r_active / pitch)) + 1;

    for (int row = -n_rows; row <= n_rows; ++row) {
        if (row == 0) continue;
        double y = (row > 0) ? row * pitch - pitch / 2.0 : row * pitch + pitch / 2.0;
        double x_offset = (row % 2 != 0) ? pitch / 2.0 : 0.0;
        for (int col = -n_cols; col <= n_cols; ++col) {
            double x = col * pitch + x_offset;
            if (!cell_is_real(x, y, half_pad, r_quinc)) continue;

            Pad p;
            p.x    = x;
            p.y    = y;
            p.size = pad_size;
            p.r_in = p.r_out = 0.0;
            p.id = id++;
            p.sx = (x>=0) ? +1 : -1 ;
	    p.sy = (y>=0) ? +1 : -1 ;
            m_pads.push_back(p);
        }
    }

    // --- 2. Anneaux de transition (un par quadrant) ---
    // ordre : (sx,sy) = (+,+), (+,-), (-,+), (-,-)
    for (int sx = +1; sx >= -1; sx -= 2) {
      for (int sy = +1; sy >= -1; sy -= 2) {
        Pad p;
        p.x = 0.0;
        p.y = 0.0;
        p.size = 0.0;
        p.r_in = r_quinc;
        p.r_out = r_trans;
        p.sx = sx;
        p.sy = sy;
        p.id = id++;
        m_pads.push_back(p);
        m_trans_ids.push_back(p.id);
      }
    }

    // --- 3. Anneaux périphériques (n_rings par quadrant) ---
    // indice dans m_ring_ids : quadrant * n_rings + ring
    for (int sx = +1; sx >= -1; sx -= 2) {
      for (int sy = +1; sy >= -1; sy -= 2) {
        for (int ring = 0; ring < n_rings; ++ring) {
	  double r_in  = r_trans + ring * ring_width;
	  double r_out = r_in + ring_width;
	  Pad p;
	  p.x = 0.0; p.y = 0.0;
	  p.size = 0.0;
	  p.r_in = r_in; p.r_out = r_out;
	  p.sx = sx; p.sy = sy;
	  p.id = id++;
	  m_pads.push_back(p);
	  m_ring_ids.push_back(p.id);
        }
      }
    }

    std::cout << "PadPlane : " << m_pads.size() << " pads\n";
}

// ---------------------------------------------------------------------------
// build_cells — pavage virtuel de tout le plan avec les cellules en briques
// ---------------------------------------------------------------------------

void PadPlane::build_cells() {
    const double pitch      = m_pitch;
    const double half_pad   = m_params.pad_size / 2.0;
    const double r_quinc    = m_params.r_quinconce;
    const double r_trans    = m_params.r_transition;
    const double r_active   = m_params.r_active;
    const int    n_rings    = m_params.n_rings;
    const double ring_width = (r_active - r_trans) / n_rings;

    m_nrows = static_cast<int>(std::ceil(r_active / pitch)) + 1;
    m_ncols = m_nrows;
    m_grid.assign((size_t)(2 * m_nrows + 1) * (2 * m_ncols + 1), -1);

    const int NS = 16;                         // sous-points par côté pour les recouvrements
    int next_real_id = 0;
    int n_virtual = 0;

    for (int row = -m_nrows; row <= m_nrows; ++row) {
        if (row == 0) continue;
        double y = (row > 0) ? row * pitch - pitch / 2.0 : row * pitch + pitch / 2.0;
        double x_offset = (row % 2 != 0) ? pitch / 2.0 : 0.0;
        for (int col = -m_ncols; col <= m_ncols; ++col) {
            double x = col * pitch + x_offset;

            Cell c;
            c.x = x;
            c.y = y;
            c.first = (int)m_contribs.size();
            c.count = 0;

            if (cell_is_real(x, y, half_pad, r_quinc)) {
                // pad réel : une seule contribution, de fraction 1
                Contrib k;
                k.pad = next_real_id++;
                k.frac = 1.0;
                m_contribs.push_back(k);
                c.count = 1;
            } else {
                // cellule virtuelle : on échantillonne NS x NS points de la tuile et on
                // regarde dans quelle électrode tombe chaque point
                std::vector<int> hits;
                hits.reserve(NS * NS);
                for (int a = 0; a < NS; ++a) {
                    double px = x - pitch / 2.0 + (a + 0.5) * pitch / NS;
                    for (int b = 0; b < NS; ++b) {
                        double py = y - pitch / 2.0 + (b + 0.5) * pitch / NS;
                        double r = std::sqrt(px * px + py * py);
                        int qi = quadrant_index(px, py);
                        int target = -1;                       // -1 : au-delà de r_active, perdu
                        if (r < r_trans) {
                            target = m_trans_ids[qi];
                        } else if (r < r_active) {
                            int ring = (int)((r - r_trans) / ring_width);
                            if (ring > n_rings - 1) ring = n_rings - 1;
                            target = m_ring_ids[qi * n_rings + ring];
                        }
                        if (target >= 0) hits.push_back(target);
                    }
                }
                if (hits.empty()) continue;           // cellule entièrement hors du plan actif
                std::sort(hits.begin(), hits.end());
                size_t s = 0;
                while (s < hits.size()) {
                    size_t e = s;
                    while (e < hits.size() && hits[e] == hits[s]) ++e;
                    Contrib k;
                    k.pad = hits[s];
                    k.frac = (double)(e - s) / (double)(NS * NS);
                    m_contribs.push_back(k);
                    ++c.count;
                    s = e;
                }
                ++n_virtual;
            }

            int index = (int)m_cells.size();
            m_cells.push_back(c);
            m_grid[(size_t)(row + m_nrows) * (2 * m_ncols + 1) + (col + m_ncols)] = index;
        }
    }
    std::cout << "PadPlane : " << m_cells.size() << " cellules (" << n_virtual
              << " virtuelles), table " << m_table.memory_bytes() / 1.0e6 << " Mo\n";
}

int PadPlane::find_cell(int row, int col) const {
    if (row < -m_nrows || row > m_nrows || col < -m_ncols || col > m_ncols) return -1;
    return m_grid[(size_t)(row + m_nrows) * (2 * m_ncols + 1) + (col + m_ncols)];
}

// ---------------------------------------------------------------------------
// project — induction de Ramo-Shockley par la table PsiTable
// ---------------------------------------------------------------------------

std::vector<PadCharge> PadPlane::project(
    const std::vector<double>& x, const std::vector<double>& y,
    const std::vector<double>& z, const std::vector<double>& n_electrons,
    const std::vector<double>& sigma_T, const std::vector<double>& /*sigma_L*/,
    const DetectorConfig& cfg,
    std::vector<std::vector<double>>& qt_buf) const
{
    const double p        = m_pitch;
    const double delta_t  = cfg.delta_t;
    const int    n_bins   = cfg.n_bins();
    const int    n_pads   = (int)m_pads.size();
    const double tau_lo   = m_table.tau_min();
    const double tau_hi   = m_table.tau_max();

    if ((int)qt_buf.size() < n_pads)
        qt_buf.assign(n_pads, std::vector<double>(n_bins, 0.0));

    std::vector<char> touched(n_pads, 0);
    std::vector<int>  touched_list;

    // tampons locaux (propres à cet appel, donc à ce thread)
    std::vector<double> vec(m_table.n_tau());
    std::vector<double> dq((size_t)((tau_hi - tau_lo) / delta_t) + 6);

    const size_t N = x.size();
    for (size_t i = 0; i < N; ++i) {
        if (n_electrons[i] <= 0.0) continue;

        const double d   = cfg.drift_distance(z[i]);
        const double t_g = cfg.t_grid(z[i]);       // arrivée du centre du nuage à la grille

        // Avertissement hérité : voxel trop près du bord actif ou au-delà
        double r_voxel = std::sqrt(x[i]*x[i] + y[i]*y[i]);
        double frac_beyond = gaussian_cdf(-(m_params.r_active - r_voxel) / sigma_T[i]);
        if (frac_beyond > 1e-6) {
            std::cout << "\033[31m[WARNING] charge beyond r_active: r=" << r_voxel
                      << " sigma=" << sigma_T[i] << " frac=" << frac_beyond << "\033[0m\n";
        }

        // bins touchés par le signal de ce voxel : k0 .. k1-1
        int k0 = (int)std::floor((t_g + tau_lo) / delta_t);
        int k1 = (int)std::ceil ((t_g + tau_hi) / delta_t);
        if (k0 < 0) k0 = 0;
        if (k1 > n_bins) k1 = n_bins;
        int nb = k1 - k0;
        if (nb <= 0) continue;
        if (nb > (int)dq.size()) nb = (int)dq.size();

        // cellules à portée du voxel : tuiles de rang kt (y dans [kt p, (kt+1) p])
        int kt_min = (int)std::floor((y[i] - m_reach) / p);
        int kt_max = (int)std::floor((y[i] + m_reach) / p);
        for (int kt = kt_min; kt <= kt_max; ++kt) {
            int    row = (kt >= 0) ? kt + 1 : kt;
            double yc  = (kt + 0.5) * p;
            double v   = y[i] - yc;
            if (std::fabs(v) > m_reach) continue;
            double x_offset = (row % 2 != 0) ? p / 2.0 : 0.0;
            int col_min = (int)std::ceil ((x[i] - m_reach - x_offset) / p);
            int col_max = (int)std::floor((x[i] + m_reach - x_offset) / p);
            for (int col = col_min; col <= col_max; ++col) {
                int ic = find_cell(row, col);
                if (ic < 0) continue;
                const Cell& cell = m_cells[ic];
                double u = x[i] - cell.x;

                m_table.psi_vector(u, v, d, &vec[0]);

                // charge induite sur la cellule pendant chaque bin, aux bords de bins
                double prev = m_table.psi_at(&vec[0], k0 * delta_t - t_g);
                for (int j = 0; j < nb; ++j) {
                    double next = m_table.psi_at(&vec[0], (k0 + j + 1) * delta_t - t_g);
                    dq[j] = n_electrons[i] * (next - prev);
                    prev = next;
                }

                // répartition sur les électrodes (1 pour un pad réel, plusieurs sinon)
                for (int c = 0; c < cell.count; ++c) {
                    const Contrib& ct = m_contribs[cell.first + c];
                    if (!touched[ct.pad]) { touched[ct.pad] = 1; touched_list.push_back(ct.pad); }
                    double* buf = &qt_buf[ct.pad][k0];
                    for (int j = 0; j < nb; ++j) buf[j] += ct.frac * dq[j];
                }
            }
        }
    }

    // --- lecture : seuils signés, charge cumulée ---
    std::vector<PadCharge> result;
    result.reserve(touched_list.size());
    for (size_t m = 0; m < touched_list.size(); ++m) {
        int id = touched_list[m];
        PadCharge pc;
        pc.pad_id = m_pads[id].id;
        pc.x      = m_pads[id].x;
        pc.y      = m_pads[id].y;

        double Q = 0.0;
        for (int k = 0; k < n_bins; ++k) {
            double q = qt_buf[id][k];
            Q += q;
            if (q > cfg.seuil_qt_pos || q < cfg.seuil_qt_neg) {
                pc.t.push_back(k * delta_t);
                pc.qt.push_back(q);
                pc.Q.push_back(Q);
            }
            qt_buf[id][k] = 0.0;
        }
        pc.q = Q;
        if (!pc.t.empty()) result.push_back(pc);     // un pad sans bin au-dessus du seuil n'est pas lu
    }
    return result;
}
