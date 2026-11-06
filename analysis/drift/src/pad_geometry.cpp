#include "pad_geometry.hpp"
#include <cmath>
#include "prof.hpp"
#ifdef T2F_PROF
ProfStats g_prof[64];
#endif
#include <algorithm>
#include <iostream>

static long g_n_beyond = 0;
static long g_n_traces_beyond = 0;
static long g_n_traces = 0;
long pad_beyond_active_count() { return g_n_beyond; }
long pad_traces_beyond_count() { return g_n_traces_beyond; }
long pad_traces_count() { return g_n_traces; }

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
void PadPlane::project_pair(
    const FragTrace& H, const FragTrace& L,
    const DetectorConfig& cfg,
    std::vector<std::vector<double>>& qt_buf,
    std::vector<PadCharge>& resH,
    std::vector<PadCharge>& resL) const
{
    const double p        = m_pitch;
    const double delta_t  = cfg.delta_t;
    const int    n_bins   = cfg.n_bins();
    const int    n_pads   = (int)m_pads.size();
    const double tau_lo   = m_table.tau_min();
    const double tau_hi   = m_table.tau_max();

    const FragTrace* F[2] = { &H, &L };

    if ((int)qt_buf.size() < n_pads)
        qt_buf.assign(n_pads, std::vector<double>(n_bins, 0.0));

    std::vector<char> touched(n_pads, 0);        // touché par l'un ou l'autre fragment
    std::vector<char> touchedF[2];               // touché par H / par L
    touchedF[0].assign(n_pads, 0);
    touchedF[1].assign(n_pads, 0);
    std::vector<int> touched_list;
    std::vector<int> kmin(n_pads, n_bins);
    std::vector<int> kmax(n_pads, -1);
    bool trace_beyond[2] = { false, false };

#ifdef T2F_PROF
    std::vector<int> done_cells;
#endif

    PROF(double w0 = omp_get_wtime();)
    PROF(ProfStats& S = g_prof[omp_get_thread_num() % 64];)

    // tampons locaux (propres à cet appel, donc à ce thread)
    const int n_tau = m_table.n_tau();
    std::vector<float> vecbuf(n_tau + 2);
    float* vec = &vecbuf[1];
    std::vector<double> vecd(n_tau);
    std::vector<double> dq((size_t)((tau_hi - tau_lo) / delta_t) + 6);
    std::vector<int>    eidx(dq.size() + 1);
    std::vector<double> eout(dq.size() + 1);
    double ew[4];

    // --- Passe 1 : borne sur la SOMME des deux fragments ---
    double thr = cfg.seuil_qt_pos;
    if (-cfg.seuil_qt_neg < thr) thr = -cfg.seuil_qt_neg;
    const double margin = m_params.bound_margin;
    const bool select = (margin > 0.0 && thr > 0.0);

    std::vector<char> keepC;
    std::vector<char> keepE;
    if (select) {
        keepC.assign(m_cells.size(), 0);
        keepE.assign(n_pads, 0);
        std::vector<double> bound_cell(m_cells.size(), 0.0);

        for (int f = 0; f < 2; ++f) {
            const std::vector<double>& x = *F[f]->x;
            const std::vector<double>& y = *F[f]->y;
            const std::vector<double>& z = *F[f]->z;
            const std::vector<double>& n_electrons = *F[f]->n_electrons;
            const size_t N = x.size();

            for (size_t i = 0; i < N; ++i) {
                if (n_electrons[i] <= 0.0) continue;
                const int    kd1 = m_table.d_cell(cfg.drift_distance(z[i]));
                const double w1  = n_electrons[i] * margin;
                int kt1 = (int)std::floor((y[i] - m_reach) / p);
                int kt2 = (int)std::floor((y[i] + m_reach) / p);
                for (int kt = kt1; kt <= kt2; ++kt) {
                    int    row = (kt >= 0) ? kt + 1 : kt;
                    double v   = y[i] - (kt + 0.5) * p;
                    if (std::fabs(v) > m_reach) continue;
                    double x_offset = (row % 2 != 0) ? p / 2.0 : 0.0;
                    int c1 = (int)std::ceil ((x[i] - m_reach - x_offset) / p);
                    int c2 = (int)std::floor((x[i] + m_reach - x_offset) / p);
                    for (int col = c1; col <= c2; ++col) {
                        int ic = find_cell(row, col);
                        if (ic < 0) continue;
                        bound_cell[ic] += w1 * m_table.dpsi_max(x[i] - m_cells[ic].x, v, kd1);
                    }
                }
            }
        }

        // borne par électrode : somme des bornes des cellules, pondérées par le recouvrement
        std::vector<double> bound_elec(n_pads, 0.0);
        for (size_t c = 0; c < m_cells.size(); ++c) {
            if (bound_cell[c] <= 0.0) continue;
            const Cell& cl = m_cells[c];
            for (int k = 0; k < cl.count; ++k) {
                const Contrib& ct = m_contribs[cl.first + k];
                bound_elec[ct.pad] += ct.frac * bound_cell[c];
            }
        }
        for (int e = 0; e < n_pads; ++e) keepE[e] = (bound_elec[e] > thr) ? 1 : 0;

        // une cellule est gardée si une de ses électrodes l'est
        for (size_t c = 0; c < m_cells.size(); ++c) {
            if (bound_cell[c] <= 0.0) continue;
            PROF(S.n_cell_touched++;)
            const Cell& cl = m_cells[c];
            for (int k = 0; k < cl.count; ++k) {
                if (keepE[m_contribs[cl.first + k].pad]) { keepC[c] = 1; break; }
            }
            PROF(if (keepC[c]) S.n_cell_keep++;)
        }
    }

    // --- Passe 2 : accumulation des deux fragments dans le même qt_buf ---
    for (int f = 0; f < 2; ++f) {
        const std::vector<double>& x = *F[f]->x;
        const std::vector<double>& y = *F[f]->y;
        const std::vector<double>& z = *F[f]->z;
        const std::vector<double>& n_electrons = *F[f]->n_electrons;
        const std::vector<double>& sigma_T = *F[f]->sigma_T;
        const size_t N = x.size();

        for (size_t i = 0; i < N; ++i) {
            if (n_electrons[i] <= 0.0) continue;

            const double d = cfg.drift_distance(z[i]);
            const double t_g = cfg.t_grid(z[i]);       // arrivée du centre du nuage à la grille

            // Avertissement hérité : voxel trop près du bord actif ou au-delà
            double r_voxel = std::sqrt(x[i]*x[i] + y[i]*y[i]);
            double frac_beyond = gaussian_cdf(-(m_params.r_active - r_voxel) / sigma_T[i]);
            if (frac_beyond > 1e-6) {
                trace_beyond[f] = true;
#pragma omp atomic
                g_n_beyond++;
            }

            // bins touchés par le signal de ce voxel : k0 .. k1-1
            int k0 = (int)std::floor((t_g + tau_lo) / delta_t);
            int k1 = (int)std::ceil ((t_g + tau_hi) / delta_t);
            if (k0 < 0) k0 = 0;
            if (k1 > n_bins) k1 = n_bins;
            int nb = k1 - k0;
            if (nb <= 0) continue;
            if (nb > (int)dq.size())
                nb = (int)dq.size();

            const bool fast = m_table.edges_prepare(k0 * delta_t - t_g, delta_t, nb + 1, &eidx[0], ew);

            PROF(S.n_vox++;)
            PROF(S.sum_ne += n_electrons[i];)

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
                    if (select && !keepC[ic]) {
                        PROF(S.n_skip_sel++;)
                        continue;
                    }
                    const Cell& cell = m_cells[ic];
                    double u = x[i] - cell.x;
                    PROF(done_cells.push_back(ic);)

                    PROF(S.n_cell++;)
                    PROF(S.sum_nb += nb;)
                    PROF(double a0 = omp_get_wtime();)
                    m_table.psi_vector_g(u, v, d, vec);
                    vecbuf[0]         = vec[0];
                    vecbuf[n_tau + 1] = vec[n_tau - 1];
                    PROF(double a1 = omp_get_wtime(); S.t_vec += a1 - a0;)

                    if (fast) {
                        m_table.edges_values_f(&vecbuf[0], &eidx[0], ew, nb + 1, &eout[0]);
                        for (int j = 0; j < nb; ++j) dq[j] = n_electrons[i] * (eout[j + 1] - eout[j]);
                    } else {
                        for (int n = 0; n < n_tau; ++n) vecd[n] = vec[n];
                        double prev = m_table.psi_at(&vecd[0], k0 * delta_t - t_g);
                        for (int j = 0; j < nb; ++j) {
                            double next = m_table.psi_at(&vecd[0], (k0 + j + 1) * delta_t - t_g);
                            dq[j] = n_electrons[i] * (next - prev);
                            prev = next;
                        }
                    }
                    PROF(double a2 = omp_get_wtime(); S.t_at += a2 - a1;)

                    // répartition sur les électrodes (1 pour un pad réel, plusieurs sinon)
                    for (int c = 0; c < cell.count; ++c) {
                        const Contrib& ct = m_contribs[cell.first + c];
                        if (select && !keepE[ct.pad]) continue;
                        if (!touched[ct.pad]) { touched[ct.pad] = 1; touched_list.push_back(ct.pad); }
                        touchedF[f][ct.pad] = 1;
			if (k0 < kmin[ct.pad])          kmin[ct.pad] = k0;
			if (k0 + nb - 1 > kmax[ct.pad]) kmax[ct.pad] = k0 + nb - 1;
                        double* buf = &qt_buf[ct.pad][k0];
                        for (int j = 0; j < nb; ++j) buf[j] += ct.frac * dq[j];
                    }
                    PROF(double a3 = omp_get_wtime(); S.t_acc += a3 - a2; S.n_contrib += cell.count;)
                }
            }
        }
    }

#pragma omp atomic
    g_n_traces += 2;
    for (int f = 0; f < 2; ++f) {
        if (trace_beyond[f]) {
#pragma omp atomic
            g_n_traces_beyond++;
        }
    }

    PROF(double w1 = omp_get_wtime();)
    // --- lecture : seuils signés appliqués sur la somme H+L ---
    resH.clear();
    resL.clear();
    resH.reserve(touched_list.size());
    resL.reserve(touched_list.size());
    PROF(std::vector<char> unread(n_pads, 0);)
      
    for (size_t m = 0; m < touched_list.size(); ++m) {
      int id = touched_list[m];
      PadCharge pc;
      pc.pad_id = m_pads[id].id;
      pc.x      = m_pads[id].x;
      pc.y      = m_pads[id].y;
      
      const int ka = kmin[id], kb = kmax[id];
      double Q = 0.0;
      bool pass = false;
      for (int k = ka; k <= kb; ++k) {
	double q = qt_buf[id][k];
	Q += q;
	if (q > cfg.seuil_qt_pos || q < cfg.seuil_qt_neg) pass = true;
      }
      if (pass) {
	pc.t.reserve(kb - ka + 1);
	pc.qt.reserve(kb - ka + 1);
	for (int k = ka; k <= kb; ++k) {
	  pc.t.push_back(k * delta_t);
	  pc.qt.push_back(qt_buf[id][k]);
	}
      }
      for (int k = ka; k <= kb; ++k) qt_buf[id][k] = 0.0;
      pc.q = Q;
      if (pass) {
	if (touchedF[0][id]) resH.push_back(pc);
	if (touchedF[1][id]) resL.push_back(pc);
      }
      PROF(if (!pass) unread[id] = 1;)
    }
    

#ifdef T2F_PROF
    for (size_t q = 0; q < done_cells.size(); ++q) {
        const Cell& cl = m_cells[done_cells[q]];
        bool all_unread = true;
        for (int k = 0; k < cl.count; ++k) {
            int pd = m_contribs[cl.first + k].pad;
            if (!(select && !keepE[pd]) && !unread[pd]) { all_unread = false; break; }
        }
        if (all_unread) S.n_cell_wasted++;
    }
#endif

    PROF(double w2 = omp_get_wtime(); S.t_out += w2 - w1; S.t_tot += w2 - w0;)
}
