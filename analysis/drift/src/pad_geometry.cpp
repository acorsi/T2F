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

inline double integrate_pad(double px, double py, double pad_size,
                            double x0, double y0, double sigma) {
    if (sigma <= 0.0) {
        bool in_x = std::abs(px - x0) <= pad_size / 2.0;
        bool in_y = std::abs(py - y0) <= pad_size / 2.0;
        return (in_x && in_y) ? 1.0 : 0.0;
    }
    double half = pad_size / 2.0;
    double fx = gaussian_cdf((px + half - x0) / sigma)
              - gaussian_cdf((px - half - x0) / sigma);
    double fy = gaussian_cdf((py + half - y0) / sigma)
              - gaussian_cdf((py - half - y0) / sigma);
    return fx * fy;
}

// Intégrale d'une gaussienne 1D sur [a, b]
inline double integrate_ring_1d(double a, double b, double x0, double sigma) {
    return gaussian_cdf((b - x0) / sigma) - gaussian_cdf((a - x0) / sigma);
}

// Intégrale gaussienne 2D sur un pad carré avec bornes en y clippées
// (pour gérer le gap entre les deux demi-anodes)
inline double integrate_pad_clipped(double px, double py, double pad_size,
                                     double x0, double y0, double sigma,
                                     double y_min, double y_max) {
    if (sigma <= 0.0) {
        bool in_x = std::abs(px - x0) <= pad_size / 2.0;
        bool in_y = (y0 >= y_min && y0 <= y_max);
        return (in_x && in_y) ? 1.0 : 0.0;
    }
    double half = pad_size / 2.0;
    double fx = gaussian_cdf((px + half - x0) / sigma)
              - gaussian_cdf((px - half - x0) / sigma);
    double fy = gaussian_cdf((std::min(py + half, y_max) - y0) / sigma)
              - gaussian_cdf((std::max(py - half, y_min) - y0) / sigma);
    return fx * fy;
}

inline double gauss_t(double t, double sigma) {
    static constexpr double inv_sqrt2pi = 0.3989422804014327;
    return (inv_sqrt2pi / sigma) * std::exp(-0.5 * (t / sigma) * (t / sigma));
}

} // namespace

// ---------------------------------------------------------------------------
// Constructeurs
// ---------------------------------------------------------------------------

PadPlane::PadPlane(const PadPlaneParams& params)
    : m_params(params)
{
    build_anode();
}

// ---------------------------------------------------------------------------
// neighbors — lookup O(1) pour type1, O(N) pour type2
// ---------------------------------------------------------------------------

std::vector<const Pad*> PadPlane::neighbors(double x0, double y0, double r) const {
    std::vector<const Pad*> result;
    double r2 = r * r;

    // type2 : O(N) — nombre de pads limité (~1316)
    for (const auto& p : m_pads) {
      double dx = p.x - x0, dy = p.y - y0;
      if (dx*dx + dy*dy <= r2)
	result.push_back(&p);
    }
    return result;
}

// ---------------------------------------------------------------------------
// project — dispatch
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// build_anode — quinconce centrale + anneau transition + demi-anneaux
// ---------------------------------------------------------------------------

void PadPlane::build_anode() {
    const double pad_size    = m_params.pad_size;
    const double inter_pad   = m_params.inter_pad;
    const double pitch       = pad_size + inter_pad;
    const double r_quinc     = m_params.r_quinconce;
    const double r_trans     = m_params.r_transition;
    const double r_active    = m_params.r_active;
    const int    n_rings     = m_params.n_rings;
    const double y_offset    = m_params.y_offset;
    const double half_pad    = pad_size / 2.0;
    const double ring_width  = (r_active - r_trans) / n_rings;
    int id = 0;

    // --- 1. Zone quinconce ---
    int n_rows = static_cast<int>(std::ceil(r_active / pitch)) + 1;
    int n_cols = static_cast<int>(std::ceil(r_active / pitch)) + 1;

    for (int row = -n_rows; row <= n_rows; ++row) {
        if (row == 0) continue;
        double y_base = row * pitch;
        double y_off  = (y_base >= 0) ? +y_offset - pitch/2.0 : pitch/2.0 - y_offset;
        double y = y_base + y_off;

        double x_offset = (row % 2 != 0) ? pitch / 2.0 : 0.0;
        for (int col = -n_cols; col <= n_cols; ++col) {
            double x = col * pitch + x_offset;

            double max_corner_r = std::sqrt(
                (std::abs(x) + half_pad) * (std::abs(x) + half_pad) +
                (std::abs(y) + half_pad) * (std::abs(y) + half_pad));
            if (max_corner_r >= r_quinc) continue;

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
    for (int sx : {+1, -1}) {
      for (int sy : {+1, -1}) {
        Pad p;
        p.x = sx * 0.0;  // ou position représentative du quadrant si besoin ailleurs
        p.y = sy * y_offset;  // y_offset désormais possiblement obsolète (mis à 0 plus tôt)
        p.size = 0.0;
        p.r_in = m_params.r_quinconce;
        p.r_out = r_trans;
        p.sx = sx;
        p.sy = sy;
        p.id = id++;
        m_pads.push_back(p);
        m_trans_ids.push_back(p.id);
      }
    }

    // --- 3. Anneaux périphériques (n_rings par quadrant) ---
    for (int sx : {+1, -1}) {
      for (int sy : {+1, -1}) {
        for (int ring = 0; ring < n_rings; ++ring) {
	  double r_in  = r_trans + ring * ring_width;
	  double r_out = r_in + ring_width;
	  Pad p;
	  p.x = 0.0; p.y = 0.0;  // idem, position non utilisée en pratique pour l'intégration radiale
	  p.size = 0.0;
	  p.r_in = r_in; p.r_out = r_out;
	  p.sx = sx; p.sy = sy;
	  p.id = id++;
	  m_pads.push_back(p);
	  m_ring_ids.push_back(p.id);
        }
      }
    }
    
    // // --- 2. Anneau de transition (un pad par demi-disque) ---
    // for (int half : {+1, -1}) {
    //     double y_off = half * y_offset;
    //     Pad p;
    //     p.x    = 0.0;
    //     p.y    = y_off;
    //     p.size = 0.0;
    //     p.r_in = m_params.r_quinconce;   // bord des pads cartésiens
    //     p.r_out = r_trans;
    //     p.half = half;
    //     p.id   = id++;
    //     m_pads.push_back(p);

    // 	if (half == +1) m_trans_id_pos = p.id; 
    // 	else             m_trans_id_neg = p.id;
    // }

    // // --- 3. Demi-anneaux périphériques ---
    // for (int half : {+1, -1}) {
    //     double y_off = half * y_offset;
    //     for (int ring = 0; ring < n_rings; ++ring) {
    //         double r_in  = r_trans + ring * ring_width;
    //         double r_out = r_in + ring_width;
    //         Pad p;
    //         p.x    = 0.0;
    //         p.y    = y_off;
    //         p.size = 0.0;
    //         p.r_in = r_in;
    //         p.r_out = r_out;
    //         p.half = half;
    //         p.id   = id++;
    //         m_pads.push_back(p);
    // 	    if (half == +1) m_ring_ids_pos.push_back(p.id);
    // 	    else             m_ring_ids_neg.push_back(p.id);
    //     }
    // }

    std::cout << "PadPlane type2 : " << m_pads.size() << " pads\n";
}

// ---------------------------------------------------------------------------
// project_type2 — quinconce + anneaux
// ---------------------------------------------------------------------------

std::vector<PadCharge> PadPlane::project(
    const std::vector<double>& x, const std::vector<double>& y,
    const std::vector<double>& z, const std::vector<double>& n_electrons,
    const std::vector<double>& sigma_T, const std::vector<double>& sigma_L,
    const DetectorConfig& cfg,
    std::vector<std::vector<double>>& qt_buf) const
{
    const double pad_size  = m_params.pad_size;
    const double inter_pad = m_params.inter_pad;
    const double pitch     = pad_size + inter_pad;
    const double v_e       = cfg.v_e;
    const double delta_t   = cfg.delta_t;
    const double seuil_qt  = cfg.seuil_qt;
    const double dkg       = cfg.dkg;
    const double n_sigma   = 4.0;

    const int    n_bins = cfg.n_bins() ;

    if ((int)qt_buf.size() < (int)m_pads.size())
        qt_buf.assign(m_pads.size(), std::vector<double>(n_bins, 0.0));

    std::vector<double> charge(m_pads.size(), 0.0);
    std::vector<int>    touched_pads;

    const size_t N = x.size();
    for (size_t i = 0; i < N; ++i) {
        if (n_electrons[i] <= 0.0) continue;

        double drift    = cfg.drift_distance(z[i]);
        double t_i      = drift / v_e;
        double sigma_ti = sigma_L[i] / v_e;
        double sigma    = sigma_T[i];
        double r_voxel  = std::sqrt(x[i]*x[i] + y[i]*y[i]);

        int t_start = std::max(0,       (int)std::floor((t_i - n_sigma*sigma_ti) / delta_t));
        int t_end   = std::min(n_bins-1,(int)std::ceil ((t_i + n_sigma*sigma_ti) / delta_t));

        auto add_charge = [&](int pad_id, double frac) {
            if (frac <= 0.0) return;
            double q_voxel = n_electrons[i] * frac;
            if (charge[pad_id] == 0.0) touched_pads.push_back(pad_id);
            charge[pad_id] += q_voxel;
            if (sigma_ti > 0.0) {
                for (int k = t_start; k <= t_end; ++k) {
                    double tk = k * delta_t;
                    qt_buf[pad_id][k] += q_voxel * gauss_t(tk - t_i, sigma_ti) * delta_t;
                }
            } else {
                std::cout << "\033[31m[WARNING] sigma_ti<=0 in project_type2\033[0m\n";
            }
        };




	double frac_total = 0.0;
	
	// --- 1. Contribution cartésienne (quinconce) ---
	if (r_voxel < m_params.r_quinconce + n_sigma * sigma) {
	  auto candidates = neighbors(x[i], y[i], std::max(3.0 * pitch, n_sigma * sigma));
	  for (const Pad* pad : candidates) {
	    if (pad->size <= 0.0) continue;
	    double frac = integrate_pad(pad->x, pad->y, pitch, x[i], y[i], sigma);
	    if (frac <= 0.0) continue;
	    frac_total += frac;
	    add_charge(pad->id, frac);
	  }
	}


	double cdf_x = gaussian_cdf(x[i] / sigma);
	double cdf_y = gaussian_cdf(y[i] / sigma);

	// --- 2. Vrais anneaux, pondérés par quadrant ---
	if (r_voxel + n_sigma * sigma > m_params.r_transition) {
	  for (int rid : m_ring_ids) {
	    const Pad& pad = m_pads[rid];
	    double frac_r = integrate_ring_1d(pad.r_in, pad.r_out, r_voxel, sigma);
	    if (frac_r <= 0.0) continue;
	    double wx = (pad.sx > 0) ? cdf_x : (1.0 - cdf_x);
	    double wy = (pad.sy > 0) ? cdf_y : (1.0 - cdf_y);
	    double frac = frac_r * wx * wy;
	    if (frac <= 0.0) continue;
	    frac_total += frac;
	    add_charge(rid, frac);
	  }
        }

	// // --- 2. Contribution des vrais anneaux périphériques ---
	// if (r_voxel + n_sigma * sigma > m_params.r_transition) {
	//   const auto& ring_ids = (half_i == +1) ? m_ring_ids_pos : m_ring_ids_neg;
	//   for (int rid : ring_ids) {
	//     const Pad& pad = m_pads[rid];
	//     double frac = integrate_ring_1d(pad.r_in, pad.r_out, r_voxel, sigma);
	//     if (frac <= 0.0) continue;
	//     frac_total += frac;
	//     add_charge(rid, frac);
	//   }
	// }
	
	// --- 3. Perte réelle au-delà de r_active (ne doit normalement jamais s'activer) ---
	double frac_beyond_active = gaussian_cdf(-(m_params.r_active - r_voxel) / sigma);
	if (frac_beyond_active > 1e-6) {
	  std::cout << "\033[31m[WARNING] charge beyond r_active: r=" << r_voxel
		    << " sigma=" << sigma << " frac=" << frac_beyond_active << "\033[0m\n";
	}
	frac_total += frac_beyond_active;

	//        std::cout << r_voxel << " " << sigma << " " << frac_beyond_active << " " << frac_total << std::endl;

        // --- 4. Transition, pondéré par quadrant, répartissant le résidu total ---
        double frac_trans_total = 1.0 - frac_total;
	if (frac_trans_total > 1e-6) {
	  for (int tid : m_trans_ids) {
	    const Pad& pad = m_pads[tid];
	    double wx = (pad.sx > 0) ? cdf_x : (1.0 - cdf_x);
	    double wy = (pad.sy > 0) ? cdf_y : (1.0 - cdf_y);
	    double frac = frac_trans_total * wx * wy;
	    add_charge(tid, frac);
	  }
	  frac_total = 1.0;
        }

        
	// double frac_trans = 1.0 - frac_total;
	// if (frac_trans > 1e-6) {
	//   int trans_id = (half_i == +1) ? m_trans_id_pos : m_trans_id_neg;
	//   if (trans_id >= 0) {
	//     add_charge(trans_id, frac_trans);
	//     frac_total += frac_trans;
	//   }
	// }	

    }

    std::vector<PadCharge> result;
    result.reserve(touched_pads.size());

    for (int id : touched_pads) {
        PadCharge pc;
        pc.pad_id = m_pads[id].id;
        pc.x      = m_pads[id].x;
        pc.y      = m_pads[id].y;
        pc.q      = charge[id];

        for (int k = 0; k < n_bins; ++k) {
            if (qt_buf[id][k] > seuil_qt) {
                pc.t.push_back(k * delta_t);
                pc.qt.push_back(qt_buf[id][k]);
            }
            qt_buf[id][k] = 0.0;
        }
        charge[id] = 0.0;
        result.push_back(std::move(pc));
    }
    return result;
}
