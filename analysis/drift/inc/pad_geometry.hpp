#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <variant>
#include "detector.hpp"

// ---------------------------------------------------------------------------
// Géométrie du plan de pads
// ---------------------------------------------------------------------------

struct Pad {
    double x, y;     // centre du pad (cm)
    double size;     // côté du pad (cm), carré (0 pour les anneaux)
    double r_in;     // rayon intérieur (anneaux seulement)
    double r_out;    // rayon extérieur (anneaux seulement)
    int    sx;       // quadrant : +1 (x>0) ou -1 (x<0) 
    int    sy;       // quadrant : +1 (y>0) ou -1 (y<0)
    int    id;
};

struct PadCharge {
    int    pad_id;
    double x, y;
    double q;
    std::vector<double> t;
    std::vector<double> qt;
};

// ---------------------------------------------------------------------------
// Paramètres géométriques
// ---------------------------------------------------------------------------

struct PadPlaneParams {
    double pad_size     = 0.3;    // côté des pads cartésiens (cm)
    double inter_pad    = 0.015;  // inter-pad (cm)
    double r_quinconce  = 8.1;    // rayon coins < r → pad central (cm)
    double r_transition = 8.2;   // séparation transition/anneaux (cm)
    double r_active     = 10.0;   // rayon externe (cm)
    int    n_rings      = 6;     // nombre de demi-anneaux périphériques
    double y_offset     = 0.0;   // décalage ±y des demi-disques (cm)
};

// ---------------------------------------------------------------------------
// PadPlane
// ---------------------------------------------------------------------------

class PadPlane {
public:
    explicit PadPlane(const PadPlaneParams& params);

    const std::vector<Pad>& pads() const { return m_pads; }

    std::vector<const Pad*> neighbors(double x0, double y0, double r) const;

    std::vector<PadCharge> project(const std::vector<double>& x,
                                   const std::vector<double>& y,
                                   const std::vector<double>& z,
                                   const std::vector<double>& n_electrons,
                                   const std::vector<double>& sigma_T,
                                   const std::vector<double>& sigma_L,
                                   const DetectorConfig& cfg,
                                   std::vector<std::vector<double>>& qt_buf) const;

private:
    PadPlaneParams  m_params;
    std::vector<Pad>      m_pads;
    std::vector<std::vector<int>> m_pad_index;
    std::vector<int> m_ring_ids, m_trans_ids ;
    
    void build_anode();
};
