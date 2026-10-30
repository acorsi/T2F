#pragma once

#include <cmath>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Géométrie du détecteur T2F
//

// Une seule anode, émission selon Y, dkg=20cm
//   z = 0       cathode
//   z = dkg     grille de Frisch
//   z = dkg+dga anode unique
//   fragments émis selon ±y
//
// Unités : cm
// ---------------------------------------------------------------------------

enum class EmissionAxis { Y, Z };

struct DetectorConfig {

    EmissionAxis emission_axis = EmissionAxis::Y;

    // --- Géométrie ---
    double dkg          = 20.0;   // distance cathode → grille (cm)
    double dga          =  1.0;   // distance grille → anode (cm)
    double vertex_r_max =  1.0;   // rayon max du vertex dans le plan XZ (cm)
    double step_size    =  0.02;  // pas de la trace (cm)

    // --- Gaz / diffusion ---
    double W        = 26.0e-6;  // énergie par paire e⁻/ion (MeV)
    double D_T      = 0.03;     // coefficient diffusion transverse   (cm/√cm)
    double D_L      = 0.015;    // coefficient diffusion longitudinale (cm/√cm)
    double v_e      = 0.005;    // vitesse de dérive (cm/ns)
    double delta_t  = 10.0;     // pas temporel (ns)
    double seuil_qt = 0.0;      // seuil Q(t) en n_electrons

    // --- Helpers ---
    double z_grid()  const { return dkg; }
    double z_anode() const { return dkg + dga; }

  double sigma_T(double z) const { return D_T * std::sqrt(drift_distance(z)); }
  double sigma_L(double z) const { return D_L * std::sqrt(drift_distance(z)); }
  
  int n_bins() const { return static_cast<int>( std::ceil((2.0 * dkg / v_e + 200.0) / delta_t)) + 1 ;}
  
  double drift_distance(double z) const {
    return dkg - z;  // dérive vers z>0, z ∈ [0, dkg]
  }

};
