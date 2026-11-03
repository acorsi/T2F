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
    double dga          =  0.5;   // distance grille → anode (cm)
    double vertex_r_max =  1.0;   // rayon max du vertex dans le plan XZ (cm)
    double step_size    =  0.02;  // pas de la trace (cm)

    // --- Gaz / diffusion ---
    double Fano     = 0.2;      // Facteur de Fano pour fluctuation du nombre d'électrons  
    double W        = 26.0e-6;  // énergie par paire e⁻/ion (MeV)

    double DT_kg    = 0.06;     // coefficient diffusion transverse   (cm/√cm)
    double DT_ga    = 0.06;        // coefficient diffusion transverse   (cm/√cm)
    double DL_kg    = 0.035;     // coefficient diffusion longitudinale (cm/√cm)
    double DL_ga    = 0.02;        // coefficient diffusion longitudinale (cm/√cm)
   
    double ve_kg    = 0.0055;   // vitesse de dérive (cm/ns) cathode-grille 
    double ve_ga    = 0.002;    // vitesse de dérive (cm/ns) grille-anode
    double delta_t  = 10.0;     // pas temporel (ns)

    // Seuils de lecture sur qt (électrons par bin), comme un déclenchement d'oscilloscope.
    // Un bin est gardé si qt > seuil_qt_pos  OU  qt < seuil_qt_neg.
    //   - les deux côtés : seuil_qt_pos = +200, seuil_qt_neg = -200 (défaut)
    //   - seulement le lobe négatif : seuil_qt_pos = 1e300
    //   - seulement le lobe positif : seuil_qt_neg = -1e300
    // Convention de signe : la charge collectée sur un pad est positive.
    double seuil_qt_pos =  200.0;
    double seuil_qt_neg = -200.0;

    // --- Helpers ---
    double z_grid()  const { return dkg; }
    double z_anode() const { return dkg + dga; }
  
    double sigma_T(double z) const { return DT_kg * std::sqrt(drift_distance(z)); }
    double sigma_L(double z) const { return DL_kg * std::sqrt(drift_distance(z)); }
    double sigma_T_gap(double z, double s) const { double a = sigma_T(z); return std::sqrt(a*a + DT_ga*DT_ga*s);} 
 
    double t_grid(double z) const { return drift_distance(z) / ve_kg; }
    double T_gap() const { return dga / ve_ga ; }
    // Nombre de bins de delta_t depuis la fission (t = 0). Le signal d'un voxel s'étend
    // jusqu'à t_grid(z) + 402.5 ns (fin de la table PsiTable) ; on prend 450 ns de marge.
    // Le plus grand t_grid est celui d'un voxel à la cathode : dkg / ve_kg.
    int n_bins() const { return static_cast<int>( std::ceil((dkg / ve_kg + 450.0) / delta_t)) + 1 ;}
  
  double drift_distance(double z) const {
    return dkg - z;  // dérive vers z>0, z ∈ [0, dkg]
  }

};
