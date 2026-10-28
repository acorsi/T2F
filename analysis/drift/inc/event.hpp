#pragma once

#include <vector>
#include "direction.hpp"

// ---------------------------------------------------------------------------
// Données d'un event en entrée
// ---------------------------------------------------------------------------

struct FragmentInput {
    std::vector<double> raw_trace;  // dE par voxel (keV), taille = N_voxels
    int32_t Z, A;
    double  KE;
};

struct EventInput {
    FragmentInput H;  // fragment lourd (heavy)
    FragmentInput L;  // fragment léger (light)
};

// ---------------------------------------------------------------------------
// Données d'un event en sortie
// ---------------------------------------------------------------------------

struct FragmentOutput {
    Direction dir;          // direction tirée (ou lue depuis fichier)
    double vx, vy, vz;     // vertex : (vx,vy) en type1, (vx,vz) en type2, vy=0

    // Trace 3D dans le détecteur : un point par voxel
    // taille = N_voxels effectifs (arrêtés à la grille)
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> dE;          // énergie déposée par voxel (MeV)

    // Dérive électronique : un par voxel, calculé à la grille
    std::vector<double> n_electrons;
    std::vector<double> sigma_T;
    std::vector<double> sigma_L;
};

struct EventOutput {
    FragmentOutput H;
    FragmentOutput L;
};
