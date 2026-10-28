#pragma once

#include <cmath>
#include <random>
#include "detector.hpp"

// ---------------------------------------------------------------------------
// Direction d'un fragment
//
// Type1 (EmissionAxis::Z) : theta par rapport à Z
//   ux = sin(theta)*cos(phi)
//   uy = sin(theta)*sin(phi)
//   uz = cos(theta)          ← composante principale
//
// Type2 (EmissionAxis::Y) : theta par rapport à Y
//   ux = sin(theta)*cos(phi)
//   uy = cos(theta)          ← composante principale
//   uz = sin(theta)*sin(phi)
// ---------------------------------------------------------------------------

struct Direction {
    double theta;   // polaire [0, pi/2] par rapport à l'axe d'émission
    double phi;     // azimutal [0, 2pi]

    // Direction opposée (second fragment)
    Direction opposite() const {
        return { M_PI - theta, phi + M_PI };
    }

    // Vecteur unitaire cartésien selon l'axe d'émission
    double ux(EmissionAxis axis) const {
        return std::sin(theta) * std::cos(phi);
    }

    double uy(EmissionAxis axis) const {
        if (axis == EmissionAxis::Y) return std::cos(theta);
        return std::sin(theta) * std::sin(phi);
    }

    double uz(EmissionAxis axis) const {
        if (axis == EmissionAxis::Z) return std::cos(theta);
        return std::sin(theta) * std::sin(phi);
    }
};

// Tire une direction uniforme sur la sphère complète (4π), en excluant la
// bande équatoriale [80°,100°] des deux côtés (trajectoires rasantes)
inline Direction draw_direction(std::mt19937_64& rng) {
    const double cos_theta_min = std::cos(80.0 * M_PI / 180.0);  // ≈ 0.17365
    std::uniform_real_distribution<double> dist_mag(cos_theta_min, 1.0);
    std::uniform_real_distribution<double> dist_phi(0.0, 2.0 * M_PI);
    std::uniform_int_distribution<int>     dist_sign(0, 1);

    double mag       = dist_mag(rng);
    double cos_theta = (dist_sign(rng) == 0) ? mag : -mag;
    double phi       = dist_phi(rng);
    return { std::acos(cos_theta), phi };
}
