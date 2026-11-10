#include "PHD.h"
#include "Particle.h"
#include "Material.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "Config.h"
// ============================================================================
// Constructeur
// ============================================================================

PHD::PHD(const std::string& tables_directory, Material& material,const std::string& elosstreat)
    : tables_directory_(tables_directory),m_elosstreat(elosstreat)
{
    if (!tables_directory_.empty() && tables_directory_.back() != '/') {
        tables_directory_ += '/';
    }

    Rho = material.GetRho() ;
}

// ============================================================================
// Interface IEnergyLossCalculator
// ============================================================================

void PHD::initParticle(Particle* particle) {
    if (!particle) {
        throw std::invalid_argument("[PHD] initParticle: null particle");
    }

    int Z = particle->GetZ();

    if (!isLoaded(Z)) {
        if (!loadTableForZ(Z)) {
            throw std::runtime_error("[PHD] Failed to load table for Z=" +
                                     std::to_string(Z));
        }
    }
    //    std::cout << "InitParticle " << Z << std::endl ;
}

double PHD::computePHD(Particle* particle, const Material& material) {
    if (!particle) {
        throw std::invalid_argument("[PHD] computeDeDx: null particle");
    }

    int    Z      = particle->GetZ();
    int    A      = particle->GetA();
    double energy = particle->GetTKE(); // MeV

    if (!isLoaded(Z)) {
        throw std::runtime_error("[PHD] Table not loaded for Z=" +
                                 std::to_string(Z));
    }

    if (A <= 0) {
        throw std::invalid_argument("[PHD] Invalid mass A=" +
                                    std::to_string(A));
    }

    double kea = energy / static_cast<double>(A);  // MeV/u

    //    std::cout << "computeDeDx " << Z << " " << A << " " << kea << std::endl ;

    
    return 0.747*pow(A*interpolateKEA(tables_.at(Z), kea),0.813)+0.0233*pow(Z/45.,0.9)*(energy-A*interpolateKEA(tables_.at(Z), kea));  // MeV/(mg/cm²)
}

std::string PHD::getMethodName() const {
    return "PHD";
}

void PHD::preloadAllTables(int Z_min, int Z_max) {
    int success = 0;
    for (int Z = Z_min; Z <= Z_max; Z++) {
        if (!isLoaded(Z)) {
            if (loadTableForZ(Z)) {
                success++;
            }
        } else {
            success++;
        }
    }
    // std::cout << "[PHD] Preloaded " << success << "/"
    //           << (Z_max - Z_min + 1) << " Z values" << std::endl;
}

// ============================================================================
// Méthodes privées
// ============================================================================

bool PHD::loadTableForZ(int Z) {
    // Nom de fichier : Z0XX_dedx.txt
    std::ostringstream filename;
    filename<<"phd_"<<Z<<".dat";

    std::ifstream file(tables_directory_ + filename.str());
    if (!file.is_open()) {
        std::cerr << "[PHD] Cannot open: " << filename.str() << std::endl;
        return false;
    }

    RealTable table;
    table.Z = Z;

    std::string line;
    while (std::getline(file, line)) {
        // Ignorer commentaires et lignes vides
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        double kea, dedx;

        if (iss >> kea >> dedx) {
            table.kea.push_back(kea);
            table.dedx.push_back(dedx);
        }
    }

    file.close();

    if (table.kea.empty()) {
        std::cerr << "[PHD] No data in: " << filename.str() << std::endl;
        return false;
    }

    table.num_points = static_cast<int>(table.kea.size());

    // S'assurer que la table est triée en ordre croissant en KE/A
    bool sorted = true;
    for (int i = 1; i < table.num_points; i++) {
        if (table.kea[i] < table.kea[i-1]) {
            sorted = false;
            break;
        }
    }

    if (!sorted) {
        std::vector<int> idx(table.num_points);
        for (int i = 0; i < table.num_points; i++) idx[i] = i;

        std::sort(idx.begin(), idx.end(), [&](int i, int j){
            return table.kea[i] < table.kea[j];
        });

        auto reorder = [&](std::vector<double>& v) {
            std::vector<double> tmp(v.size());
            for (size_t i = 0; i < idx.size(); i++) tmp[i] = v[idx[i]];
            v = tmp;
        };

        reorder(table.kea);
        reorder(table.dedx);
    }

    // std::cout << "[PHD] Loaded Z=" << Z
    //           << "  (" << table.num_points << " pts, KE/A: "
    //           << table.kea.front() << " - " << table.kea.back()
    //           << " MeV/u)" << std::endl;

    tables_[Z] = std::move(table);
    return true;
}

double PHD::interpolateKEA(const RealTable& table, double kea) const {
    const int n = static_cast<int>(table.kea.size());

    if (n < 2) {
        std::cerr << "[PHD] Table too small Z=" << table.Z
                  << " n=" << n << std::endl;
        return 0.0;
    }

    if (kea <= 0.0) return 0.0;

    // Extrapolation basse energie
    if (kea <= table.kea.front()) {
        double k1 = table.kea[0];
        double k2 = table.kea[1];
        double d1 = table.dedx[0];
        double d2 = table.dedx[1];
        double slope = (d2 - d1) / (k2 - k1);
        return std::max(0.0, d1 + slope * (kea - k1));
    }

    // Extrapolation haute energie
    if (kea >= table.kea.back()) {
        double k1 = table.kea[n-2];
        double k2 = table.kea[n-1];
        double d1 = table.dedx[n-2];
        double d2 = table.dedx[n-1];
        double slope = (d2 - d1) / (k2 - k1);
        return std::max(0.0, d2 + slope * (kea - k2));
    }

    // Recherche binaire
    int left  = 0;
    int right = n - 1;

    while (right - left > 1) {
        int mid = (left + right) / 2;
        if (kea > table.kea[mid]) {
            left = mid;
        } else {
            right = mid;
        }
    }

    double k1 = table.kea[left];
    double k2 = table.kea[right];
    double d1 = table.dedx[left];
    double d2 = table.dedx[right];

    double t = (kea - k1) / (k2 - k1);
    //    std::cout << "Interpolate : " << d1 + t * (d2 - d1) << std::endl ;
    
    return d1 + t * (d2 - d1);
}

bool PHD::isLoaded(int Z) const {
    return tables_.find(Z) != tables_.end();
}
