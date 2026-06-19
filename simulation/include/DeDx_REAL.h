#ifndef DEDX_REAL_H
#define DEDX_REAL_H

#include "IEnergyLossCalculator.h"
#include <vector>
#include <map>
#include <string>

/**
 * Calculateur de perte d'énergie basé sur les tables experimentales.
 *
 * Pour chaque Z, une seule table est chargée depuis ELoss_ExpTables/Z0XX_dedx.txt
 * Format des fichiers : deux colonnes KE/A [MeV/u] et dE/dx [MeV/(mg/cm²)]
 *
 * Le dE/dx pour un fragment (Z, A, E) est obtenu par :
 *   1. Calcul de E/A
 *   2. Interpolation dans la table du Z correspondant
 *
 * Retourne dE/dx en MeV/(mg/cm²).
 */
class DeDx_REAL : public IEnergyLossCalculator {
public:
    /**
     * Constructeur
     * @param tables_directory  Répertoire contenant les fichiers Z0XX_dedx.txt
     * @param material          Matériau cible (pour compatibilité interface)
     */
    DeDx_REAL(const std::string& tables_directory, Material& material, const std::string& elosstreat);

    /**
     * Destructeur
     */
    ~DeDx_REAL() override = default;

    /**
     * Charge la table pour le Z de la particule.
     * Appeler avant computeDeDx() pour chaque nouveau type de fragment.
     */
    void initParticle(Particle* particle) override;

    /**
     * Calcule dE/dx par interpolation en KE/A.
     * @return dE/dx en MeV/(mg/cm²)
     */
    double computeDeDx(Particle* particle, const Material& material) override;

    /**
     * Nom de la méthode
     */
    std::string getMethodName() const override;
  /**
     * Pré-charge toutes les tables pour Z_min à Z_max.
     * Utile en mode parallèle pour éviter le chargement concurrent.
     */
    void preloadAllTables(int Z_min = 20, int Z_max = 80);

private:
    // -----------------------------------------------------------------------
    // Structures internes
    // -----------------------------------------------------------------------

    /**
     * Table experimentale pour un Z donné.
     * KE/A stocké en ordre croissant.
     */
    struct RealTable {
        int Z;
        int num_points;
        std::vector<double> kea;   // KE/A [MeV/u], croissant
        std::vector<double> dedx;  // dE/dx [MeV/(mg/cm²)]
    };

    // -----------------------------------------------------------------------
    // Membres privés
    // -----------------------------------------------------------------------
    float Rho ;
    std::string tables_directory_;

    /** Tables indexées par Z */
    std::map<int, RealTable> tables_;

    // -----------------------------------------------------------------------
    // Méthodes privées
    // -----------------------------------------------------------------------

    /** Charge la table pour un Z */
    bool loadTableForZ(int Z);

    /**
     * Interpole dE/dx en KE/A dans une table.
     * Extrapolation linéaire si KE/A hors plage.
     */
    double interpolateKEA(const RealTable& table, double kea) const;

    /** Vérifie si la table pour ce Z est chargée */
    bool isLoaded(int Z) const;
    std::string m_elosstreat;
};

#endif // DEDX_REAL_H
