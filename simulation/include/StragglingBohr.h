#include "IStragglingAndNoiseCalculator.h"
#include "Particle.h"
#include "Material.h"
#include <memory>
#include <time.h>
#include <gsl/gsl_rng.h>  
#include <gsl/gsl_randist.h>

#ifndef STRAGGLING_BOHR_H
#define STRAGGLING_BOHR_H

class StragglingBohr : public IStragglingAndNoiseCalculator {
 private:
  double getStragglingBohr(int Z, double KE, double step, double factor) ;
  double getElectronicNoise(double sigma) ;
  void initLandauTable() ;
  double rndLandau(double sigma) ;
  
 public:
  /**
   * Constructeur
   * @param material : materiau
   */
  StragglingBohr(Material &material) ;
  
  /**
   *Calcul du Straggling 
   * @param particle : Particule (Z,A,E)
   * @param stepSize : épaisseur de matériau en mm
   */
  double computeStraggling(Particle *particle, double stepSize, double factor=1) override {
    return getStragglingBohr(particle->GetZ(), particle->GetB(), stepSize, factor) ;
  }

  double computeElectronicNoise(double sigma) override {
    return getElectronicNoise(sigma) ;
  }
  
    std::string getMethodName() const override {
    return "StragglingBohr" ;
  }
 private:
  double atomicElectronDensity ;
  
  // Table de quantiles précalculés pour Landau
  static constexpr int LANDAU_TABLE_SIZE = 1000;
  double landau_quantile_table[LANDAU_TABLE_SIZE];
  bool landau_table_initialized;
  unsigned short xsubi[3] ;
  
};

#endif // STRAGGLING_BOHR_H
