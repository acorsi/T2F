#ifndef TRACE_SIMULATOR_H
#define TRACE_SIMULATOR_H

#include "Fragment.h"
#include "PHD.h"
#include "DeDx_REAL.h"
#include "IStragglingAndNoiseCalculator.h"
#include "Material.h"
#include <vector>
#include <memory>
#include <random>
#include <tuple>

class TraceSimulator {
private:
  std::unique_ptr<Material> material_;
  std::unique_ptr<DeDx_REAL> energyLossCalc_;  // <- Polymorphique !
  std::unique_ptr<PHD> PHDCalc_;  // <- Polymorphique !
  std::unique_ptr<IStragglingAndNoiseCalculator> stragglingAndNoiseCalc_ ;
  double stepSize_;
  
  std::tuple<std::vector<double>, std::vector<double>, double, std::vector<double>> propagateFragment(int A, int Z, double initialEnergy) const;
  
public:
  /**
   * Constructeur
   * @param material Matériau
   * @param energyLossCalc Méthode de calcul (BetheBloch, SRIM, etc.)
   * @param stepSize Pas de propagation en mm
   */
  TraceSimulator(Material* material, 
		 DeDx_REAL* energyLossCalc,
		 IStragglingAndNoiseCalculator* stragglingCalc,
		 double stepSize, PHD* phdCalc);
  
  std::tuple<std::vector<double>, std::vector<double>, double, std::vector<double>> simulate(const Fragment& fragment) const;
  
  double getStepSize() const { return stepSize_; }
  Material* getMaterial() { return material_.get(); }
  DeDx_REAL* getEnergyLossCalculator() { return energyLossCalc_.get(); }
  PHD* getPHDCalculator() { return PHDCalc_.get(); }
  IStragglingAndNoiseCalculator* getStragglingCalculator() { return stragglingAndNoiseCalc_.get(); }
  mutable std::mt19937 rng_ ;
  mutable std::uniform_real_distribution<double> uniform_ ;
  
};

#endif // TRACE_SIMULATOR_H
