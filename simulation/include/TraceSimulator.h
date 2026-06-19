#ifndef TRACE_SIMULATOR_H
#define TRACE_SIMULATOR_H

#include "Fragment.h"
#include "IEnergyLossCalculator.h"
#include "IStragglingAndNoiseCalculator.h"
#include "Material.h"
#include <vector>
#include <memory>
#include <random>
#include <tuple>

class TraceSimulator {
private:
  std::unique_ptr<Material> material_;
  std::unique_ptr<IEnergyLossCalculator> energyLossCalc_;  // <- Polymorphique !
  std::unique_ptr<IStragglingAndNoiseCalculator> stragglingAndNoiseCalc_ ;
  double stepSize_;
  
  std::tuple<std::vector<double>, std::vector<double>, double> propagateFragment(int A, int Z, double initialEnergy) const;
  
public:
  /**
   * Constructeur
   * @param material Matériau
   * @param energyLossCalc Méthode de calcul (BetheBloch, SRIM, etc.)
   * @param stepSize Pas de propagation en mm
   */
  TraceSimulator(Material* material, 
		 IEnergyLossCalculator* energyLossCalc,
		 IStragglingAndNoiseCalculator* stragglingCalc,
		 double stepSize = 1.0);
  
  std::tuple<std::vector<double>, std::vector<double>, double> simulate(const Fragment& fragment) const;
  
  double getStepSize() const { return stepSize_; }
  Material* getMaterial() { return material_.get(); }
  IEnergyLossCalculator* getEnergyLossCalculator() { return energyLossCalc_.get(); }
  IStragglingAndNoiseCalculator* getStragglingCalculator() { return stragglingAndNoiseCalc_.get(); }
  mutable std::mt19937 rng_ ;
  mutable std::uniform_real_distribution<double> uniform_ ;
  
};

#endif // TRACE_SIMULATOR_H
