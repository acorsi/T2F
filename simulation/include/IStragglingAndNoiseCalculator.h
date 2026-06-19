#ifndef ISTRAGGLING_CALCULATOR_H
#define ISTRAGGLING_CALCULATOR_H

#include "Particle.h"
#include "Material.h"

/**
 *Interface abstraite pour calcul straggling
 */
class IStragglingAndNoiseCalculator {
 public :
  virtual ~IStragglingAndNoiseCalculator() = default ;
  virtual double computeStraggling(Particle *particle, double stepSize, double factor=1) = 0 ;
  virtual double computeElectronicNoise(double sigma) = 0 ;
  virtual std::string getMethodName() const = 0 ; 
} ;

#endif //ISTRAGGLING_CALCULATOR_H
