#ifndef IENERGY_LOSS_CALCULATOR_H
#define IENERGY_LOSS_CALCULATOR_H

#include "Particle.h"
#include "Material.h"

/**
 * Interface abstraite pour le calcul de perte d'énergie.
 * Permet de supporter différentes méthodes (Bethe-Bloch, SRIM, etc.)
 */
class IEnergyLossCalculator {
public:
  virtual ~IEnergyLossCalculator() = default;

  /**
   *Initialize la particule dont on souhaite calculer la perte d'énergie
   */
  virtual void initParticle(Particle* particle) =0;
  /**
   * Calcule dE/dx pour une particule dans un matériau
   * @param particle Particule (A, Z, énergie)
   * @param material Matériau
   * @return dE/dx en MeV/mm
   */
  virtual double computeDeDx(Particle* particle, const Material& material) = 0;
  
  /**
   * Nom de la méthode (pour debug/logs)
   */
  virtual std::string getMethodName() const = 0;
  };

#endif // IENERGY_LOSS_CALCULATOR_H
