#ifndef FRAGMENT_H
#define FRAGMENT_H

#include "Particle.h"
#include <vector>
#include <optional>
#include "ProcessedData.h"

/**
 * Fragment de fission.
 * Contient les propriétés physiques et les données de simulation.
 */
class Fragment {
private:
  // Propriétés du fragment de fission (fichier GEF)
  int A_;
  int Z_;
  double kineticEnergy_;
  int neutronMultiplicity_;
  
  // Traces et traitement
  double trueR_;
  std::vector<double> betheTrace_;
  std::vector<double> rawTrace_;
  std::vector<double> denoisedTrace_;
  std::optional<ProcessedData> processedData_;
  
public:
  Fragment(int A, int Z, double kineticEnergy, int neutronMultiplicity);

  // Redefinition << pour Print
  friend std::ostream& operator<<(std::ostream& os, const Fragment& f);
  
  // Interdit la copie (on n'en a pas besoin)
  Fragment(const Fragment&) = delete;
  Fragment& operator=(const Fragment&) = delete;
  
  // Move par défaut (générés automatiquement)
  Fragment(Fragment&&) = default;
  Fragment& operator=(Fragment&&) = default;
  
  // Destructeur par défaut
  ~Fragment() = default;
  
  // Getters propriétés physiques
  int getA() const { return A_; }
  int getZ() const { return Z_; }
  double getKineticEnergy() const { return kineticEnergy_; }
  int getNeutronMultiplicity() const { return neutronMultiplicity_; }
  
  
  /**
   * Crée un objet Particle pour calculs dE/dx
   * Pas de cache, recréé à chaque appel (négligeable en performance)
   */
  Particle createParticle() const {
    return Particle(static_cast<double>(Z_), 
		    static_cast<double>(A_), 
		    kineticEnergy_);
  }
  
  /**
   * Met à jour l'énergie cinétique
   */
  void setKineticEnergy(double tke) { kineticEnergy_ = tke; }
  
  // Getters/Setters traces
  const std::vector<double>& getRawTrace() const { return rawTrace_; }
  const std::vector<double>& getBetheTrace() const { return betheTrace_; }
  const std::vector<double>& getDenoisedTrace() const { return denoisedTrace_; }

  void setBetheTrace(const std::vector<double>& trace) { betheTrace_ = trace; }
  void setRawTrace(const std::vector<double>& trace) { rawTrace_ = trace; }
  void setDenoisedTrace(const std::vector<double>& trace) { denoisedTrace_ = trace;}// std::cout<<" Frag denois:"<< denoisedTrace_[denoisedTrace_.size()-1] << "  " << rawTrace_[rawTrace_.size()-1] << std::endl; }
  
  //void setTrueRange(const double R) { trueR_=R; }
  
  bool hasProcessedData() const { return processedData_.has_value(); }
  const ProcessedData& getProcessedData() const { return processedData_.value(); }
  void setProcessedData(const ProcessedData& data) 
    { processedData_ = data; 
      //processedData_->setTrueRange(trueR_);
    }
  
    //    printProcessedData(); }
  //void printProcessedData(){  processedData_->Print() ; }
  
  //const FeatureSet& getFeatures() const { return processedData_->getFeatures(); }
};

#endif // FRAGMENT_H
