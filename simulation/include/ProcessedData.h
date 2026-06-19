#ifndef PROCESSED_DATA_H
#define PROCESSED_DATA_H
#include <iostream>
#include <vector>
//#include "FeatureSet.h"

class ProcessedData {
public:
    enum class Type {
        FEATURES,  // Mode actuel : denoising + extraction de features
        LATENT     // Mode futur : autoencoder
    };

private:
  Type type_;
  
  // Données selon le type
  //FeatureSet features_;              // Utilisé si type_ == FEATURES
  std::vector<double> latentVector_; // Utilisé si type_ == LATENT
  
public:
  // Constructeurs factory
  //static ProcessedData fromFeatures(const FeatureSet& features);
  static ProcessedData fromLatent(const std::vector<double>& latent);
  
  // Getters
  Type getType() const { return type_; }
  
  //bool isFeatures() const { return type_ == Type::FEATURES; }
  bool isLatent() const { return type_ == Type::LATENT; }
  
  //const FeatureSet& getFeatures() const;
  const std::vector<double>& getLatent() const;
  //void setTrueRange(double range) { features_.trueR = range ; }
  //void Print() {std::cout << "- range = " << features_.range << std::endl ;} ;
  
private:
  // Constructeur privé (utiliser les factory methods)
  ProcessedData(Type type);
};

#endif // PROCESSED_DATA_H
