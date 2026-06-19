#include "StragglingBohr.h"
#include <random>

StragglingBohr::StragglingBohr(Material &material) {
  
  atomicElectronDensity = 0.15569*material.GetRho()/material.GetA()*material.GetZ() ;
  

  time_t mytime ;
  mytime = time (NULL);
  int timint = mytime - (int)(mytime/1000)*1000 ;
  

  xsubi[0] = (unsigned short)(timint+pow(-1,timint)*145) ;
  xsubi[1] = (unsigned short)(timint+980) ;
  xsubi[2] = (unsigned short)(timint+pow(-1,timint)*15) ;
  
  initLandauTable() ;
}

double StragglingBohr::getStragglingBohr(int Z, double beta, double step, double factor) {
  double relativisticCorrections = (1-beta*beta/2.) / (1-beta*beta) ;
  double conversion_mm_cm = 0.1 ;
  double stStrag = sqrt(Z*Z * atomicElectronDensity * relativisticCorrections * step * conversion_mm_cm) ;
  
  /**
   *stStrag est la standard deviation du straggling.
   *Il faut maintenant tirer dans la distribution en considérant stStrag.
   *La distribution est une Landau.
   */

  //  std::cout << Z << " " << beta << " " << step << " " << stStrag << std::endl ;
  double result = rndLandau(stStrag*factor);

  return result;
  
}

void StragglingBohr::initLandauTable() {
  const int N_samples = 10000000;
  std::vector<double> samples(N_samples);
  
  // Générateur de nombres aléatoires simple
  std::mt19937 gen(12345);
  std::uniform_real_distribution<> dis(1e-10, 1.0 - 1e-10);  // Éviter 0 et 1
  
  // Générer des échantillons Landau(0,1)
  for (int i = 0; i < N_samples; i++) {
    double u = dis(gen);
    // Inverse de la CDF Landau : λ = -ln(-ln(u))
    samples[i] = -log(-log(u));
  }
  
  // Calculer moyenne
  double sum = 0;
  for (auto s : samples) sum += s;
  double mean = sum / N_samples;
  
  //  std::cout << "Moyenne Landau (avant centrage): " << mean << std::endl;
  // Devrait être ≈ 0.577
  
  // Centrer
  for (int i = 0; i < N_samples; i++) {
    samples[i] -= mean;
  }
  
  // Trier et extraire quantiles
  std::sort(samples.begin(), samples.end());
  
  for (int i = 0; i < LANDAU_TABLE_SIZE; ++i) {
    double p = (double)i / (LANDAU_TABLE_SIZE - 1);
    int idx = (int)(p * (N_samples - 1));
    landau_quantile_table[i] = samples[idx];
  }
  
  landau_table_initialized = true;
}

double StragglingBohr::rndLandau(double sigma) {
  // Fonction Landau RAPIDE avec lookup table
  // Conversion sigma -> xi
  double xi = sigma/4.02; 
  double max = 50*sigma;
  double result ;
  result = -20*sigma ;
  
  while(result<-10*sigma || result>max) {
    // Générer une variable uniforme
    double u = erand48(xsubi);
    
    // Lookup dans la table avec interpolation linéaire
    double index_float = u * (LANDAU_TABLE_SIZE - 1);
    int index = (int)index_float;
    double frac = index_float - index;
    
    double lambda;
    if (index >= LANDAU_TABLE_SIZE - 1) {
      lambda = landau_quantile_table[LANDAU_TABLE_SIZE - 1];
    } else {
      // Interpolation linéaire
      lambda = landau_quantile_table[index] * (1.0 - frac) + 
	landau_quantile_table[index + 1] * frac;
    }
    result =  xi * lambda;
  }
  return result ;
}

double StragglingBohr::getElectronicNoise(double sigma) {
  return  sigma*sqrt(-2*log(erand48(xsubi)))*cos(2*3.14159*erand48(xsubi)) ;
}
