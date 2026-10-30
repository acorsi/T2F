#include "Particle.h"
#include "Material.h"
#include <utility>

#ifndef DEDX_LG_H
#define DEDX_LG_H

class DeDx_LG { 

 public:
  
  DeDx_LG(Material &Mater) ;
  
  void GetProtonCoeff() ;
  void GetDensityCoeff() ;
  std::pair<double,double> Get(Particle*, Material) ;

  double Z2 ;
  double M2 ;
  double Rho ;
  double A1 ;
  double Z1 ;
  double V1 ;
  double TKE ;
  double E_A ;
  double VFermi ;
  double Zeff ;
  double Factor ;
  double Bethe ;
  double Lindhard ;
  double Shell ;
  double Barkas ;
  double Density ;
  
  double DE_C ;
  double DE_a ;
  double DE_m ;
  double DE_d0 ;
  double DE_X0 ;
  double DE_X1 ;
  void SetParticle(Particle *) ;


 private:
  double AA[12] ;
  double Se ; //Electronic stopping
  double Sn ; //Nuclear stopping
  void StandardBethe(Particle *, Material) ;
  double LS(double Etha, double Gamma, double Zeff) ;
  double DensityEffect(double X) ;
  double DiffPhaseShift(double k, double etha, double Z1, double gamma) ;
  double ArgGamma(double Re, double Im) ;
  double Sk(double k, double Z1) ;
  double GetV2FV(double V) ;
  double GetProtonStopping() ;
  double GetEffectiveVelocity(double V1) ;
  double GetIonizationDegree(double Yr) ;
  double GetScreeningLength(double q) ;
  double GetChargeRatio(double q, double Lambda) ;
  double GetZ3Effect() ;
  double HeStopping() ;
  
} ;

#endif
