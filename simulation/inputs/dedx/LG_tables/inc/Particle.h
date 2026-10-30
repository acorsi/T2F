#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#ifndef PARTICLE_H
#define PARTICLE_H

class Particle {
public:
  Particle(double z, double a, double tke) ;
  ~Particle() ;
  double GetZ() ;
  double GetA() ;
  double GetB() ;
  double GetG() ;
  double GetTKE() ;
  void Set(double z, double a, double TKE) ;
  void SetTKE(double) ;
private:
  double Z ;
  double A ;
  double Mc2 ;
  double E_A ;
  double TKE ;
  double Beta ;
  double Gamma ; 
  double Veloc ;
} ;

#endif
