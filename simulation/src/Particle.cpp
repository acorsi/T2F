#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#include "Particle.h"
#include "Var.h"

using namespace std ;

////////////////////////////////////////////
Particle::Particle(double z, double a, double tke) {
  Z=z ;
  A=a ;
  TKE = tke ;
  Mc2 = A*Mu ;
  E_A=TKE/A ;
  Gamma=TKE/Mc2+1. ;
  Beta=sqrt(1.-1./Gamma/Gamma) ;
  Veloc=Beta*clum ;
  
  //  cout << Z << " " << A << " " << E_A << " "
  //       << Beta << " " << Gamma << endl ;
}

Particle::~Particle() {

}

//////////////////////////////////////////////////

void Particle::Set(double z, double a, double tke) {
  //  cout << "     Set Particle Z=" << z << " - A=" << a << " - TKE=" << tke << endl ;
  Z=z ;
  A=a ;
  TKE = tke ;
  Mc2 = A*Mu ;
  E_A=TKE/A ;
  Gamma=TKE/Mc2+1. ;
  Beta=sqrt(1.-1./Gamma/Gamma) ;
  Veloc=Beta*clum ;
}

void Particle::SetTKE(double tke) {
  TKE = tke ;
  E_A=TKE/A ;
  Gamma=TKE/Mc2+1. ;
  Beta=sqrt(1.-1./Gamma/Gamma) ;
}
//////////////////////////////////////////////////

double Particle::GetA() {
  
  return A ;

} ;

//////////////////////////////////////////////////

double Particle::GetZ() {
  
  return Z ;

} ;

//////////////////////////////////////////////////

double Particle::GetB() {
  
  return Beta ;

} ;

//////////////////////////////////////////////////

double Particle::GetG() {
  
  return Gamma ;

} ;

//////////////////////////////////////////////////

double Particle::GetTKE() {
  
  return TKE ;

} ;
