#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#ifndef MATERIAL_H
#define MATERIAL_H

class Material {
 public:
  Material() ;
  void SetZ(double z) ;
  void SetAuma(double z) ;
  void SetMaterialConstant() ;
  void SetRho(double rho) ;
  void SetThick(double thick ) ;
  double GetZ() ;
  double GetA() ;
  double GetAuma() ;
  double GetI() ;
  double GetZA() ;
  double GetVF() ;
  double GetRho() ;
  double GetThick() ;

 private:
  double Z, A, Auma, Z_A, Rho, I, VF, Thick ;
} ;

#endif
