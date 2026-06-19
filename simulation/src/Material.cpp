#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#include "Material.h"

using namespace std ;

Material::Material() {
  ifstream in ;
  in.open("../data/input.tpc",ios::in) ;
  if(!in) {
    cout << endl << "File ../../data/input.tpc not found." << endl ;
    exit(1) ;
  }

  double w ;
  in >> Z >> Auma >> w ;
  in >> Rho ;
  in >> Thick ;
  in.close() ;

  
  SetMaterialConstant() ;
  
  std::cout << "     # Density=" << Rho
	    << " g/cm^3" ;
  std::cout << " - Thickness=" << Thick << " mm" << endl ;
  
}

////////////////////////////////////////////

void Material::SetZ(double z) {
  Z=z ;
} ;

////////////////////////////////////////////

void Material::SetAuma(double a) {
  Auma=a ;
} ;

////////////////////////////////////////////

void Material::SetRho(double rho) {
  Rho = rho ;
} ;

///////////////////////////////////////////

void Material::SetThick(double thick ) {
  Thick = thick ;
} ;

////////////////////////////////////////////

void Material::SetMaterialConstant() {
  double zz, za, i, rho, vf, ag, ap ;
  char line[256] ;
  ifstream in ;
  
  in.open("../data/MaterialConstant.d",ios::in) ;
  if(!in) {
    cout << "Material Constant File Not Found. Exit" << endl ;
    exit(1) ;
  }

  in.getline(line,256) ;
  while(1) {
    in >> zz ;
    if(in.eof()) {
      cout << "Element Not Found in Material File. Exit." << endl ;
      exit(1) ;
    }
    in >> line >> line >> za >> i >> rho >> vf ;
    if(abs(zz-Z)<0.1) break ;
  }
  Z_A=Z/Auma ;
  VF=vf ;
  I=i ;
  //  Auma = Z/Z_A ;
  ag=ceil(Auma) ;
  ap=floor(Auma) ;
  if(abs(Auma-ag)<abs(Auma-ap))
    A=ag ;
  else
    A=ap ;
  //  cout << Z << " " << I << " " << VF << " " << Z_A << " " << A << " " << Auma << endl ;
} ;

////////////////////////////////////////////

double Material::GetZ() {
  return Z ;
} ;

////////////////////////////////////////////

double Material::GetZA() {
  return Z_A ;
} ;

////////////////////////////////////////////

double Material::GetA() {
  return A ;
} ;

////////////////////////////////////////////

double Material::GetAuma() {
  return Auma ;
} ;

////////////////////////////////////////////

double Material::GetI() {
  return I ;
} ;

////////////////////////////////////////////

double Material::GetVF() {
  return VF ;
} ;

////////////////////////////////////////////

double Material::GetRho() {
  return Rho ;
} ;

////////////////////////////////////////////

double Material::GetThick() {
  return Thick ;
} ;

////////////////////////////////////////////
