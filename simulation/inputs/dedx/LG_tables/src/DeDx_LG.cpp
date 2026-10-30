#include <cmath>

#include "../inc/DeDx_LG.h"
#include "../inc/Var.h"

using namespace std ;

DeDx_LG::DeDx_LG(Material &Mater) {
  Z2=Mater.GetZ() ;
  M2=Mater.GetAuma() ;
  VFermi=Mater.GetVF() ;
  Rho=Mater.GetRho() ;

  GetProtonCoeff() ;
  GetDensityCoeff() ;

}

void DeDx_LG::SetParticle(Particle *Beam) {
  A1=Beam->GetA() ;
  Z1=Beam->GetZ() ;
  V1=Beam->GetB()*clum ;
  TKE=Beam->GetTKE() ;
  V1=sqrt(TKE/A1/0.025) ; //In units of V0=25keV/uma
  E_A=TKE/A1 ; 
}

void DeDx_LG::GetProtonCoeff() {
  
  long double AA1, AA2, AA3, AA4, AA5, AA6, AA7, AA8;// AA9, AA10, AA11, AA12 ;
  ifstream in ;
  in.open("../../data/PCoeff.dat",ios::in) ;
  if(!in) {
    cout << "Proton energy loss file does not exist. Exit." << endl ;
    exit(1) ;
  }
  
  
  if(Z2>92) exit(1) ;
  int ZZ ;
  while(1) {
    in >> ZZ ;
    //    cout << ZZ ;
    if(in.eof()) exit(1) ;
    in >> AA1 >> AA2 >> AA3 >> AA4 >> AA5 >> AA6 >> AA7 >> AA8 ;
    //    cout << " " << AA1 << endl ;
    //    if(ZZ==60) exit(1) ;
    if(abs(ZZ-Z2)<0.1) break ;
  }
  in.close() ;
  AA[0] = AA1 ;
  AA[1] = AA2 ;
  AA[2] = AA3 ;
  AA[3] = AA4 ;
  AA[4] = AA5 ;
  AA[5] = AA6 ;
  AA[6] = AA7 ;
  AA[7] = AA8 ;

  // cout << AA[0] << " " 
  //      << AA[1] << " " 
  //      << AA[2] << " " 
  //      << AA[3] << " " 
  //      << AA[4] << " " 
  //      << AA[5] << " " 
  //      << AA[6] << " " 
  //      << AA[7] << endl ;
}

void DeDx_LG::GetDensityCoeff() {

  ifstream in ;

  in.open("../../data/DensityEffect.d",ios::out) ;
  if(!in) {
    cout << endl << "Density Effect File Not Found. Exit." << endl ;
    exit(1) ;
  }
  
  double var, a, m, d0, X0, X1, C ;
  char line[256] ;
  in.getline(line,256) ;
  int ZZ ;
  while(1) {
    in >> ZZ ;
    if(in.eof()) {
      cout << endl << "Element Not Found In Density Effect. Exit" << endl ;
      exit(1) ;
    }
    in >> var >> var >> var >> var >> var ;
    in >> C >> X0 >> X1 >> a >> m >> d0 ;
    if(abs(ZZ-Z2)<0.1) break ;
  }

  DE_C = C ;
  DE_X0 = X0 ;
  DE_X1 = X1 ;
  DE_a = a ;
  DE_m = m ;
  DE_d0 = d0 ;
  in.close() ;

}

void DeDx_LG::StandardBethe(Particle *Beam, Material Mater) {

  ///////////////////////////////////////////////////
  //      FOR DETAILS ON STANDARD BETHE SEE        //
  //        B.A. Weaver and A.J. Westphal          //
  //      Nucl. Instr. Meth. B 187, 285 (2002)     //
  //            -------------------------          //
  //  The density effect treatment is taken from   //
  //R.N. Sternheimer, M.J. Berger and S.M. Seltzer //
  //Atom. Data. Nucl. Data Tables 30, 261 (1984)   //
  //            -------------------------          //
  //Proton stopping power parameterization is from //
  // H.H.Andersen and J.F. Ziegler "The stopping   //
  // and ranges of ions in matter - Vol3"          //
  // Pergamon (1977)                               //
  //            -------------------------          //
  // The details on the scaling of proton stopping //
  // powers in order to obtain heavy ion stopping  //
  // is also from                                  //
  // H.H.Andersen and J.F. Ziegler "The stopping   //
  // and ranges of ions in matter - Vol3"          //
  // Pergamon (1977)                               //
  //            -------------------------          //
  // The He stopping powers are from               //
  // H.H.Andersen and J.F. Ziegler "The stopping   //
  // and ranges of ions in matter - Vol3"          //
  // Pergamon (1977)                               //
  ///////////////////////////////////////////////////
  

  //INCIDENT PARTICLE EFFECTIVE CHARGE
  double Az, Bz ;
  Az = 1.16 - 1.91e-3*Mater.GetZ() + 1.26e-5*Mater.GetZ()*Mater.GetZ() ;
  Bz = 1.18 - 7.50e-3*Mater.GetZ() + 4.53e-5*Mater.GetZ()*Mater.GetZ() ;
  Zeff = Beam->GetZ()*(1-Az*exp(-Bz*Beam->GetB()/Alpha/pow(Beam->GetZ(),2./3.))) ;

  //DEFINES FACTOR OF DeDx
  Factor = Constant_K*Mater.GetZA()*Zeff*Zeff/Beam->GetB()/Beam->GetB() ;

  //BETHE PART OF DEDX
  Bethe = log(2*Mec2*Beam->GetB()*Beam->GetB()*Beam->GetG()*Beam->GetG()) ;
  Bethe -= log(Mater.GetI()) + Beam->GetB()*Beam->GetB() ;
  
  //LINDHARD-SORENSEN CORRECTION TO DEDX
  double Etha ;
  Etha=Alpha*Zeff/Beam->GetB() ;
  Lindhard = LS(Etha,Beam->GetG(),Zeff) ;

  //SHELL CORRECTION
  double nu, I ;
  I = Mater.GetI() ;
  nu = Beam->GetB()*Beam->GetG() ;
  Shell=( (4.22377e-3/pow(100*nu,2.) +
	   3.04043/pow(100*nu,4) -
	   3.8106e2/pow(100*nu,6.))*I*I +
	  (3.850190e-5/pow(100*nu,2.) -
	   1.667989e-2/pow(100*nu,4) +
	   1.57955/pow(100*nu,6))*I*I*I ); 
  
  //BARKAS CORRECTION
  double V ;
  V=Beam->GetB()*Beam->GetG()/Alpha/sqrt(Mater.GetZ()) ;
  Barkas = 1+2.*Zeff/sqrt(Mater.GetZ())*GetV2FV(V) ;
  
  //DENSITY EFFECT
  Density=DensityEffect(log10(Beam->GetB()*Beam->GetG())) ;
  //    Density=0. ;
}

pair<double,double> DeDx_LG::Get(Particle *Beam, Material Mater) {
  TKE=Beam->GetTKE() ;
  if(TKE<=0) return make_pair(0,0) ; 


  if(abs(Z1-1)<0.1) {
    if(TKE/A1>10) {
      StandardBethe(Beam,Mater) ;
      Se = Factor*(Bethe*Barkas+Lindhard-Shell/Z2-Density/2.) ;
    }else
      Se = GetProtonStopping() ;

    ///////////////////////////////////
    //CALCULATION OF NUCLEAR STOPPING//
    ///////////////////////////////////
    double EPS ;
    EPS=32.53*M2*TKE*1000/(Z1*Z2*(A1+M2)*(pow(Z1,0.23)+pow(Z2,0.23))) ;
    if(EPS>=30) {
      Sn = log(EPS)/2./EPS ;
    }else{
      double var ;
      var = 0.01321*pow(EPS,0.21226)+0.19593*pow(EPS,0.5) ;
      Sn = 0.5*log(1+1.1383*EPS)/(EPS+var) ;
    }
    
    //Unit convertion
    Sn=Sn*Z1*Z2*A1*8.462/((A1+M2)*(pow(Z1,0.23)+pow(Z2,0.23)))*0.60222/M2*1000 ; //MeV/(g/cm**2)
    
    return make_pair(Se/1000., Sn/1000.) ; //MeV/(mg/cm^2)
    
  }else if(Z1>2.1) {
    
    if(TKE/A1>10) {
      StandardBethe(Beam,Mater) ;
      return make_pair(Factor*(Bethe*Barkas+Lindhard-Shell/Z2-Density/2.)/1000.,0) ; //MeV/(mg/cm^2)
    }

    V1=sqrt(TKE/A1/0.025) ; //In units of V0=25keV/uma
    
    double SeRef ;
    //    SeRef = GetProtonStopping() ;
    
    double Yr ;
    Yr = GetEffectiveVelocity(V1) ;
    
    double q ;
    q = GetIonizationDegree(Yr) ;

    double Lambda ;
    Lambda = GetScreeningLength(q) ;

    double QRatio ;
    QRatio = GetChargeRatio(q,Lambda) ;

    double Z3Effect ;
    Z3Effect=GetZ3Effect() ;

    //LOW ENERGIE PART OF STOPPING POWER
    if(abs(Yr-0.13)<0.001 || abs(Yr-1./pow(Z1,2./3.))<0.001) {
      double VRmin ;
      if(Yr*pow(Z1,2./3.)>1.)
	VRmin = Yr*pow(Z1,2./3.) ;
      else VRmin = 1. ;
      double VMin ;
      VMin = 0.5*VRmin ;
      if(VRmin*VRmin-0.8*VFermi*VFermi>0)
	VMin += 0.5*sqrt(VRmin*VRmin-0.8*VFermi*VFermi) ;
      double Old=E_A ;
      E_A=25*VMin*VMin/1000. ;
      SeRef = GetProtonStopping() ;
      
      double Power ;
      Power=0.5 ;
      if( abs(Z2-6)<0.1 || ( (abs(Z2-14)<0.1 || abs(Z2-32)<0.1) && abs(Z1-19)<0.1 ) ) Power = 0.375 ;

      Se = SeRef*pow(Z3Effect*QRatio*Z1,2.)*pow(Old/E_A,Power) ;

    }else {
      SeRef = GetProtonStopping() ;
      
      Se = SeRef*pow(Z3Effect*QRatio*Z1,2.) ;
    }
    
    ///////////////////////////////////
    //CALCULATION OF NUCLEAR STOPPING//
    ///////////////////////////////////
    double EPS ;
    EPS=32.53*M2*TKE*1000/(Z1*Z2*(A1+M2)*(pow(Z1,0.23)+pow(Z2,0.23))) ;
    if(EPS>=30) {
      Sn = log(EPS)/2./EPS ;
    }else{
      double var ;
      var = 0.01321*pow(EPS,0.21226)+0.19593*pow(EPS,0.5) ;
      Sn = 0.5*log(1+1.1383*EPS)/(EPS+var) ;
    }
    
    //Unit convertion
    Sn=Sn*Z1*Z2*A1*8.462/((A1+M2)*(pow(Z1,0.23)+pow(Z2,0.23)))*0.60222/M2*1000 ; //MeV/(g/cm**2)

    //    cout << Se/1000. << " " << Sn/1000. << endl ; 
    
    return make_pair(Se/1000.,Sn/1000.) ; //MeV/(mg/cm^2)
  }else{

    ///////////////////////////
    //   HE STOPPING POWER   //
    ///////////////////////////

    if(TKE/A1>10) {
      StandardBethe(Beam,Mater) ;
      Se = Factor*(Bethe*Barkas+Lindhard-Shell/Z2-Density/2.) ;
    }else
     Se = HeStopping() ;
    ///////////////////////////////////
    //CALCULATION OF NUCLEAR STOPPING//
    ///////////////////////////////////
    double EPS ;
    EPS=32.53*M2*TKE*1000/(Z1*Z2*(A1+M2)*(pow(Z1,0.23)+pow(Z2,0.23))) ;
    if(EPS>=30) {
      Sn = log(EPS)/2./EPS ;
    }else{
      double var ;
      var = 0.01321*pow(EPS,0.21226)+0.19593*pow(EPS,0.5) ;
      Sn = 0.5*log(1+1.1383*EPS)/(EPS+var) ;
    }
    
    //Unit convertion
    Sn=Sn*Z1*Z2*A1*8.462/((A1+M2)*(pow(Z1,0.23)+pow(Z2,0.23)))*0.60222/M2*1000 ; //MeV/(g/cm**2)
    
    return make_pair(Se/1000.,Sn/1000.) ; //MeV/(mg/cm^2)
    
  }
}

double DeDx_LG::GetEffectiveVelocity(double V1) {
  double Vr ;

  if(V1>=VFermi)
    Vr = V1*(1.+0.2*pow(VFermi/V1,2.)) ;
  else
    Vr = 0.75*VFermi*(1.+2./3.*pow(V1/VFermi,2.) - pow(V1/VFermi,4.)/15.) ;


  if(Vr<0.13*pow(Z1,2./3.)) Vr=0.13*pow(Z1,2./3.) ;
  if(Vr<1) Vr=1 ; 
  return Vr/pow(Z1,2./3.) ;

  // if(Vr/pow(Z1,2./3.)>0.13) 
  //   return Vr/pow(Z1,2./3.) ;
  // else
  //   return 0.13 ;

} 

double DeDx_LG::GetIonizationDegree(double Yr) {
  double C1, C2, C3, C4 ;
  double Q ;
  C1= 0.8030 ;
  C2=-1.3167 ;
  C3=-0.38157 ;
  C4=-0.008983 ;
  
  Q = (1.-exp( C1*pow(Yr,0.3)
	       + C2*pow(Yr,0.6)
	       + C3*Yr
	       + C4*Yr*Yr )) ;
  if(Q<0) Q=0 ;
  if(Q>1) Q=1 ;

  //  cout << Yr << " " << Q << endl ;

  return Q ;

}


double DeDx_LG::GetScreeningLength(double q) {
  double B, BMax ;
  double L0, L1, LMin ;
  double QMin, QMax ;
  double zz ;
  if(Z1<16) zz=Z1; else zz=16;
    
  if(0.32>0.12+0.025*Z1)
    BMax=0.32 ;
  else
    BMax=0.12+0.025*Z1 ;

  if(0.43<BMax)
    B=0.43/pow(Z1,0.333) ;
  else
    B=BMax/pow(Z1,0.333) ;

  if(1.2<0.6+Z1/30.)
    LMin=1.2 ;
  else
    LMin=0.6+Z1/30. ;
  L0 = (0.8-q*LMin)/pow(Z1,0.333) ;

  if(0.9-0.025*Z1<0)
    QMin = 0 ;
  else
    QMin = 0.9-0.025*Z1 ;

  QMax = 1-0.025*zz ;
    
  
  if(q<0.2) {
    if(L0<0)
      return 0 ;
    else
      return L0 ;
  }else if(q<QMin) {
    L1=B*(q-0.2)/abs(QMin-0.2) ;
  }else if(q<QMax){
    L1=B ;
  }else{
    L1=B*(1-q)/(0.025*zz) ;
  }

  if(L1>L0)
    return L1 ;
  else
    return L0*1.03 ;
}

double DeDx_LG::GetChargeRatio(double q, double Lambda) {

  //  return q+0.5*(1.-q)/pow(VFermi,2.)*log(1+pow(2.*Lambda*VFermi,2.)) ;
  return q+0.5*(1.-q)/pow(VFermi,2.)*log(1+pow(4.*Lambda*VFermi/1.919,2.)) ;
  
}

double DeDx_LG::GetZ3Effect() {
  double a ;
  
  if(log(TKE*1000./A1)<0)
    a=7.6 ;
  else
    a=(7.6-log(TKE*1000./A1));
  a=-a*a ;

  return 1. + 1/Z1/Z1*(0.18+0.0015*Z2)*exp(a) ;

}

double DeDx_LG::LS(double etha, double gamma, double Z1) {

  double Term1, Term2, Term3 ;
  double Fact1, Fact2 ;
  double Sum ;
  
  Sum=0. ;
  for(double k=1;k<50;k+=1.00000) {
    if(k<1.1)
      Term1=0 ;
    else {
      Fact1 = DiffPhaseShift(k, etha, Z1, gamma) ;
      Term1 = k/etha/etha * (k-1.)/(2.*k-1.) *sin(Fact1)*sin(Fact1) ; 
    }
    
    Fact2 = DiffPhaseShift(-k, etha, Z1, gamma) ;
    Term2 = k/etha/etha * (k+1)/(2*k+1) *sin(Fact2)*sin(Fact2) ;
    
    Term3 = k/(4*k*k-1) * 1./(pow(gamma*k,2) + etha*etha) - 1./k ;
      
    Sum += Term1+Term2+Term3 ;
    
  }
  
  return Sum + 0.5*(1-1./gamma/gamma) ;

}

double DeDx_LG::DensityEffect(double X) {
  //  double Z ;
  //  double C, X0, X1, a, m, d0 ;
  //  char line[256] ;

  if(X<=DE_X0) 
    return DE_d0*pow(10,2*(X-DE_X0)) ;
  else if(X<DE_X1)
    return 4.6052*X+DE_a*pow(DE_X1-X,DE_m)-DE_C ;
  else
    return 4.6052*X-DE_C ;

}

double DeDx_LG::DiffPhaseShift(double k, double etha, double Z1, double gamma) {

  double Term1, Term2, Term3, Term4, Term5 ;
  double sk, sk_1 ;
  double Fact1, Fact2 ;
    
  sk = Sk(k,Z1) ;
  sk_1 = Sk(k-1,Z1) ;
  
  Fact1 = etha*(sk_1-sk)/(sk*sk_1+etha*etha) ;
  Term1 = 0.5*atan(Fact1) ;

  Fact2 = etha/gamma/(k*(k-1)+pow(etha/gamma,2.)) ;
  Term2 = 0.5*atan(Fact2) ;

  Term3 = ArgGamma(sk+1,etha) ;
  
  Term4 = ArgGamma(sk_1+1,etha) ;
  
  Term5 = acos(-1.)/2.*(sk-sk_1-k/abs(k)) ;
  
  return Term1+Term2-Term3+Term4-Term5 ;

}

double DeDx_LG::ArgGamma(double Re, double Im) {

  double T, B;
  double C[7] ;

  double Res ;

  Re = Re-1. ;
  C[0] = 1.000000000190015 ;
  C[1] = 76.18009172947146 ;
  C[2] = -86.50532032941677 ;
  C[3] = 24.01409824083091 ;
  C[4] = -1.231739572450155 ;
  C[5] = 0.1208650973866179e-2 ;
  C[6] = -0.5395239384943e-5 ;

  T=0 ;
  B=C[0] ;
  for(int i=1;i<7;i++) {
    T+= -Im*C[i]/(pow(Re+i,2.)+Im*Im) ;
    B+= C[i]*(Re+i)/(pow(Re+i,2.)+Im*Im) ;
  }
  
  // //Log(|gamma|)
  // Res = 0.5*(Re+0.5)*log(pow(Re+5.5,2) + Im*Im) 
  //   - Im*atan(Im/(Re+5.5)) -(Re+5.5)
  //   + log(sqrt(2*acos(-1.)*(T*T+B*B))) ;

  //Arg(gamma)
  Res = Im/2.*log( pow(Re+5.5,2) + Im*Im ) +
    (Re+0.5)*atan(Im/(Re+5.5)) - Im +
    atan(T/B) ;
  
  //  cout << "===>" << Res << endl ;
  // if(Res>acos(-1)) {
  //   Res = Res - ceil(Res/acos(-1))*acos(-1) ;
  // }
  return Res ;

}

double DeDx_LG::Sk(double k, double Z1) {

  return sqrt(k*k-pow(Alpha*Z1,2.)) ;

}

double DeDx_LG::GetV2FV(double V) {
  
  double V2F[10] ;
  V2F[0] = 0.33 ;
  V2F[1] = 0.31 ;
  V2F[2] = 0.27 ;
  V2F[3] = 0.23 ;
  V2F[4] = 0.21 ;
  V2F[5] = 0.19 ;
  V2F[6] = 0.17 ;
  V2F[7] = 0.16 ;
  V2F[8] = 0.15 ;
  V2F[9] = 0.14 ;
  
  int indx = (int)ceil(V)-1 ;
  if(indx>9) return 0.13/V/V ;
  else if(indx<1) return 0.33/V/V ;
  else{
    double a=(V2F[indx]-V2F[indx-1]) ;
    double b= V2F[indx]-a*(indx+1) ;
    return (a*V+b)/V/V ;
  }

}

double DeDx_LG::GetProtonStopping() {
  
  double SLow, SHigh ;
  //  double B, G ;
  
  double Ep=E_A ; //MeV/uma

  double EpOld ;
  EpOld=Ep*1000 ;

  Ep*=1000. ; //keV/uma
  if(Ep<25) {
    Ep=25 ;
  }

  SLow = AA[0]*powl(Ep,AA[1]) + AA[2]*powl(Ep,AA[3]) ;
  SHigh = AA[4]/powl(Ep,AA[5])*logl(AA[6]/Ep+AA[7]*Ep) ;
  Se=1./SLow+1./SHigh ;
  Se=1./Se ;
  if(EpOld<25) {
    double Pwr=0.45 ;
    if(Z2<=6) Pwr=0.25 ;
    return Se*powl(EpOld/25.,Pwr)*0.6022/M2*1000. ; // MeV/(mg/cm**2)
  }else{
    return Se*0.6022/M2*1000. ; // MeV/(mg/cm**2)
  }

}

double DeDx_LG::HeStopping() {
  double E_AOld=E_A ;

  double E ;
  if(E_A*1000>1.) E=E_A*1000 ; else E=1. ;
  double B=log(E) ;
  
  double A ;
  A=0.2865 + 0.1266*B - 0.001429*B*B + 0.02402*B*B*B
    - 0.01135*B*B*B*B + 0.001475*B*B*B*B*B ;
  
  double var ;
  if(A>30) A=30 ;
  var = 1.-exp(-A) ;
  
  //Z^3 Effect
  double logE ;
  logE=log(E) ;
  if(logE<0) logE = 0. ;  
  A = (1.+(0.007+0.00005*Z2)*exp(-pow(7.6-logE,2.))) ;
  var *= A*A ;
  
  E_A=E_AOld ;
  double SeRef = GetProtonStopping() ;
  Se = SeRef*var*Z1*Z1 ;
  
  if(E_AOld*1000.>1)
    return Se ;
  else
    return Se/sqrt(E_AOld*1000) ;
  
}
