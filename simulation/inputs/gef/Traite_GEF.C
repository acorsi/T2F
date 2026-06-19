#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#include <stdlib.h>

using namespace std ;

#define NMAX 50000000

int main() {

  ifstream in ;
  //  in.open("252Cf_3.lmd") ;
  in.open("252Cf_LG.lmd") ;
  string line ;

  for(int i=0;i<62;i++) {
    getline(in, line) ;
    //    cout << line << endl ;
  }

  int A1, Z1, A2, Z2, a1, a2, n1, n2 ;
  float dum, KE1, KE2, TKE, ke1, ke2, tke, DE ;

  int CPT=0 ;
  float valr ;
  while(1) {
    in >> dum >> dum >> dum >> dum ;
    if(in.eof()) break ;
    in >> Z1 >> Z2 >> a1 >> a2 >> A1 >> A2 ;
    in >> dum >> dum >> dum >> dum >> KE1 ;
    in >> dum >> dum ;
    in >> KE2 ;
    in >> dum >> dum >> dum >> dum >> dum ;
    in >> dum >> dum >> dum >> dum >> dum ;
    in >> n1 >> n2 >> tke >> TKE >> dum ;

    /* KE1 = (ke1*A1)/a1 ; */
    /* KE2 = (ke2*A2)/a2 ; */
    /* DE = TKE-(KE1+KE2) ; */
    
    /* valr = (float)drand48() ; */
    /* if(n1!=0) */
    /*   KE1+=DE*valr ; */
    
    /* if(n2!=0) */
    /*   KE2 = TKE-KE1 ; */
    /* else */
    /*   KE1 = TKE-KE2 ; */
    
    //    cout << KE1 << " " << KE2 << " " << TKE << endl ;
    
    if(isnan(KE1)) { cout << "On rencontre un NaN." << endl ; exit(1) ; }
    if(isnan(KE2)) { cout << "On rencontre un NaN." << endl ; exit(1) ; }

    
    cout << Z1 << " " << A1 << " " << KE1 << " 0 "
	 << Z2 << " " << A2 << " " << KE2 << " 0 " << n1 << " " << n2 << endl ;
    
    if(n1!=a1-A1) cout << n1 << " " << a1 << " " << A1 << endl ;
    if(n2!=a2-A2) cout << n2 << " " << a2 << " " << A2 << endl ;
    if(a1+a2!=252) cout << a1 << " " << a2 << endl ;
    CPT++;
    if(CPT>NMAX) break ;
  }
  in.close() ;
  
  return 1 ;  

}
