#include "../inc/Material.h"
#include "../inc/DeDx_LG.h"

#include <iostream>
#include <fstream> 
#include <memory>
#include <string>


std::vector<double> loadFromFile(const std::string& filename) {
    std::vector<double> data;
    std::ifstream file(filename);
    if (!file) {
      std::cerr << "File not found!" << std::endl;
      std::exit(1) ;
    }
    double val;
    while (file >> val) data.push_back(val);
    return data;
}

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n"
              << "Options:\n"
              << "  --sequential     Run in sequential mode (default)\n"
              << "  --parallel N     Run in parallel mode with N threads\n"
              << "  --events N       Number of events to process (default: 1000)\n"
              << "  --help           Show this help\n";
}

int main(int argc, char** argv) {
  
  try {
    // Configuration
    int Nevent =1e6 ;    
    std::string File ;
    std::ofstream out ;
    
    // 1. Setup Material (P10)
    Material *material = new Material();
    
    DeDx_LG *ELossCalc = new DeDx_LG(*material);

    int A;
    float KE, DKE, Se, Sn ;

    Particle *FF = new Particle(40,100,100) ;
    
    for(int Z=20; Z<=72;Z++) {
      A=(int)(2.57*Z) ;
      //      for(int A=A_mean-6;A<=A_mean+6;A+=12) {	    	

	File = "../tables/dedx_" + std::to_string(Z) + ".lg" ; 
	out.open(File,std::ios::out) ;

	DKE=0.001 ;
	for(KE=0.001;KE<=150;KE+=DKE) {
	  FF->Set(Z,A,KE) ;
	  ELossCalc->SetParticle(FF) ;
	  
	  auto[Se,Sn] = ELossCalc->Get(FF,*material) ;
	  out << KE/A << " " << Se << " " << Sn << std::endl ;
          
	  if(KE>50) DKE=5 ;
	  else if(KE>20) DKE=5 ;
	  else if(KE>5) DKE=2 ;
	  else if(KE>0.5) DKE=0.5 ;
	  else if(KE>0.1) DKE=0.1 ;
	  else if(KE>0.01) DKE=0.01;
	}
	out.close() ;
	
	//      }
    }
    
    return 0;
    
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
