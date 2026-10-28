#include "TraceSimulator.h"
#include <memory>
#include <random>

TraceSimulator::TraceSimulator (Material* material, 
                               IEnergyLossCalculator* energyLossCalc,
			       IStragglingAndNoiseCalculator* stragglingCalc,
                               double stepSize)
  : stepSize_(stepSize), rng_(std::random_device{}()), uniform_(0.0,1.0) {
    material_ = std::unique_ptr<Material>(material);
    energyLossCalc_ = std::unique_ptr<IEnergyLossCalculator>(energyLossCalc);
    stragglingAndNoiseCalc_ = std::unique_ptr<IStragglingAndNoiseCalculator>(stragglingCalc) ;


}

std::tuple<std::vector<double>, std::vector<double>, double> TraceSimulator::simulate(const Fragment& fragment) const {
  return propagateFragment(fragment.getA(), fragment.getZ(), fragment.getKineticEnergy());
}

std::tuple<std::vector<double>, std::vector<double>, double> TraceSimulator::propagateFragment(int A, int Z, double initialEnergy) const {
  std::vector<double> clean;
  std::vector<double> noisy;
  
  Particle particle(static_cast<double>(Z), static_cast<double>(A), initialEnergy);
  double currentEnergy = initialEnergy;
  double elementalStep=0.04 ; //en mm
  int NStep = stepSize_/elementalStep ;
  
  energyLossCalc_->initParticle(&particle) ;
  
  double range=0.0;
  
  while (currentEnergy > 0.0) {
    
    //On fait des pas de 40 microns pour calculer la perte d'énergie sur épaisseur totale stepSize_
    //Attention si stepSize_ pas multiple de elementalStep => pas géré.
    double dEdx=0 ;
    double DeltaX=0 ;
    double energyLastStep=currentEnergy;
    double dEdxsum=0.0;    
    //           double MoyenneStrag=0.0 ;


    for(int sub=0;sub<NStep;sub++) {
      particle.SetTKE(currentEnergy);
      double energyPrev = currentEnergy ;
      
      // Appel polymorphique !
      dEdx = energyLossCalc_->computeDeDx(&particle, *material_) *
	elementalStep * 0.1 * 1000;

      
      
      //      std::cout << "TraceSimu : " << currentEnergy << " " << dEdx << std::endl ; 
      /**
       *On ajoute le Straggling.
       **/

      //on modifie l'amplitude du Straggling pour le denoising data set
      //      double factor = uniform_(rng_)*2.;

      double factor=1 ;
      double Strag = stragglingAndNoiseCalc_->computeStraggling(&particle, elementalStep,3*factor) ;
      currentEnergy -= dEdx + Strag ;
      if(currentEnergy<0) { currentEnergy = 0; break;} 
      while(currentEnergy>=energyPrev) { //On empêche que la particule gagne de l'énergie...
	Strag = stragglingAndNoiseCalc_->computeStraggling(&particle, elementalStep,3*factor) ;
	currentEnergy = energyPrev -dEdx -Strag;
      }

      //      std::cout << energyPrev << " " << dEdx << " " << Strag << " " << dEdx + Strag << std::endl ;
      
      range+=elementalStep ;
      DeltaX+=elementalStep ;
      
      if(currentEnergy<0) {
	currentEnergy=0 ;
	dEdxsum+=energyPrev ;
	break ;
      }
      dEdxsum+=dEdx ;
      
    }
    clean.push_back(dEdxsum) ;
    
    if(currentEnergy>0){
      dEdx = energyLastStep-currentEnergy ;
      /**
       *On commente la partie bruit Electronique qui n'est
       *plus faite ici mais dans le code Drift qui transporte
       *la trace d'ionisation vers le plan de pad
       **/
      
      
      /**
       *On ajoute un bruit électronique simple.
       *Ce bruit ne rentre pas dans la conservation d'énergie
       *et peut conduire à dEdx<0 (qu'on force à zéro).
       **/

      //      double factor = uniform_(rng_)*2. ;
      // double factor=1 ;
      // double sigma=0.060/2.35*factor ; //Largeur bruit Gaussien en MeV => environ 2000 électrons de bruit FWHM
      // double electronicNoise = stragglingAndNoiseCalc_->computeElectronicNoise(sigma) ;

      // if(dEdx+electronicNoise<0) {
      // 	noisy.push_back(0);
      // }else{
      // 	noisy.push_back(dEdx+electronicNoise);
      // }
      noisy.push_back(dEdx);
    }else{
      noisy.push_back(energyLastStep);
    }
    
    if (currentEnergy < 0.0) currentEnergy = 0.0;
  }
  
  //std::cout << A << " " << Z << " " << initialEnergy << " " << range << std::endl ;
  
  return {clean, noisy, range};
}
