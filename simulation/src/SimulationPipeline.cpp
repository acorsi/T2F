#include "SimulationPipeline.h"
#include <cstddef>
#include <iostream>
#include "StragglingBohr.h"
#include "Material.h"
//#include "DeDx_LG.h"
#include "DeDx_REAL.h"
#include "IEnergyLossCalculator.h"
//#include "processing/ITraceProcessor.h"
#include <omp.h>

SimulationPipeline::SimulationPipeline(IDataReader* reader,
                                       IDataWriter* writer,
                                       TraceSimulator* traceSimulator,
                                       //ITraceProcessor* processor
                                       const std::string& elosstreat
                                       )
  : enableParallel_(false), numThreads_(-1), elosstreat_(elosstreat){
  reader_ = std::unique_ptr<IDataReader>(reader);
  writer_ = std::unique_ptr<IDataWriter>(writer);
  traceSimulator_ = std::unique_ptr<TraceSimulator>(traceSimulator);
  //processor_ = std::unique_ptr<ITraceProcessor>(processor);
  
}

void SimulationPipeline::run(const std::filesystem::path &ELossPath) {
  std::cout << std::endl << "Pipeline running ..." << std::endl;
  int LoopCount = 0;
  const size_t BATCH_SIZE = 10000;
  std::vector<FragmentPair*> batch;
  batch.reserve(BATCH_SIZE);
  
  //    auto globalStart = std::chrono::high_resolution_clock::now();
  
  // ===== TRAITEMENT PARALLÈLE AVEC SIMULATEUR et PROCESSEUR LOCAUX =====
  std::vector<std::unique_ptr<TraceSimulator>> localSimulators;
  //std::vector<std::unique_ptr<ITraceProcessor>> localProcessors;

  if (enableParallel_) {
    localSimulators.reserve(numThreads_);
    //if(processor_) localProcessors.reserve(numThreads_);
    
    for (int i = 0; i < numThreads_; i++) {
      IEnergyLossCalculator *dedx ;
      std::cout << "Loading Simulators for thread " << i ; 
      Material* mat = new Material(*traceSimulator_->getMaterial());
     /* if(DEDX=="LG") {
	dedx = new DeDx_LG(*mat);
      }else if(DEDX=="SRIM"){
	dedx = new DeDx_SRIM(ELossPath,*mat) ;
      }else if(DEDX=="LG_Corr"){
	dedx = new DeDx_REAL(ELossPath,*mat);
      }else if(DEDX=="SRIM_Corr"){
	dedx = new DeDx_REAL(ELossPath,*mat);
      }else if(DEDX=="Geant4_Corr"){
	dedx = new DeDx_REAL(ELossPath,*mat);
      }*/
	  dedx = new DeDx_REAL(ELossPath,*mat,elosstreat_) ;
      StragglingBohr *stragglingbohr = new StragglingBohr(*mat) ;
      localSimulators.emplace_back(
				   new TraceSimulator(mat, dedx, stragglingbohr, traceSimulator_->getStepSize())
				  );
      
      //if(processor_) localProcessors.emplace_back(processor_->clone(i));
      
      std::cout << " : Done." << std::endl ;
    }
  }

  std::cout <<std::endl ; 
  
  while (reader_->hasNext()) {
    batch.clear();
    for (size_t i = 0; i < BATCH_SIZE && reader_->hasNext(); i++) {
      FragmentPair* pair = reader_->readNextPair();
      if (pair) batch.push_back(pair);
    }
    
    if (batch.empty()) break;
    
    if(enableParallel_) {
      
#pragma omp parallel num_threads(numThreads_)
      {
	int tid = omp_get_thread_num();
	TraceSimulator* mySimulator = localSimulators[tid].get();
	//ITraceProcessor* myProcessor = processor_ ? localProcessors[tid].get() : nullptr;
        
#pragma omp for schedule(dynamic)
	for (size_t i = 0; i < batch.size(); i++) {
	  processFragmentPairWithSimulator(batch[i], mySimulator/*, myProcessor*/);
	}
      }
    } else {
      for (auto* pair : batch) {
	processFragmentPairWithSimulator(pair, traceSimulator_.get()/*, processor_.get()*/);
      }
    }
    
    // Écrire le batch
    for (auto* pair : batch) {
      writer_->writeFragmentPair(*pair, ++LoopCount);
      //      writer_->writeDenoisingPair(*pair);
      delete pair;
    }
    
  }
}

void SimulationPipeline::setParallel(bool enable, int numThreads) {
    enableParallel_ = enable;
    numThreads_ = numThreads;
}

void SimulationPipeline::processFragmentPairWithSimulator(FragmentPair* pair, TraceSimulator* simulator) {
  for (int i = 0; i < 2; i++) {
    Fragment& frag = (i == 0) ? pair->getLightFragment() 
      : pair->getHeavyFragment();
    
    // Simulation trace avec le simulateur fourni
    auto [clean,noisy,range] = simulator->simulate(frag);
    frag.setBetheTrace(clean);
    frag.setRawTrace(noisy);
    //frag.setTrueRange(range) ;

    // std::cout << "Fragment " << i << std::endl ;
    // for(size_t nf=0;nf<noisy.size();nf++) {
    //   std::cout << nf << " " << noisy.at(nf) << std::endl ;
    // }
    
    double stepSize = simulator->getStepSize();
    /*
    if (processor) {
      //      std::cout << "We enter in process mode :" << std::endl ;
      auto [processed, denoised] = processor->process(noisy,stepSize);
      frag.setDenoisedTrace(denoised) ;
      frag.setProcessedData(processed);

      
    }*/
  }
}

// void SimulationPipeline::processFragment(Fragment* fragment) {
//   std::vector<double>trace = traceSimulator_->simulate() ;
  
// }
