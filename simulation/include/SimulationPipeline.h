#ifndef SIMULATION_PIPELINE_H
#define SIMULATION_PIPELINE_H

#include "IDataReader.h"
#include "IDataWriter.h"
#include "TraceSimulator.h"
//#include "ITraceProcessor.h"
#include "FragmentPair.h"
#include <vector>
#include <memory>
#include <functional>

/**
 * Pipeline de simulation complet.
 * 
 * Workflow :
 * 1. Lit les paires de fragments (GEFReader)
 * 2. Simule les traces (TraceSimulator)
 * 3. Traite les traces (ITraceProcessor : denoising + features)
 * 4. Écrit les résultats (HDF5Writer)
 * 
 * Supporte la parallélisation OpenMP.
 */
class SimulationPipeline {
private:
  std::unique_ptr<IDataReader> reader_;
  std::unique_ptr<IDataWriter> writer_;
  std::unique_ptr<TraceSimulator> traceSimulator_;
  //std::unique_ptr<ITraceProcessor> processor_;
  
  bool enableParallel_;
  int numThreads_;
  std::string elosstreat_;
  /**
   * Traite une paire de fragments
   */
  void processFragmentPairWithSimulator(FragmentPair* pair, TraceSimulator* simulator);
  
public:
  /**
   * Constructeur
   * @param reader Lecteur de données (ex: GEFReader)
   * @param writer Écrivain de données (ex: HDF5Writer)
   * @param traceSimulator Simulateur de traces
   * @param processor Processeur de traces (ex: DenoisingPipeline)
   */
  SimulationPipeline(IDataReader* reader,
		     IDataWriter* writer,
		     TraceSimulator* traceSimulator,
		     const std::string& elosstreat
             /*ITraceProcessor* processor*/);
  
  /**
   * Exécute le pipeline complet
   */
  void run(const std::filesystem::path &ELossPath);
  
  /**
   * Active/désactive la parallélisation
   */
  void setParallel(bool enable, int numThreads = -1);

  /**
   * Définit la factory pour créer des processors locaux (parallélisme)
   * @param factory Fonction qui retourne un nouveau ITraceProcessor*
   */
  
};

#endif // SIMULATION_PIPELINE_H
