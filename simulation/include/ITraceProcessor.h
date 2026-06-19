#ifndef ITRACE_PROCESSOR_H
#define ITRACE_PROCESSOR_H

#include "ProcessedData.h"
#include <vector>

/**
 * Interface abstraite pour le traitement de traces.
 * Permet de supporter différents pipelines :
 * - Denoising + Features (mode actuel)
 * - AutoEncoder (mode futur)
 * - Hybride
 */
class ITraceProcessor {
public:
    virtual ~ITraceProcessor() = default;
    
    /**
     * Traite une trace brute
     * @param rawTrace Trace brute (avec straggling + bruit)
     * @return Données traitées (features ou latent vector)
     */
  virtual std::pair<ProcessedData, std::vector<double>>  process(const std::vector<double>& rawTrace, double stepSize) = 0;

  virtual ITraceProcessor* clone(int ID) const = 0;
};

#endif // ITRACE_PROCESSOR_H
