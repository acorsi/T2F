#ifndef IDATA_WRITER_H
#define IDATA_WRITER_H

#include "FragmentPair.h"

/**
 * Interface abstraite pour l'écriture de données.
 * Permet de supporter différents formats (HDF5, ROOT, CSV, etc.)
 */
class IDataWriter {
public:
    virtual ~IDataWriter() = default;
    
    /**
     * Écrit une paire de fragments
     */
  virtual void writeFragmentPair(const FragmentPair& pair, const int eventID) = 0;
  virtual void writeDenoisingPair(const FragmentPair& pair) = 0;
    
    /**
     * Finalise l'écriture (flush, fermeture fichier, etc.)
     */
    virtual void finalize() = 0;
};

#endif // IDATA_WRITER_H
