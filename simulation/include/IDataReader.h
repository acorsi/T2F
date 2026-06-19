#ifndef IDATA_READER_H
#define IDATA_READER_H

#include "FragmentPair.h"

/**
 * Interface abstraite pour la lecture de données de fragments.
 * Permet de supporter différents formats (GEF, FREYA, etc.)
 */
class IDataReader {
public:
    virtual ~IDataReader() = default;
    
    /**
     * Lit la prochaine paire de fragments
     * @return Pointeur vers FragmentPair (nullptr si fin de fichier)
     */
    virtual FragmentPair* readNextPair() = 0;
    
    /**
     * Vérifie s'il reste des données à lire
     */
    virtual bool hasNext() const = 0;
    
};

#endif // IDATA_READER_H
