#ifndef GEF_READER_H
#define GEF_READER_H

#include "IDataReader.h"
#include <fstream>
#include <sstream> 
#include <string>

/**
 * Lecteur de fichiers GEF (format texte).
 * Lit des paires de fragments de fission avec leurs propriétés (A, Z, KE, n).
 */
class GEFReader : public IDataReader {
private:
  std::string filepath_;
  std::ifstream fileStream_;
  int currentEventID_;
  bool endOfFile_;
  long int maxCount_ ;
  
public:
  /**
   * Constructeur
   * @param filepath Chemin vers le fichier GEF
   */
  explicit GEFReader(const std::string& filepath, const long int maxcount, const long int skip=0);
  
  /**
   * Destructeur (ferme le fichier)
   */
  ~GEFReader() override;
  
  // Implémentation de IDataReader
  FragmentPair* readNextPair() override;
  bool hasNext() const override;

};

#endif // GEF_READER_H
