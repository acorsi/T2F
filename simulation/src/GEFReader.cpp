#include "GEFReader.h"

GEFReader::GEFReader(const std::string& filepath, const long int maxcount, const long int skip)
  : filepath_(filepath), maxCount_(maxcount), currentEventID_(0), endOfFile_(false) {
    // TODO: Ouvrir le fichier et initialiser
  fileStream_.open(filepath_,std::ios::in) ;
  if(!fileStream_) {
    std::cout << "File " << filepath_ << " not found. Exit." << std::endl ;
    exit(1) ;
  }else{
    std::cout << std::endl << "GEFReader : Opening File " << filepath_ << " : OK." << std::endl ;
    std::cout << "Reading " << maxCount_ << " events or up to EOF." << std::endl ;
  }

  // ===== SKIP =====
  if (skip > 0) {
    std::cout << "Skipping " << skip << " lines..." << std::flush;
    std::string line;
    for (long int i = 0; i < skip && std::getline(fileStream_, line); i++) {}
    std::cout << " Done." << std::endl;
  }

  // ===== PRÉCHARGER LE FICHIER EN CACHE =====
  //On force la mise en cache du fichier GEF
  //==> Exécution totale environ 2x plus rapide
  std::cout << "Preloading GEF file into cache..." << std::flush;
  auto start = std::chrono::high_resolution_clock::now();
  
  // Sauvegarder la position actuelle
  auto currentPos = fileStream_.tellg();

  std::string line;
  size_t bytesRead = 0;
  
  for (int i = 0; i < maxCount_ && std::getline(fileStream_, line); i++) {
    bytesRead += line.size() + 1;  // +1 pour le '\n'
  }
  
  // // Lire tout le fichier (force mise en cache)
  // fileStream_.seekg(0, std::ios::end);
  // size_t fileSize = fileStream_.tellg();
  // fileStream_.seekg(0, std::ios::beg);
  
  // std::vector<char> buffer(fileSize);
  // fileStream_.read(buffer.data(), fileSize);
  
  // Restaurer la position
  fileStream_.clear();
  fileStream_.seekg(currentPos);
  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
  std::cout << " Done (" << bytesRead / (1024 * 1024) << " MB in " 
	    << ms << " ms)." << std::endl << std::endl ;
  // ===========================================
    
}

GEFReader::~GEFReader() {
  if (fileStream_.is_open()) {
    fileStream_.close();
  }
}

FragmentPair* GEFReader::readNextPair() {
  // TODO: Implémenter la lecture

  //Test EOF ou maxCount_ atteint
  if(!hasNext()) {
    return nullptr;
  }

  std::string line;
  if (!std::getline(fileStream_, line)) {
    endOfFile_ = true ;
    return nullptr;
  }

  std::istringstream iss(line);
  int A_light, Z_light, n_light;
  int A_heavy, Z_heavy, n_heavy;
  int dum;
  float KE_light, KE_heavy;
  
  if (!(iss >> Z_light >> A_light >> KE_light >> dum 
	>> Z_heavy >> A_heavy >> KE_heavy >> dum >> n_light >> n_heavy)) {
    throw std::runtime_error("Invalid GEF format");
  }

  // ===== CRÉER LES VARIABLES FRAGMENT =====
  Fragment lightFragment(A_light, Z_light, KE_light, n_light);
  Fragment heavyFragment(A_heavy, Z_heavy, KE_heavy, n_heavy);
  
  // ===== CRÉER LA PAIRE EN TRANSFÉRANT =====
  FragmentPair* pair = new FragmentPair(
					std::move(lightFragment),   // Transfert (move)
					std::move(heavyFragment),   // Transfert (move)
					currentEventID_++
				       );
    
  return pair;
  
}

bool GEFReader::hasNext() const {
  if(currentEventID_>=maxCount_) return false ;
  return !endOfFile_;
}

