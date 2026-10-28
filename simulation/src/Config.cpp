#include "Config.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

Config readConfig(const std::string& filename)
{

  static const std::unordered_set<std::string> known = {
    "sequential", "parallel", "skipgefline", "events",
    "elossmodel", "elosstreat", "stepsize", "pressure"};
  
  std::ifstream fin(filename);

  if (!fin) {
    throw std::runtime_error(
			     "Cannot open configuration file: " + filename);
  }
  
  std::unordered_map<std::string,std::string> values;

  int lineNo=0 ;
  std::string line;
  while (std::getline(fin, line)) {
    ++lineNo;
    
    // Supprime les lignes vides
    if (line.empty())
      continue;
    
    // Ignore les commentaires
    if (line[0] == '#')
      continue;
    
    auto pos = line.find('#');
    if (pos != std::string::npos)
      line = line.substr(0, pos);
    
    std::istringstream iss(line);

    std::string key, value;
    if(!(iss >> key)) continue ;
    
    if (!known.count(key))
      throw std::runtime_error("Config: clef inconnue '" + key +
			       "' (ligne " + std::to_string(lineNo) + ")");
    
    if (!(iss >> value))
      throw std::runtime_error("Config: valeur manquante pour '" + key +
			       "' (ligne " + std::to_string(lineNo) + ")");
    
    values[key] = value;
  }

  Config cfg;
  
  if (values.count("sequential"))
        cfg.sequential = (values["sequential"] == "yes");

    if (values.count("parallel"))
        cfg.parallel = std::stoi(values["parallel"]);

    if (values.count("skipgefline"))
        cfg.skipgefline = std::stoll(values["skipgefline"]);

    if (values.count("events"))
        cfg.events = std::stoll(values["events"]);

    if (values.count("evgenerator"))
        cfg.evgenerator = values["evgenerator"];

    if (values.count("elossmodel"))
        cfg.elossmodel = values["elossmodel"];

    if (values.count("elosstreat"))
        cfg.elosstreat = values["elosstreat"];

    if (values.count("stepsize"))
        cfg.stepsize = std::stod(values["stepsize"]);

    if (values.count("pressure"))
        cfg.pressure = std::stod(values["pressure"]);

    return cfg;
}
