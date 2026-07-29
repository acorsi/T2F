#include "Config.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

Config readConfig(const std::string& filename)
{
    std::ifstream fin(filename);

    if (!fin) {
        throw std::runtime_error(
                "Cannot open configuration file: " + filename);
    }

    std::unordered_map<std::string,std::string> values;

    std::string line;
    while (std::getline(fin, line)) {

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
        if (iss >> key >> value)
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
