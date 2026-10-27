#pragma once

#include <string>

struct Config {
    bool sequential = false;
    int parallel = 1;
    long long events = 2000000;
    long long skipgefline = 0;

    std::string elossmodel = "G4";
    std::string elosstreat = "corr";
    std::string evgenerator = "corr";

    double stepsize = 2;
    double pressure = 300.0;

    
};

Config readConfig(const std::string& filename);
