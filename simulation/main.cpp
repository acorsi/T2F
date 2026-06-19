#include "GEFReader.h"
#include "HDF5Writer.h"
#include "HDF5WriteConfig.h"
#include "Material.h"
#include "Config.h"
#include "DeDx_REAL.h"
#include "TraceSimulator.h"
#include "SimulationPipeline.h"
#include "StragglingBohr.h"
#include "IEnergyLossCalculator.h"

#include <iostream>
#include <memory>
#include <string>


std::vector<double> loadFromFile(const std::string& filename) {
    std::vector<double> data;
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "File not found!" << std::endl;
        std::exit(1) ;
    }
    double val;
    while (file >> val) data.push_back(val);
    return data;
}


int main(int argc, char** argv) {

    try {

        Config cfg = readConfig("../config.txt");

        long int SKIP_GEF_LINE = cfg.skipgefline;

        int Nevent = cfg.events;
        std::string DEDX = cfg.elossmodel;
        std::string elosstreat = cfg.elosstreat;

        double stepSize = cfg.stepsize;
        bool parallel = !cfg.sequential;
        int numThreads = cfg.parallel;

        std::cout << "Configuration loaded from config.txt\n";
        std::cout << "Mode: "
            << (parallel ? "PARALLEL" : "SEQUENTIAL")
            << "\n";

        if (parallel)
            std::cout << "Threads: "
                << numThreads
                << "\n";

        std::cout << "Events: "
            << Nevent
            << "\n";

        std::cout << "Energy loss model: "
            << DEDX
            << "\n";

        std::cout << "Pressure: "
            << cfg.pressure
            << " mbar\n";

        std::filesystem::path base="/Users/acorsi/codes/T2F_v2"; 

        auto  gefFile = base / "simulation/inputs/gef/MyYields";
        auto  hdf5File = base / "simulation_output" / ("T2F_"+DEDX+"_"+elosstreat+".h5");


        // 1. Setup Material (P10)
        std::unique_ptr<Material> material = std::make_unique<Material> ();

        // 2. Setup Physics
        //    

        std::unique_ptr<IEnergyLossCalculator> energyLossCalc ;
        std::filesystem::path ELossPath;

        if (elosstreat=="native") {
            ELossPath = base / "simulation/inputs/dedx" / DEDX;   // from code
        }
        else if (elosstreat=="svd"||elosstreat=="corr") {
            ELossPath = base / "dedx_from_data" / DEDX;           // from tracks
        }
        else {
            throw std::runtime_error(
            "Unknown DEDX model: " + DEDX);
        }

        energyLossCalc = std::make_unique<DeDx_REAL>(ELossPath,*material,cfg.elosstreat);
        //1 ligne commentaire puis TKE/A dEdx_tot(MeV*cm2/mg)

        std::unique_ptr<StragglingBohr> stragglingBohr = std::make_unique<StragglingBohr>(*material) ;
        std::unique_ptr<TraceSimulator> traceSimulator = std::make_unique<TraceSimulator>(material.release(),
                energyLossCalc.release(),
                stragglingBohr.release(),
                stepSize);  // 2mm step


        // 3. Setup Processing

        //std::unique_ptr<IDenoiser> denoiser = std::make_unique<IdentityDenoiser>() ;
        //    std::unique_ptr<IDenoiser> denoiser = std::make_unique<UNet1DDenoiser>(model_path, 150, 1024, false);
        //std::unique_ptr<FeatureExtractor> featureExtractor = std::make_unique<FeatureExtractor>();
        //std::unique_ptr<DenoisingPipeline> processor = std::make_unique<DenoisingPipeline>(denoiser.release(), featureExtractor.release());

        // 4. Setup I/O
        std::unique_ptr<GEFReader> reader = std::make_unique<GEFReader>(gefFile,Nevent,SKIP_GEF_LINE);
        HDF5WriteConfig writeConfig ;  // Configuration par défaut
        std::unique_ptr<HDF5Writer> writer =  std::make_unique<HDF5Writer>(hdf5File);
        writer->addMetadata("material", "P10");
        writer->addMetadata("step_size", stepSize);

        // 5. Create Pipeline
        std::unique_ptr<SimulationPipeline> pipeline = std::make_unique<SimulationPipeline>(reader.release(), writer.release(), traceSimulator.release(),cfg.elosstreat/*, processor.release()*/);

        // 6. Run
        pipeline->setParallel(parallel, numThreads);
        //      pipeline->setParallel(true, 12);  // 12 threads

        std::cout << "Starting simulation..." << std::endl;
        pipeline->run(ELossPath);

        std::cout << std::endl << "Simulation completed! " << std::endl
            << "Output in : " << hdf5File << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
