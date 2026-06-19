#include "process.h"
#include "event.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>

constexpr double TRACE_LEN_CM = 8.0;
constexpr double BIN_SIZE_CM  = 0.004;  // 40 microns
constexpr double DKG = 10.0;
constexpr double DGA = 1.0;

struct TestCase {
    std::string name;
    double theta_deg;
    double phi_deg;
};

int main()
{
    std::vector<TestCase> cases = {
        {"theta0",        0.0,  0.0},
        {"theta45_phi0",  45.0, 0.0},
        {"theta45_phi90", 45.0, 90.0},
        {"theta60_phi0",  60.0, 0.0},
        {"theta60_phi45", 60.0, 45.0},
        {"theta60_phi90", 60.0, 90.0},
    };

    Processor proc;
    std::mt19937 rng(42);  // seed fixe pour reproductibilité (non utilisé ici car pas de tirage dans processFragment actuellement)

    int nBins = static_cast<int>(TRACE_LEN_CM / BIN_SIZE_CM);

    for (const auto& tc : cases)
    {
        double theta = tc.theta_deg * M_PI / 180.0;
        double phi   = tc.phi_deg   * M_PI / 180.0;

        double v[3] = {
            std::sin(theta) * std::cos(phi),
            std::sin(theta) * std::sin(phi),
            -std::cos(theta)
        };
        double pInit[3] = {0.0, 0.0, DKG + DGA};

        Fragment frag;
        frag.A = 0; frag.Z = 0; frag.n = 0; frag.KE = 0.0;

        constexpr double TOTAL_ELECTRONS = 3.0e6;
        double dE_per_bin = TOTAL_ELECTRONS / nBins;
        frag.raw_trace.assign(nBins, dE_per_bin);  // dépôt constant, échelle réaliste

        proc.processFragment(frag, pInit, v, rng);

        std::string filename = "out_" + tc.name + ".csv";
        std::ofstream out(filename);
        out << "row,col,charge\n";
        for (const auto& hit : frag.proc_trace)
            out << hit.row << "," << hit.col << "," << hit.charge << "\n";
        out.close();

        std::cout << tc.name << " : " << frag.proc_trace.size()
                  << " pads touchés -> " << filename << "\n";
    }

    return 0;
}
