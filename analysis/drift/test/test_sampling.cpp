#include "process.h"
#include "event.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>

constexpr double TRACE_LEN_CM = 8.0;
constexpr double BIN_SIZE_CM  = 0.004;
constexpr double DKG = 10.0;
constexpr double DGA = 1.0;

int main()
{
    std::vector<double> steps_cm = {0.004, 0.01, 0.02, 0.03, 0.04, 0.05,
                                     0.1, 0.2, 0.5, 1.0, 2.0};
    // 40,100,200,300,400,500um, puis 1mm,2mm,5mm,1cm,2cm
    std::vector<double> thetas_deg = {0.0, 45.0, 60.0};

    int nBins = static_cast<int>(TRACE_LEN_CM / BIN_SIZE_CM);
    constexpr double TOTAL_ELECTRONS = 3.0e6;
    double dE_per_bin = TOTAL_ELECTRONS / nBins;

    std::mt19937 rng(42);

    std::ofstream summary("/home/claude/project/test/sampling_study.csv");
    summary << "theta_deg,step_um,time_ms,n_hits,total_charge\n";

    for (double theta_deg : thetas_deg)
    {
        double theta = theta_deg * M_PI / 180.0;
        double v[3] = { std::sin(theta), 0.0, -std::cos(theta) };
        double pInit[3] = {0.0, 0.0, DKG + DGA};

        for (double step : steps_cm)
        {
            Processor proc;
            proc.setSamplingStep(step);

            Fragment frag;
            frag.A = 0; frag.Z = 0; frag.n = 0; frag.KE = 0.0;
            frag.raw_trace.assign(nBins, dE_per_bin);

            auto t0 = std::chrono::steady_clock::now();
            proc.processFragment(frag, pInit, v, rng);
            auto t1 = std::chrono::steady_clock::now();

            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double totalCharge = 0.0;
            for (const auto& hit : frag.proc_trace) totalCharge += hit.charge;

            summary << theta_deg << "," << (step * 1e4) << "," << ms << ","
                    << frag.proc_trace.size() << "," << totalCharge << "\n";

            std::cout << "theta=" << theta_deg << " step=" << (step*1e4) << "um  "
                      << "time=" << ms << "ms  n_hits=" << frag.proc_trace.size()
                      << "  total_charge=" << totalCharge << "\n";
        }
    }

    summary.close();
    std::cout << "\nRésumé sauvegardé : sampling_study.csv\n";
    return 0;
}
