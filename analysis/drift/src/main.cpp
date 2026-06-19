#include "event.h"
#include "read_h5.h"
#include "process.h"
#include "write_h5.h"
#include <iostream>
#include <string>
#include <chrono>
#include <limits>

static void printFragment(const std::string& tag, const Fragment& f)
{
    std::cout << "  " << tag
              << "  Z=" << f.Z << " A=" << f.A
              << " n=" << f.n << " KE=" << f.KE
              << "  raw[0..4]:";
    for (int i = 0; i < 5 && i < (int)f.raw_trace.size(); ++i)
        std::cout << " " << f.raw_trace[i];
    std::cout << "  n_hits=" << f.proc_trace.size() << " :";
    for (int i = 0; i < 5 && i < (int)f.proc_trace.size(); ++i)
        std::cout << " (" << f.proc_trace[i].row << ","
                   << f.proc_trace[i].col << ","
                   << f.proc_trace[i].charge << ")";
    std::cout << "\n";
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " input.h5 output.h5 [maxEvents]\n";
        return 1;
    }

    hsize_t maxEvents = (argc >= 4) ? std::stoull(argv[3])
                                     : std::numeric_limits<hsize_t>::max();

    ReaderH5  reader(argv[1]);
    Processor processor;

    hsize_t nEvents = reader.nEvents();
    if (maxEvents < nEvents) nEvents = maxEvents;
    WriterH5  writer(argv[2], argv[1], nEvents);

    hsize_t nProcessed = 0;
    bool firstChunk = true;
    double t_read = 0, t_process = 0, t_write = 0;

    for (hsize_t offset = 0; offset < nEvents; offset += CHUNK_SIZE)
    {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<Event> chunk = reader.readChunk(offset, CHUNK_SIZE);
        auto t1 = std::chrono::steady_clock::now();
        if (chunk.empty()) break;

        processor.processChunk(chunk);
        auto t2 = std::chrono::steady_clock::now();

        writer.writeChunk(chunk, offset);
        auto t3 = std::chrono::steady_clock::now();

        t_read    += std::chrono::duration<double>(t1 - t0).count();
        t_process += std::chrono::duration<double>(t2 - t1).count();
        t_write   += std::chrono::duration<double>(t3 - t2).count();

        if (firstChunk) {
            std::cout << "\n--- Verification (3 premiers events) ---\n";
            for (int i = 0; i < 3 && i < (int)chunk.size(); ++i) {
                std::cout << "Event " << i << ":\n";
                printFragment("L", chunk[i].L);
                printFragment("H", chunk[i].H);
            }
            std::cout << "----------------------------------------\n\n";
            firstChunk = false;
        }

        nProcessed += chunk.size();
        std::cout << "Processed " << nProcessed << " / " << nEvents
                  << " events\r" << std::flush;
    }

    std::cout << "\nDone. Total: " << nProcessed << " events\n";
    std::cout << "Temps lecture  : " << t_read    << " s\n";
    std::cout << "Temps process  : " << t_process << " s\n";
    std::cout << "Temps écriture : " << t_write   << " s\n";
    return 0;
}
