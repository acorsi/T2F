#include "h5reader.hpp"
#include "h5writer.hpp"
#include "detector.hpp"
#include "processor.hpp"
#include "pad_geometry.hpp"

#include <iostream>
#include <fstream>
#include <numeric>
#include <chrono>
#include <omp.h>
#include <ctime>

std::vector<double> merge_trace(const std::vector<double>& raw, int n_merge) {
    if (n_merge <= 1) return raw;
    size_t N_out = (raw.size() + n_merge - 1) / n_merge;
    std::vector<double> merged(N_out);
    for (size_t i = 0; i < N_out; ++i) {
        size_t start = i * n_merge;
        size_t end   = std::min(start + (size_t)n_merge, raw.size());
        merged[i] = std::accumulate(raw.begin() + start, raw.begin() + end, 0.0);
    }
    return merged;
}

int main(int argc, char* argv[]) {
    // Usage : process [N_event]

    const std::string in_path  = "../../../../T2F_New/simulation_output/T2F_G4_40mu.h5";
    const std::string out_path = "../drift_output/drift_G4_40mu_merged.h5";

    // --- Lecture metadata ---
    H5Reader reader(in_path);
    const hsize_t N_total = reader.size("/H/KE");
    const double step_size_in = reader.read_attribute<double>("/metadata", "step_size") / 10.0;

    const hsize_t N = (argc > 1) ? std::min((hsize_t)std::stoul(argv[1]), N_total) : N_total;

    const double step_size_target = 0.05;
    const int    n_merge          = std::max(1, (int)(step_size_target / step_size_in));
    const double step_size_eff    = n_merge * step_size_in;

    std::cout << "step_size_in      = " << step_size_in  * 10 << " mm\n";
    std::cout << "step_size_target  = " << step_size_target * 10 << " mm\n";
    std::cout << "n_merge           = " << n_merge             << "\n";
    std::cout << "step_size_eff     = " << step_size_eff * 10  << " mm\n";
    std::cout << "Events à traiter  = " << N                   << "\n";

    // --- Configuration détecteur ---
    DetectorConfig cfg ;
    cfg.step_size = step_size_eff;

    // --- Plan de pads ---
    PadPlaneParams pParam ;
    
    PadPlane plane = PadPlane(pParam);

    std::ofstream pad_file("my_pad.txt");
    pad_file << "# id x y r_in r_out\n";
    for (const auto& p : plane.pads())
        pad_file << p.id << " " << p.x << " " << p.y
                 << " " << p.r_in << " " << p.r_out << " " << p.sx << " " << p.sy << "\n";
    pad_file.close();
    std::cout << "Géométrie pads écrite dans my_pad.txt ("
              << plane.pads().size() << " pads)\n";

    // --- Préparation fichier de sortie ---
    H5Writer writer(out_path, N);

    std::time_t t = std::time(nullptr);
    char tbuf[32];
    std::strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));

    writer.write_attribute_string("/metadata", "created",    tbuf);
    writer.write_attribute_string("/metadata", "input_file", in_path);

    writer.write_attribute<double>("/metadata", "step_size_in",  step_size_in);
    writer.write_attribute<double>("/metadata", "step_size_eff", step_size_eff);
    writer.write_attribute<int>   ("/metadata", "n_merge",       n_merge);
    writer.write_attribute<int>   ("/metadata", "n_events",      (int)N);

    writer.write_attribute<double>("/metadata", "dkg",           cfg.dkg);
    writer.write_attribute<double>("/metadata", "dga",           cfg.dga);
    writer.write_attribute<double>("/metadata", "vertex_r_max",  cfg.vertex_r_max);
    writer.write_attribute<double>("/metadata", "W",             cfg.W);
    writer.write_attribute<double>("/metadata", "D_T",           cfg.D_T);
    writer.write_attribute<double>("/metadata", "D_L",           cfg.D_L);
    writer.write_attribute<double>("/metadata", "v_e",           cfg.v_e);
    writer.write_attribute<double>("/metadata", "delta_t",       cfg.delta_t);

    writer.write_attribute<double>("/metadata", "z_anode",     (cfg.dkg + cfg.dga));
    writer.write_attribute<double>("/metadata", "pad_size",    pParam.pad_size);
    writer.write_attribute<double>("/metadata", "inter_pad",   pParam.inter_pad);
    writer.write_attribute<double>("/metadata", "r_active",    pParam.r_active);
    writer.write_attribute<double>("/metadata", "r_quinconce", pParam.r_quinconce);
    writer.write_attribute<double>("/metadata", "r_transition",pParam.r_transition);
    writer.write_attribute<int>   ("/metadata", "n_rings",     pParam.n_rings);
    writer.write_attribute<double>("/metadata", "y_offset",    pParam.y_offset);

    writer.write_attribute<int>("/metadata", "n_pads", (int)plane.pads().size());

    for (const std::string& frag : {"/H", "/L"}) {
        writer.add_dataset<double>  (frag + "/theta");
        writer.add_dataset<double>  (frag + "/phi");
        writer.add_dataset<double>  (frag + "/vx");
        writer.add_dataset<double>  (frag + "/vy");
        writer.add_dataset<double>  (frag + "/vz");
        writer.add_vlen_dataset     (frag + "/x");
        writer.add_vlen_dataset     (frag + "/y");
        writer.add_vlen_dataset     (frag + "/z");
        writer.add_vlen_dataset     (frag + "/dE");
        writer.add_vlen_dataset     (frag + "/px");
        writer.add_vlen_dataset     (frag + "/py");
        writer.add_vlen_dataset     (frag + "/pq");
        writer.add_vlen_dataset     (frag + "/pt");
        writer.add_vlen_dataset     (frag + "/pqt");
        writer.add_vlen_dataset     (frag + "/pnt");
        writer.add_vlen_dataset     (frag + "/pid");
    }

    // --- Boucle par chunks ---
    const hsize_t chunk_size = 20000;
    const int     nthreads   = 14;

    std::cout << "\n\033[1;32m" << nthreads << " running threads...\033[0m\n\n";

    const uint64_t base_seed = 42;
    std::vector<Processor> procs;
    procs.reserve(nthreads);
    for (int t = 0; t < nthreads; ++t)
        procs.emplace_back(cfg, base_seed + t);

    const int n_bins = cfg.n_bins() ;
    std::vector<std::vector<std::vector<double>>> qt_bufs(
        nthreads,
        std::vector<std::vector<double>>(plane.pads().size(),
                                         std::vector<double>(n_bins, 0.0)));

    for (hsize_t start = 0; start < N; start += chunk_size) {
        hsize_t count = std::min(chunk_size, N - start);

        auto t0 = std::chrono::steady_clock::now();

	auto H_KE = reader.read_chunk<double>("/H/KE", start, count) ;
        auto H_trace = reader.read_vlen_chunk("/H/raw_trace", start, count);
        auto L_trace = reader.read_vlen_chunk("/L/raw_trace", start, count);

        auto t1 = std::chrono::steady_clock::now();

        std::vector<double> H_theta(count), H_phi(count);
        std::vector<double> H_vx(count), H_vy(count), H_vz(count);
        std::vector<double> L_theta(count), L_phi(count);
        std::vector<double> L_vx(count), L_vy(count), L_vz(count);

        std::vector<std::vector<double>> H_x(count), H_y(count), H_z(count), H_dE(count);
        std::vector<std::vector<double>> H_px(count), H_py(count), H_pq(count);
        std::vector<std::vector<double>> H_pt(count), H_pqt(count), H_pnt(count), H_pid(count);
        std::vector<std::vector<double>> L_x(count), L_y(count), L_z(count), L_dE(count);
        std::vector<std::vector<double>> L_px(count), L_py(count), L_pq(count);
        std::vector<std::vector<double>> L_pt(count), L_pqt(count), L_pnt(count), L_pid(count);

        std::vector<std::vector<PadCharge>> H_pads(count), L_pads(count);

#pragma omp parallel for num_threads(nthreads)
        for (hsize_t i = 0; i < count; ++i) {
            int tid = omp_get_thread_num();

            EventInput in;
            in.H = { merge_trace(H_trace[i], n_merge), 0, 0, 0 };
            in.L = { merge_trace(L_trace[i], n_merge), 0, 0, 0 };

            EventOutput out;
            procs[tid].process(in, out);

            H_pads[i] = plane.project(out.H.x, out.H.y, out.H.z,
                                      out.H.n_electrons, out.H.sigma_T, out.H.sigma_L,
                                      cfg, qt_bufs[tid]);
            L_pads[i] = plane.project(out.L.x, out.L.y, out.L.z,
                                      out.L.n_electrons, out.L.sigma_T, out.L.sigma_L,
                                      cfg, qt_bufs[tid]);

            H_theta[i] = out.H.dir.theta; H_phi[i] = out.H.dir.phi;
            H_vx[i] = out.H.vx; H_vy[i] = out.H.vy; H_vz[i] = out.H.vz;
            H_x[i] = out.H.x; H_y[i] = out.H.y; H_z[i] = out.H.z; H_dE[i] = out.H.dE;
            
            double Sum = 0.0;
            for (const auto& pc : H_pads[i]) {
                H_pnt[i].push_back((double)pc.t.size());
                for (double t : pc.t)  H_pt[i].push_back(t);
                for (double qt : pc.qt) H_pqt[i].push_back(qt);
                H_px[i].push_back(pc.x);
                H_py[i].push_back(pc.y);
                H_pq[i].push_back(pc.q);
                H_pid[i].push_back((double)pc.pad_id);

		//		Sum+=pc.q*cfg.W ;
	    }

	    //            std::cout << H_KE[i] << " " << Sum << std::endl ;

	    
            L_theta[i] = out.L.dir.theta; L_phi[i] = out.L.dir.phi;
            L_vx[i] = out.L.vx; L_vy[i] = out.L.vy; L_vz[i] = out.L.vz;
            L_x[i] = out.L.x; L_y[i] = out.L.y; L_z[i] = out.L.z; L_dE[i] = out.L.dE;

            for (const auto& pc : L_pads[i]) {
                L_pnt[i].push_back((double)pc.t.size());
                for (double t : pc.t)  L_pt[i].push_back(t);
                for (double qt : pc.qt) L_pqt[i].push_back(qt);
                L_px[i].push_back(pc.x);
                L_py[i].push_back(pc.y);
                L_pq[i].push_back(pc.q);
                L_pid[i].push_back((double)pc.pad_id);
            }
        }

        auto t2 = std::chrono::steady_clock::now();

        writer.write_chunk<double> ("/H/theta", H_theta, start);
        writer.write_chunk<double> ("/H/phi",   H_phi,   start);
        writer.write_chunk<double> ("/H/vx",    H_vx,    start);
        writer.write_chunk<double> ("/H/vy",    H_vy,    start);
        writer.write_chunk<double> ("/H/vz",    H_vz,    start);
        writer.write_vlen_chunk    ("/H/x",     H_x,     start);
        writer.write_vlen_chunk    ("/H/y",     H_y,     start);
        writer.write_vlen_chunk    ("/H/z",     H_z,     start);
        writer.write_vlen_chunk    ("/H/dE",    H_dE,    start);
        writer.write_vlen_chunk    ("/H/px",    H_px,    start);
        writer.write_vlen_chunk    ("/H/py",    H_py,    start);
        writer.write_vlen_chunk    ("/H/pq",    H_pq,    start);
        writer.write_vlen_chunk    ("/H/pt",    H_pt,    start);
        writer.write_vlen_chunk    ("/H/pqt",   H_pqt,   start);
        writer.write_vlen_chunk    ("/H/pnt",   H_pnt,   start);
        writer.write_vlen_chunk    ("/H/pid",   H_pid,   start);

        writer.write_chunk<double> ("/L/theta", L_theta, start);
        writer.write_chunk<double> ("/L/phi",   L_phi,   start);
        writer.write_chunk<double> ("/L/vx",    L_vx,    start);
        writer.write_chunk<double> ("/L/vy",    L_vy,    start);
        writer.write_chunk<double> ("/L/vz",    L_vz,    start);
        writer.write_vlen_chunk    ("/L/x",     L_x,     start);
        writer.write_vlen_chunk    ("/L/y",     L_y,     start);
        writer.write_vlen_chunk    ("/L/z",     L_z,     start);
        writer.write_vlen_chunk    ("/L/dE",    L_dE,    start);
        writer.write_vlen_chunk    ("/L/px",    L_px,    start);
        writer.write_vlen_chunk    ("/L/py",    L_py,    start);
        writer.write_vlen_chunk    ("/L/pq",    L_pq,    start);
        writer.write_vlen_chunk    ("/L/pt",    L_pt,    start);
        writer.write_vlen_chunk    ("/L/pqt",   L_pqt,   start);
        writer.write_vlen_chunk    ("/L/pnt",   L_pnt,   start);
        writer.write_vlen_chunk    ("/L/pid",   L_pid,   start);

        auto t3 = std::chrono::steady_clock::now();

        double dt_read  = std::chrono::duration<double>(t1 - t0).count();
        double dt_proc  = std::chrono::duration<double>(t2 - t1).count();
        double dt_write = std::chrono::duration<double>(t3 - t2).count();

        std::cout << "Chunk [" << start << "," << start+count << ") : "
                  << "read=" << dt_read << "s  "
                  << "proc=" << dt_proc << "s  "
                  << "write=" << dt_write << "s  "
                  << "(" << start + count << "/" << N << ")\n";
    }

    std::cout << "\nTerminé → " << out_path << "\n";
    return 0;
}
