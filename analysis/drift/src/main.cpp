#include "h5reader.hpp"
#include "h5writer.hpp"
#include "detector.hpp"
#include "processor.hpp"
#include "pad_geometry.hpp"
#include "prof.hpp"

#include <cstdint>
#include <iostream>
#include <fstream>
#include <numeric>
#include <chrono>
#include <omp.h>
#include <ctime>
#include <map>

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

static uint64_t mix_seed(uint64_t base, uint64_t idx) {
    uint64_t z = base + 0x9E3779B97F4A7C15ULL * (idx + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static std::map<std::string, double> g_wt;

static void timed_vlen(H5Writer& w, const std::string& name,
                       const std::vector<std::vector<double> >& d, hsize_t start) {
    double t0 = omp_get_wtime();
    w.write_vlen_chunk(name, d, start);
    g_wt[name] += omp_get_wtime() - t0;
}

static std::vector<float> to_float(const std::vector<double>& v) {
    return std::vector<float>(v.begin(), v.end());
}

template<typename T>
static void timed_vlen(H5Writer& w, const std::string& name,
                       const std::vector<std::vector<T> >& d, hsize_t start) {
    double t0 = omp_get_wtime();
    w.write_vlen_chunk<T>(name, d, start);
    g_wt[name] += omp_get_wtime() - t0;
}

int main(int argc, char* argv[]) {
  // Usage : process [N] [seuil_pos] [seuil_neg] [n_threads] [frag_mask]
  //   frag_mask : 3 = H et L (défaut), 1 = H seul, 2 = L seul
  //   seuils en électrons par bin ; sans argument : valeurs de detector.hpp (+200 / -200).
  //   Pour tout lire (contrôles de conservation) : ./process 200 0 0
  //   n_threads : 8 par défaut ; 1 pour des tirages reproductibles d'une version à l'autre.

    const std::string in_path  = "../../../simulation_output/T2F_G4_40mu.h5";
    const std::string out_path = "../drift_output/drift_G4_40mu_merged.h5";

    //Mesure du temps cpu
    double wall0 = omp_get_wtime();
    
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
    if (argc > 2) cfg.seuil_qt_pos = std::stod(argv[2]);
    if (argc > 3) cfg.seuil_qt_neg = std::stod(argv[3]);
    std::cout << "seuil_qt_pos      = " << cfg.seuil_qt_pos << " e-\n";
    std::cout << "seuil_qt_neg      = " << cfg.seuil_qt_neg << " e-\n";

    // --- Plan de pads ---
    PadPlaneParams pParam ;
    
    PadPlane plane(pParam, cfg);   // construit aussi la table d'induction (~3 s, ~80 Mo)

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
    writer.write_attribute<double>("/metadata", "Fano",          cfg.Fano);
    writer.write_attribute<double>("/metadata", "DT_kg",         cfg.DT_kg);
    writer.write_attribute<double>("/metadata", "DT_ga",         cfg.DT_ga);
    writer.write_attribute<double>("/metadata", "DL_kg",         cfg.DL_kg);
    writer.write_attribute<double>("/metadata", "DL_ga",         cfg.DL_ga);
    writer.write_attribute<double>("/metadata", "ve_kg",         cfg.ve_kg);
    writer.write_attribute<double>("/metadata", "ve_ga",         cfg.ve_ga);
    writer.write_attribute<double>("/metadata", "delta_t",       cfg.delta_t);
    writer.write_attribute<double>("/metadata", "seuil_qt_pos",  cfg.seuil_qt_pos);
    writer.write_attribute<double>("/metadata", "seuil_qt_neg",  cfg.seuil_qt_neg);
    writer.write_attribute<int>   ("/metadata", "n_bins",        cfg.n_bins());

    writer.write_attribute<double>("/metadata", "z_anode",     (cfg.dkg + cfg.dga));
    writer.write_attribute<double>("/metadata", "pad_size",    pParam.pad_size);
    writer.write_attribute<double>("/metadata", "inter_pad",   pParam.inter_pad);
    writer.write_attribute<double>("/metadata", "r_active",    pParam.r_active);
    writer.write_attribute<double>("/metadata", "r_quinconce", pParam.r_quinconce);
    writer.write_attribute<double>("/metadata", "r_transition",pParam.r_transition);
    writer.write_attribute<int>   ("/metadata", "n_rings",     pParam.n_rings);
    writer.write_attribute<double>("/metadata", "y_offset",    pParam.y_offset);
    writer.write_attribute<double>("/metadata", "reach_pitch", pParam.reach_pitch);
    writer.write_attribute<double>("/metadata", "bound_margin", pParam.bound_margin);
    writer.write_attribute<int>("/metadata", "n_tau", plane.table().n_tau());
    writer.write_attribute<int>("/metadata", "n_pads", (int)plane.pads().size());

    for (const std::string &frag : {"/H", "/L"}) {
      writer.add_dataset<float>     (frag + "/KE");
      writer.add_dataset<float>     (frag + "/theta");
      writer.add_dataset<float>     (frag + "/phi");
      writer.add_dataset<float>     (frag + "/vx");
      writer.add_dataset<float>     (frag + "/vy");
      writer.add_dataset<float>     (frag + "/vz");
      writer.add_vlen_dataset<float>(frag + "/x");
      writer.add_vlen_dataset<float>(frag + "/y");
      writer.add_vlen_dataset<float>(frag + "/z");
      writer.add_vlen_dataset<float>(frag + "/dE");
      writer.add_vlen_dataset<float>(frag + "/px");
      writer.add_vlen_dataset<float>(frag + "/py");
      writer.add_vlen_dataset<float>(frag + "/pq");
      writer.add_vlen_dataset<float>(frag + "/pt");        
      writer.add_vlen_dataset<float>(frag + "/pqt");        
      writer.add_vlen_dataset<float>(frag + "/pnt");        
      writer.add_vlen_dataset<float>(frag + "/pid");
      //        writer.add_vlen_dataset<float>     (frag + "/pQ");        
    }

    // --- Boucle par chunks ---
    const hsize_t chunk_size = 20000;
    int           nthreads   = 8;
    if (argc > 4) nthreads = std::max(1, std::stoi(argv[4]));

    std::cout << "\n\033[1;32m" << nthreads << " running threads...\033[0m\n\n";

    int frag_mask = 3;
    if (argc > 5) frag_mask = std::stoi(argv[5]);
    std::cout << "frag_mask         = " << frag_mask << "\n";
    const std::vector<double> empty_vec;      // trace vide, en lecture seule (thread-safe)
    
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

        auto H_KE = reader.read_chunk<double>("/H/KE", start, count);
        auto L_KE = reader.read_chunk<double>("/L/KE", start, count);        
        auto H_trace = reader.read_vlen_chunk("/H/raw_trace", start, count);
        auto L_trace = reader.read_vlen_chunk("/L/raw_trace", start, count);

        auto t1 = std::chrono::steady_clock::now();

	std::vector<float> H_KEf(count);
        std::vector<float> H_theta(count), H_phi(count);
        std::vector<float> H_vx(count), H_vy(count), H_vz(count);
        std::vector<float> L_KEf(count);
        std::vector<float> L_theta(count), L_phi(count);
        std::vector<float> L_vx(count), L_vy(count), L_vz(count);

        std::vector<std::vector<float>> H_x(count), H_y(count), H_z(count), H_dE(count);
        std::vector<std::vector<float>> H_px(count), H_py(count), H_pq(count);
        std::vector<std::vector<float>> H_pt(count), H_pqt(count), H_pQ(count);
        std::vector<std::vector<int32_t>> H_pnt(count), H_pid(count);
        std::vector<std::vector<float>> L_x(count), L_y(count), L_z(count), L_dE(count);
        std::vector<std::vector<float>> L_px(count), L_py(count), L_pq(count);
        std::vector<std::vector<float>> L_pt(count), L_pqt(count), L_pQ(count);
        std::vector<std::vector<int32_t>> L_pnt(count), L_pid(count);

        std::vector<std::vector<PadCharge>> H_pads(count), L_pads(count);

#pragma omp parallel for num_threads(nthreads) schedule(dynamic, 16)
        for (hsize_t i = 0; i < count; ++i) {
          int tid = omp_get_thread_num();
	  procs[tid].reseed(mix_seed(base_seed, start + i));

            EventInput in;
            in.H = { merge_trace(H_trace[i], n_merge), 0, 0, 0 };
            in.L = { merge_trace(L_trace[i], n_merge), 0, 0, 0 };

            EventOutput out;
            procs[tid].process(in, out);
	    
	    FragTrace fH = { &out.H.x, &out.H.y, &out.H.z, &out.H.n_electrons, &out.H.sigma_T };
            FragTrace fL = { &out.L.x, &out.L.y, &out.L.z, &out.L.n_electrons, &out.L.sigma_T };
            if (!(frag_mask & 1)) {
	      fH.x = &empty_vec; fH.y = &empty_vec; fH.z = &empty_vec;
	      fH.n_electrons = &empty_vec; fH.sigma_T = &empty_vec;
            }
            if (!(frag_mask & 2)) {
	      fL.x = &empty_vec; fL.y = &empty_vec; fL.z = &empty_vec;
	      fL.n_electrons = &empty_vec; fL.sigma_T = &empty_vec;
            }
            plane.project_pair(fH, fL, cfg, qt_bufs[tid], H_pads[i], L_pads[i]);

	    // H_pads[i] = plane.project(out.H.x, out.H.y, out.H.z,
            //                           out.H.n_electrons, out.H.sigma_T, out.H.sigma_L,
            //                           cfg, qt_bufs[tid]);
            // L_pads[i] = plane.project(out.L.x, out.L.y, out.L.z,
            //                           out.L.n_electrons, out.L.sigma_T, out.L.sigma_L,
            //                           cfg, qt_bufs[tid]);

	    H_KEf[i] = (float)H_KE[i];
            H_theta[i] = (float)out.H.dir.theta; H_phi[i] = (float)out.H.dir.phi;
            H_vx[i] = (float)out.H.vx; H_vy[i] = (float)out.H.vy; H_vz[i] = (float)out.H.vz;
            H_x[i] = to_float(out.H.x); H_y[i] = to_float(out.H.y); H_z[i] = to_float(out.H.z); H_dE[i] = to_float(out.H.dE);
            
            // double Sum = 0.0;
            for (const auto& pc : H_pads[i]) {
                H_pnt[i].push_back((int32_t)pc.t.size());
                for (double t : pc.t)  H_pt[i].push_back((float)t);
                for (double qt : pc.qt) H_pqt[i].push_back((float)qt);
		//                for (double Qc : pc.Q)  H_pQ[i].push_back(Qc);
                H_px[i].push_back((float)pc.x);
                H_py[i].push_back((float)pc.y);
                H_pq[i].push_back((float)pc.q);
                H_pid[i].push_back((int32_t)pc.pad_id);

		// Sum+=pc.q*cfg.W ;
	    }

	    // if(fabs(Sum-H_KE[i])/H_KE[i]>30.e-2)
	    //   std::cout << H_KE[i] << " " << Sum << std::endl ;

	    L_KEf[i] = (float)L_KE[i];
            L_theta[i] = (float)out.L.dir.theta; L_phi[i] = (float)out.L.dir.phi;
            L_vx[i] = (float)out.L.vx; L_vy[i] = (float)out.L.vy; L_vz[i] = (float)out.L.vz;
            L_x[i] = to_float(out.L.x); L_y[i] = to_float(out.L.y); L_z[i] = to_float(out.L.z); L_dE[i] = to_float(out.L.dE);

            for (const auto& pc : L_pads[i]) {
                L_pnt[i].push_back((int32_t)pc.t.size());
                for (double t : pc.t)  L_pt[i].push_back((float)t);
                for (double qt : pc.qt) L_pqt[i].push_back((float)qt);
		//                for (double Qc : pc.Q)  L_pQ[i].push_back(Qc);
                L_px[i].push_back((float)pc.x);
                L_py[i].push_back((float)pc.y);
                L_pq[i].push_back((float)pc.q);
                L_pid[i].push_back((int32_t)pc.pad_id);
            }
        }

        auto t2 = std::chrono::steady_clock::now();

	writer.write_chunk<float>("/H/KE", H_KEf, start);
        writer.write_chunk<float> ("/H/theta", H_theta, start);
        writer.write_chunk<float> ("/H/phi",   H_phi,   start);
        writer.write_chunk<float> ("/H/vx",    H_vx,    start);
        writer.write_chunk<float> ("/H/vy",    H_vy,    start);
        writer.write_chunk<float> ("/H/vz",    H_vz,    start);
	
	timed_vlen<float> (writer, "/H/x",   H_x,   start);
        timed_vlen<float> (writer, "/H/y",   H_y,   start);
        timed_vlen<float> (writer, "/H/z",   H_z,   start);
        timed_vlen<float> (writer, "/H/dE",  H_dE,  start);
        timed_vlen<float> (writer, "/H/px",  H_px,  start);
        timed_vlen<float> (writer, "/H/py",  H_py,  start);
        timed_vlen<float> (writer, "/H/pq",  H_pq,  start);
        timed_vlen<float> (writer, "/H/pt",  H_pt,  start);
        timed_vlen<float> (writer, "/H/pqt", H_pqt, start);
	//        timed_vlen<float> (writer, "/H/pQ",  H_pQ,  start);
        timed_vlen<int32_t> (writer, "/H/pnt", H_pnt, start);
        timed_vlen<int32_t> (writer, "/H/pid", H_pid, start);

	writer.write_chunk<float>("/L/KE", L_KEf, start);        
        writer.write_chunk<float> ("/L/theta", L_theta, start);
        writer.write_chunk<float> ("/L/phi",   L_phi,   start);
        writer.write_chunk<float> ("/L/vx",    L_vx,    start);
        writer.write_chunk<float> ("/L/vy",    L_vy,    start);
        writer.write_chunk<float>("/L/vz", L_vz, start);

	timed_vlen<float>(writer, "/L/x",   L_x,   start);
        timed_vlen<float>(writer, "/L/y",   L_y,   start);
        timed_vlen<float>(writer, "/L/z",   L_z,   start);
        timed_vlen<float>(writer, "/L/dE",  L_dE,  start);
        timed_vlen<float>(writer, "/L/px",  L_px,  start);
        timed_vlen<float>(writer, "/L/py",  L_py,  start);
        timed_vlen<float>(writer, "/L/pq",  L_pq,  start);
        timed_vlen<float>(writer, "/L/pt",  L_pt,  start);
        timed_vlen<float>(writer, "/L/pqt", L_pqt, start);
	//        timed_vlen<float>(writer, "/L/pQ",  L_pQ,  start);
        timed_vlen<int32_t>(writer, "/L/pnt", L_pnt, start);
        timed_vlen<int32_t>(writer, "/L/pid", L_pid, start);	
	
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


#ifdef T2F_PROF
    {
      ProfStats T = ProfStats();
      std::cout << "\n--- PROFIL (s cumulées sur les threads) ---\n";
      for (int k = 0; k < 64; ++k) {
	const ProfStats& S = g_prof[k];
	if (S.n_vox == 0) continue;
	std::cout << "thread " << k << " : tot=" << S.t_tot
		  << " vox=" << S.n_vox << " cell=" << S.n_cell << "\n";
	T.t_tot += S.t_tot; T.t_vec += S.t_vec; T.t_at += S.t_at;
	T.t_acc += S.t_acc; T.t_out += S.t_out;
        T.n_vox += S.n_vox;
        T.n_cell += S.n_cell;
        T.n_contrib += S.n_contrib;
        T.sum_ne += S.sum_ne;
        T.n_skip_sel += S.n_skip_sel;
        T.n_cell_touched += S.n_cell_touched;
        T.n_cell_keep += S.n_cell_keep;
        T.n_cell_wasted += S.n_cell_wasted;
	T.sum_nb += S.sum_nb;
        
      }
      std::cout << "psi_vector : " << T.t_vec << "  ("
                << 100.0 * T.t_vec / T.t_tot << " %)\n"
                << "psi_at     : " << T.t_at << "  ("
                << 100.0 * T.t_at / T.t_tot << " %)\n"
                << "accumul.   : " << T.t_acc << "  ("
                << 100.0 * T.t_acc / T.t_tot << " %)\n"
                << "lecture    : " << T.t_out << "  ("
                << 100.0 * T.t_out / T.t_tot << " %)\n"
                << "reste      : "
                << T.t_tot - T.t_vec - T.t_at - T.t_acc - T.t_out << "\n"
                << "voxels=" << T.n_vox
                << "  cellules/voxel=" << (double)T.n_cell / T.n_vox
                << "  contrib/cellule=" << (double)T.n_contrib / T.n_cell
                << "\n"
                << "µs par cellule : " << 1.0e6 * T.t_tot / T.n_cell << "\n"
                << "  n_e moyen par voxel=" << T.sum_ne / T.n_vox << "\n"
                << "paires sautées par la sélection : "
                << 100.0 * T.n_skip_sel / (T.n_cell + T.n_skip_sel)
                << " % des paires\n"
                << "cellules gardées par la borne : "
                << 100.0 * T.n_cell_keep / T.n_cell_touched
                << " % des cellules touchées\n"
                << "cellules inutiles (électrode non lue) : "
                << 100.0 * T.n_cell_wasted / T.n_cell
                << " % des cellules calculées\n"
		<< "nb moyen par cellule : " << T.sum_nb / T.n_cell
		<< "  (n_tau = " << plane.table().n_tau() << ")\n";
      
    }
#endif

    std::cout << "voxels avec charge au-delà de r_active (frac > 1e-6) : "
              << pad_beyond_active_count() << "\n";

    std::cout << "traces traitées : " << pad_traces_count()
              << " ; avec au moins un voxel au-delà de r_active : "
              << pad_traces_beyond_count() << "\n";

    std::cout << "\nTemps réel total : " << omp_get_wtime() - wall0 << " s\n";
    for (std::map<std::string,double>::const_iterator it = g_wt.begin(); it != g_wt.end(); ++it)
      std::cout << "write " << it->first << " : " << it->second << " s\n";
    
    std::cout << "\nTerminé → " << out_path << "\n";
    return 0;
}
