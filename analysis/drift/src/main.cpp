#include "h5reader.hpp"
#include "h5writer.hpp"
#include "detector.hpp"
#include "processor.hpp"
#include "pad_geometry.hpp"
#include "prof.hpp"
#include "config_file.hpp"
#include "derived_params.hpp"
#include "noise.hpp"
#include "cap_noise.hpp"
#include "preamp_response.hpp"
#include "readout.hpp"

// genere a chaque build par cmake/git_version.cmake (T2F_GIT_VERSION, T2F_GIT_BRANCH)
#if __has_include("git_version.hpp")
#include "git_version.hpp"
#endif
#ifndef T2F_GIT_VERSION
#define T2F_GIT_VERSION "unknown"
#define T2F_GIT_BRANCH  "unknown"
#endif

#include <cstdint>
#include <cstdlib>
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

// ---------------------------------------------------------------------------
// Paramètres d'exécution (hors detecteur et pads), lus dans le fichier d'input
// ---------------------------------------------------------------------------
struct RunParams {
    std::string in_path;
    std::string out_path;
    int    n_events;          // 0 = tous
    int    n_threads;
    int    chunk_size;
    int    base_seed;
    int    frag_mask;         // 3 = H et L, 1 = H seul, 2 = L seul
    double step_size_target;  // cm

    // --- bruit electronique et lecture (voir noise.hpp, readout.hpp) ---
    int         noise_enable;       // 0 : meme chaine sans bruit ni seuil de lecture ; 1 : avec bruit
    std::string preamp_response;    // "test_gauss" (seule reponse disponible pour l'instant)
    double      preamp_gauss_fwhm;  // ns
    double      enc_total;          // e- rms (modele uniform seulement)
    double      f_par, f_ser, f_can;
    // modele "capacitance" (voir cap_noise.hpp)
    std::string noise_model;        // "uniform" : un seul ENC (enc_total) ; "capacitance" : ENC(C) par pad
    double      enc0;               // e- : ENC a C = 0
    double      enc_ref;            // e- : ENC a la capacite c_ref_pf
    double      c_ref_pf;           // pF
    double      c_cable_pf;         // pF : cable + connecteurs
    double      c_routing_pf;       // pF : routage (valeur par defaut commune a tous les pads)
    std::string pad_capacitance_file;   // "" : C = c_cable_pf + c_routing_pf pour tous les pads
    double      adc_period_ns;
    double      read_thr_sigma;
    int         read_n_consec;
    double      window_margin_ns;
    int         noise_seed;
    // --- pads fantomes (voir ghost.hpp) ---
    int         ghost_enable;       // 0 : pas de fantomes ; sans effet si noise_enable = 0
    double      ghost_prob;         // < 0 : p calibree au lancement ; >= 0 : valeur imposee
    double      ghost_window_ns;
    int         ghost_calib_trials;
};

// Lit toutes les cles du fichier. Les valeurs absentes gardent le defaut de
// DetectorConfig / PadPlaneParams / RunParams. Une cle inconnue est une erreur.
static void read_config(ConfigFile& c, RunParams& R, DetectorConfig& cfg, PadPlaneParams& pp, TableOptions& to)
{
    // --- fichiers (obligatoires) ---
    R.in_path  = c.get_string("input_file");
    R.out_path = c.get_string("output_file");

    // --- execution ---
    R.n_events         = c.get_int   ("n_events",         0);
    R.n_threads        = c.get_int   ("n_threads",        8);
    R.chunk_size       = c.get_int   ("chunk_size",       20000);
    R.base_seed        = c.get_int   ("base_seed",        42);
    R.frag_mask        = c.get_int   ("frag_mask",        3);
    R.step_size_target = c.get_double("step_size_target", 0.05);

    // --- lecture (sur le signal vrai, en electrons par bin) ---
    // Seule la polarite positive est lue (la charge collectee est positive) : il n'y a plus de
    // seuil negatif. seuil_qt est la preselection des pads ; avec le bruit, le seuil de lecture
    // (read_thr_sigma) s'applique ensuite sur la sortie bruitee.
    const double seuil_qt = c.get_double("seuil_qt", 10.0);
    cfg.seuil_qt_pos = seuil_qt;
    cfg.seuil_qt_neg = -1.0e300;
    cfg.eps_qt       = c.get_double("eps_qt",       cfg.eps_qt);

    // --- bruit electronique et lecture ---
    R.noise_enable      = c.get_int   ("noise_enable",      0);
    R.preamp_response   = c.get_string("preamp_response",   "test_gauss");
    R.preamp_gauss_fwhm = c.get_double("preamp_gauss_fwhm", 190.0);
    R.enc_total         = c.get_double("enc_total",         2000.0);
    R.f_par             = c.get_double("f_par",             0.475);
    R.f_ser             = c.get_double("f_ser",             0.475);
    R.f_can             = c.get_double("f_can",             0.05);
    R.noise_model       = c.get_string("noise_model",       "uniform");
    R.enc0              = c.get_double("enc0",              1150.0);
    R.enc_ref           = c.get_double("enc_ref",           1400.0);
    R.c_ref_pf          = c.get_double("c_ref_pf",          93.0);
    R.c_cable_pf        = c.get_double("c_cable_pf",        93.0);
    R.c_routing_pf      = c.get_double("c_routing_pf",      10.0);
    R.pad_capacitance_file = c.get_string("pad_capacitance_file", "");
    R.adc_period_ns     = c.get_double("adc_period_ns",     20.0);
    R.read_thr_sigma    = c.get_double("read_thr_sigma",    4.0);
    R.read_n_consec     = c.get_int   ("read_n_consec",     1);
    R.window_margin_ns  = c.get_double("window_margin_ns",  100.0);
    R.noise_seed        = c.get_int   ("noise_seed",        12345);
    R.ghost_enable       = c.get_int   ("ghost_enable",       1);
    R.ghost_prob         = c.get_double("ghost_prob",         -1.0);
    R.ghost_window_ns    = c.get_double("ghost_window_ns",    940.0);
    R.ghost_calib_trials = c.get_int   ("ghost_calib_trials", 20000);

    // --- detecteur ---
    cfg.dkg          = c.get_double("dkg",          cfg.dkg);
    cfg.dga          = c.get_double("dga",          cfg.dga);
    cfg.vertex_r_max = c.get_double("vertex_r_max", cfg.vertex_r_max);
    cfg.Fano         = c.get_double("Fano",         cfg.Fano);
    cfg.W            = c.get_double("W",            cfg.W);
    cfg.DT_kg        = c.get_double("DT_kg",        cfg.DT_kg);
    cfg.DT_ga        = c.get_double("DT_ga",        cfg.DT_ga);
    cfg.DL_kg        = c.get_double("DL_kg",        cfg.DL_kg);
    cfg.DL_ga        = c.get_double("DL_ga",        cfg.DL_ga);
    cfg.ve_kg        = c.get_double("ve_kg",        cfg.ve_kg);
    cfg.ve_ga        = c.get_double("ve_ga",        cfg.ve_ga);
    cfg.delta_t      = c.get_double("delta_t",      cfg.delta_t);

    // --- plan de pads ---
    pp.pad_size     = c.get_double("pad_size",     pp.pad_size);
    pp.inter_pad    = c.get_double("inter_pad",    pp.inter_pad);
    pp.r_quinconce  = c.get_double("r_quinconce",  pp.r_quinconce);
    pp.r_transition = c.get_double("r_transition", pp.r_transition);
    pp.r_active     = c.get_double("r_active",     pp.r_active);
    pp.n_rings      = c.get_int   ("n_rings",      pp.n_rings);
    pp.y_offset     = c.get_double("y_offset",     pp.y_offset);

    // --- induction ---
    pp.reach_pitch  = c.get_double("reach_pitch",  pp.reach_pitch);
    pp.bound_margin = c.get_double("bound_margin", pp.bound_margin);

    // --- table d'induction (0 = automatique) et controles ---
    to.n_d          = c.get_int   ("table_n_d",      20);
    to.n_img        = c.get_int   ("table_n_img",    20);
    to.m_cells      = c.get_int   ("table_m_cells",  0);
    to.step_out     = c.get_double("table_step_out", 0.0);
    to.r_src        = c.get_double("table_r_src",    0.0);
    to.kappa        = c.get_double("table_kappa",    3.0);
    to.step_cap     = c.get_double("table_step_cap", 0.0);
    to.acc_tol      = c.get_double("table_acc_tol",  1.0e-3);
    to.reach_tol    = c.get_double("reach_tol",      1.0e-5);
    to.max_table_mb = c.get_double("max_table_mb",   500.0);
    to.R            = c.get_int   ("table_R",        0);
    to.tau_tol      = c.get_double("table_tau_tol",  0.005);
    to.tau_dmin     = c.get_double("table_tau_dmin", 2.0);
    to.tau_Rref     = c.get_int   ("table_tau_Rref", 12);
    to.ignore       = c.get_string("ignore",         "");

    // une cle jamais lue = faute de frappe : on s'arrete
    c.check_unused();

    // --- verifications elementaires ---
    if (to.n_d < 4)                              throw std::runtime_error("table_n_d doit etre >= 4");
    if (to.R < 0 || to.R > 12)                   throw std::runtime_error("table_R doit etre entre 0 (auto) et 12");
    if (to.tau_tol < 0.0 || to.tau_dmin < 0.0)   throw std::runtime_error("table_tau_tol et table_tau_dmin doivent etre >= 0");
    if (to.tau_Rref < 2 || to.tau_Rref > 32)     throw std::runtime_error("table_tau_Rref doit etre entre 2 et 32");
    if (to.n_img < 1)                            throw std::runtime_error("table_n_img doit etre >= 1");
    if (to.kappa < 0.0 || (to.kappa > 0.0 && to.kappa < 1.0))
                                                 throw std::runtime_error("table_kappa doit etre 0 (grille uniforme) ou >= 1");
    if (to.step_cap < 0.0 || to.acc_tol < 0.0)   throw std::runtime_error("table_step_cap et table_acc_tol doivent etre >= 0");
    if (to.step_out < 0.0 || to.r_src < 0.0 || to.m_cells < 0)
                                                 throw std::runtime_error("table_step_out, table_r_src, table_m_cells doivent etre >= 0");
    if (R.n_events < 0)                          throw std::runtime_error("n_events doit etre >= 0");
    if (R.n_threads < 1)                         throw std::runtime_error("n_threads doit etre >= 1");
    if (R.chunk_size < 1)                        throw std::runtime_error("chunk_size doit etre >= 1");
    if (R.frag_mask < 0 || R.frag_mask > 3)      throw std::runtime_error("frag_mask doit etre 0, 1, 2 ou 3");
    if (R.step_size_target <= 0.0)               throw std::runtime_error("step_size_target doit etre > 0");
    if (cfg.dkg <= 0.0 || cfg.dga <= 0.0)        throw std::runtime_error("dkg et dga doivent etre > 0");
    if (cfg.ve_kg <= 0.0 || cfg.ve_ga <= 0.0)    throw std::runtime_error("ve_kg et ve_ga doivent etre > 0");
    if (cfg.delta_t <= 0.0)                      throw std::runtime_error("delta_t doit etre > 0");
    if (pp.pad_size <= 0.0 || pp.inter_pad < 0.0) throw std::runtime_error("pad_size > 0 et inter_pad >= 0 requis");
    if (seuil_qt < 0.0)                          throw std::runtime_error("seuil_qt doit etre >= 0");
    if (R.noise_enable != 0 && R.noise_enable != 1) throw std::runtime_error("noise_enable doit etre 0 ou 1");
    if (R.preamp_response != "test_gauss")       throw std::runtime_error("preamp_response : seule la valeur test_gauss est disponible pour l'instant");
    if (R.preamp_gauss_fwhm <= 0.0)              throw std::runtime_error("preamp_gauss_fwhm doit etre > 0");
    if (R.enc_total < 0.0)                       throw std::runtime_error("enc_total doit etre >= 0");
    if (R.f_par < 0.0 || R.f_ser < 0.0 || R.f_can < 0.0)
                                                 throw std::runtime_error("f_par, f_ser, f_can doivent etre >= 0");
    if (R.enc_total > 0.0 && R.f_par + R.f_ser + R.f_can <= 0.0)
                                                 throw std::runtime_error("f_par + f_ser + f_can doit etre > 0");
    if (R.noise_model != "uniform" && R.noise_model != "capacitance")
                                                 throw std::runtime_error("noise_model doit etre uniform ou capacitance");
    if (R.noise_model == "capacitance") {
        if (R.enc0 <= 0.0)                       throw std::runtime_error("enc0 doit etre > 0");
        if (R.enc_ref < R.enc0)                  throw std::runtime_error("enc_ref doit etre >= enc0");
        if (R.c_ref_pf <= 0.0)                   throw std::runtime_error("c_ref_pf doit etre > 0");
        if (R.c_cable_pf < 0.0 || R.c_routing_pf < 0.0)
                                                 throw std::runtime_error("c_cable_pf et c_routing_pf doivent etre >= 0");
    } else if (!R.pad_capacitance_file.empty()) {
        throw std::runtime_error("pad_capacitance_file demande noise_model = capacitance");
    }
    if (R.adc_period_ns <= 0.0)                  throw std::runtime_error("adc_period_ns doit etre > 0");
    if (R.read_thr_sigma < 0.0)                  throw std::runtime_error("read_thr_sigma doit etre >= 0");
    if (R.read_n_consec < 1)                     throw std::runtime_error("read_n_consec doit etre >= 1");
    if (R.window_margin_ns < 0.0)                throw std::runtime_error("window_margin_ns doit etre >= 0");
    if (R.ghost_enable != 0 && R.ghost_enable != 1) throw std::runtime_error("ghost_enable doit etre 0 ou 1");
    if (R.ghost_prob > 1.0)                      throw std::runtime_error("ghost_prob doit etre <= 1 (negatif : calibration automatique)");
    if (R.ghost_window_ns < 0.0)                 throw std::runtime_error("ghost_window_ns doit etre >= 0");
    if (R.ghost_calib_trials < 100)              throw std::runtime_error("ghost_calib_trials doit etre >= 100");
}

int main(int argc, char* argv[]) {
  // Usage : process fichier.cfg [cle=valeur ...]
  //   Tous les parametres sont dans le fichier d'input (voir t2f.cfg).
  //   Les arguments cle=valeur surchargent le fichier, par exemple :
  //       ./process t2f.cfg n_events=200 seuil_qt=0
  //   Le resultat ne depend pas de n_threads : chaque evenement a sa propre graine (identique bit a bit a 1, 2 ou 8 threads).

    if (argc < 2) {
        std::cerr << "Usage : " << argv[0] << " fichier.cfg [cle=valeur ...]\n";
        return 1;
    }

    //Mesure du temps cpu
    double wall0 = omp_get_wtime();

    // --- Lecture du fichier d'input et des surcharges ---
    ConfigFile     conf;
    RunParams      R;
    DetectorConfig cfg;
    PadPlaneParams pParam;
    TableOptions   topt;
    try {
        conf.load(argv[1]);
        for (int a = 2; a < argc; ++a) conf.set_override(argv[a]);
        read_config(conf, R, cfg, pParam, topt);
    } catch (const std::exception& e) {
        std::cerr << "Erreur de configuration : " << e.what() << "\n";
        return 1;
    }

    const std::string in_path  = R.in_path;
    const std::string out_path = R.out_path;
    const uint64_t    base_seed = (uint64_t)R.base_seed;
    const hsize_t     chunk_size = (hsize_t)R.chunk_size;
    const int         nthreads  = R.n_threads;
    const int         frag_mask = R.frag_mask;

    std::cout << "Fichier d'input   = " << argv[1] << "\n";
    std::cout << "input_file        = " << in_path  << "\n";
    std::cout << "output_file       = " << out_path << "\n";

    // --- Lecture metadata ---
    H5Reader reader(in_path);
    const hsize_t N_total = reader.size("/H/KE");
    const double step_size_in = reader.read_attribute<double>("/metadata", "step_size") / 10.0;

    const hsize_t N = (R.n_events > 0) ? std::min((hsize_t)R.n_events, N_total) : N_total;

    const double step_size_target = R.step_size_target;
    const int    n_merge          = std::max(1, (int)(step_size_target / step_size_in));
    const double step_size_eff    = n_merge * step_size_in;

    std::cout << "step_size_in      = " << step_size_in  * 10 << " mm\n";
    std::cout << "step_size_target  = " << step_size_target * 10 << " mm\n";
    std::cout << "n_merge           = " << n_merge             << "\n";
    std::cout << "step_size_eff     = " << step_size_eff * 10  << " mm\n";
    std::cout << "Events à traiter  = " << N                   << "\n";

    cfg.step_size = step_size_eff;
    std::cout << "seuil_qt          = " << cfg.seuil_qt_pos << " e-  (polarite positive seulement)\n";
    std::cout << "eps_qt            = " << cfg.eps_qt << " e-\n";
    std::cout << "frag_mask         = " << frag_mask << "\n";
    std::cout << "\n\033[1;32m" << nthreads << " running threads...\033[0m\n\n";

    const std::vector<double> empty_vec;      // trace vide, en lecture seule (thread-safe)

    // --- Parametres derives et controles (arret explicite si un controle echoue) ---
    CheckList     checks;
    DerivedParams D;
    try {
        checks.set_ignored(topt.ignore);
        topt.R = resolve_table_R(cfg, pParam, topt, checks);   // choix / verification du pas de table en tau
        D = derive_params(cfg, pParam, topt, checks);
        print_derived(D, cfg);
        checks.finish();
    } catch (const std::exception& e) {
        std::cerr << "\nArret : " << e.what() << "\n";
        return 1;
    }
    cfg.tau_tail_ns = D.tau_tail_ns;   // n_bins() en depend : a fixer avant les Processor et les tampons

    // --- Reponse du preampli, modele de bruit, lecture (voir noise.hpp, readout.hpp) ---
    // La reponse reelle (MINOS) sera une autre classe derivee de PreampResponse, a brancher ici.
    GaussTestResponse gauss_resp(R.preamp_gauss_fwhm);
    const PreampResponse* response = &gauss_resp;
    const uint64_t noise_seed = (uint64_t)(unsigned int)R.noise_seed;

    NoiseModel* noise_ptr   = 0;
    Readout*    readout_ptr = 0;
    CapNoiseModel* cap_ptr  = 0;
    try {
        const int adc_ratio = (int)std::floor(R.adc_period_ns / cfg.delta_t + 0.5);
        if (adc_ratio < 1 || std::fabs(adc_ratio * cfg.delta_t - R.adc_period_ns) > 1.0e-6 * cfg.delta_t)
            throw std::runtime_error("adc_period_ns doit etre un multiple entier de delta_t");
        NoiseParams nparams;
        nparams.enc_total = R.enc_total;
        nparams.f_par = R.f_par; nparams.f_ser = R.f_ser; nparams.f_can = R.f_can;
        if (R.noise_model == "capacitance") {
            // le NoiseModel est celui de la capacite de reference ; le seuil de lecture (global) est
            // read_thr_sigma * enc_ref ; chaque pad a ensuite son bruit serie (set_pad_noise)
            CapNoiseParams cp;
            cp.enc0 = R.enc0; cp.enc_ref = R.enc_ref; cp.c_ref_pf = R.c_ref_pf;
            cp.f_par = R.f_par; cp.f_ser = R.f_ser; cp.f_can = R.f_can;
            cap_ptr = new CapNoiseModel(cp);
            cap_ptr->fill_noise_params(nparams);
        }
        nparams.dt = cfg.delta_t;
        nparams.adc_ratio = adc_ratio;
        ReadoutParams rparams;
        rparams.noise_on  = (R.noise_enable == 1);
        rparams.thr_sigma = R.read_thr_sigma;
        rparams.n_consec  = R.read_n_consec;
        rparams.margin_ns = R.window_margin_ns;
        std::vector<double> h_resp;
        response->sample(cfg.delta_t, h_resp);
        noise_ptr   = new NoiseModel(nparams, h_resp);
        readout_ptr = new Readout(rparams, *noise_ptr, cfg.n_bins());
    } catch (const std::exception& e) {
        std::cerr << "\nArret (bruit / lecture) : " << e.what() << "\n";
        return 1;
    }
    const NoiseModel& noise   = *noise_ptr;
    const Readout&    readout = *readout_ptr;

    std::cout << "reponse du preampli : " << response->name() << " (" << response->describe() << ")\n";
    if (R.noise_enable == 1) {
        std::cout << noise.summary() << "\n";
        std::cout << "seuil de lecture  = " << readout.threshold() << " e- (" << R.read_thr_sigma
                  << " x " << noise.params().enc_total << (cap_ptr ? " = enc_ref" : " = enc_total")
                  << "), global, " << R.read_n_consec << " echantillon(s) consecutif(s)\n";
        if (cap_ptr) {
            std::cout << "modele capacitance : ENC(C)^2 = enc0^2 + (b C)^2 avec enc0 = " << R.enc0 << " e-, b = "
                      << cap_ptr->slope() << " e-/pF (enc_ref = " << R.enc_ref << " e- a c_ref = " << R.c_ref_pf
                      << " pF) ; enc_total n'est pas utilise\n";
        } else if (R.noise_model == "uniform") {
            std::cout << "modele de bruit   : uniform (meme ENC pour tous les pads)\n";
        }
    } else {
        std::cout << "bruit desactive : meme chaine (reponse, CAN de " << R.adc_period_ns
                  << " ns), sans bruit ni seuil de lecture\n";
    }
    std::cout << "fenetre ecrite    = signal vrai + queue complete de h (" << readout.tail_bins() << " bins) + marge ("
              << readout.margin_bins() << " bins) de chaque cote ; grille de lecture " << readout.grid_bins()
              << " bins (physique " << readout.n_bins() << ")\n";

    // --- Plan de pads ---
    PadPlane plane(pParam, cfg, D.tp, D.reach_cm);   // construit aussi la table d'induction

    // --- Bruit par pad (modele capacitance) : capacite totale et ENC de chaque pad ---
    std::vector<double> pad_c_pf, pad_enc, pad_enc_scale;      // meme ordre que plane.pads()
    if (cap_ptr) {
        try {
            const size_t npd = plane.pads().size();
            std::vector<int> ids(npd);
            for (size_t ip = 0; ip < npd; ++ip) ids[ip] = plane.pads()[ip].id;
            if (!R.pad_capacitance_file.empty()) read_pad_capacitances(R.pad_capacitance_file, ids, pad_c_pf);
            else                                 pad_c_pf.assign(npd, R.c_cable_pf + R.c_routing_pf);
            std::vector<double> ser_scale(npd);
            pad_enc.assign(npd, 0.0); pad_enc_scale.assign(npd, 1.0);
            for (size_t ip = 0; ip < npd; ++ip) {
                ser_scale[ip]    = cap_ptr->ser_scale(pad_c_pf[ip]);
                pad_enc[ip]      = cap_ptr->enc_total(pad_c_pf[ip]);
                pad_enc_scale[ip] = cap_ptr->enc_scale(pad_c_pf[ip]);
            }
            readout_ptr->set_pad_noise(ids, ser_scale, pad_enc_scale);
        } catch (const std::exception& e) {
            std::cerr << "\nArret (capacites des pads) : " << e.what() << "\n";
            return 1;
        }
        double cmin = pad_c_pf[0], cmax = pad_c_pf[0], emin = pad_enc[0], emax = pad_enc[0];
        for (size_t ip = 1; ip < pad_c_pf.size(); ++ip) {
            if (pad_c_pf[ip] < cmin) cmin = pad_c_pf[ip];
            if (pad_c_pf[ip] > cmax) cmax = pad_c_pf[ip];
            if (pad_enc[ip] < emin)  emin = pad_enc[ip];
            if (pad_enc[ip] > emax)  emax = pad_enc[ip];
        }
        std::cout << "capacite par pad  : ";
        if (R.pad_capacitance_file.empty())
            std::cout << "valeur unique " << cmin << " pF = c_cable_pf (" << R.c_cable_pf << ") + c_routing_pf ("
                      << R.c_routing_pf << ", valeur par defaut PROVISOIRE)";
        else
            std::cout << "lue dans " << R.pad_capacitance_file << " : de " << cmin << " a " << cmax << " pF";
        std::cout << " -> ENC des pads de " << emin << " a " << emax << " e-\n";
    }

    // --- Pads fantomes (avec bruit seulement) : p calibree ici, liste de tous les pads du detecteur ---
    const bool ghosts_on = (R.noise_enable == 1 && R.ghost_enable == 1);
    double ghost_p_mean = 0.0, ghost_p_min = 0.0, ghost_p_max = 0.0;   // modele capacitance : p selon l'ENC de chaque pad
    GhostModel* ghost_ptr = 0;
    if (ghosts_on) {
        try {
            GhostParams gp;
            gp.enable = true; gp.prob = R.ghost_prob; gp.window_ns = R.ghost_window_ns;
            gp.calib_trials = R.ghost_calib_trials;
            ghost_ptr = new GhostModel(noise, gp, readout.threshold(), R.read_n_consec, readout.n_bins());
            GaussRng gr(mix_seed(noise_seed, 0x67687374ULL));      // graine de calibration, fixe par noise_seed
            ghost_ptr->calibrate(gr);
            if (cap_ptr) ghost_ptr->calibrate_scales(gr, pad_enc_scale);   // p exacte par valeur d'ENC (sans effet si ghost_prob impose)
            std::vector<PadSite> sites;
            for (size_t ip = 0; ip < plane.pads().size(); ++ip) {
                PadSite ps;
                ps.pad_id = plane.pads()[ip].id; ps.x = plane.pads()[ip].x; ps.y = plane.pads()[ip].y;
                sites.push_back(ps);
            }
            readout_ptr->set_ghosts(ghost_ptr, sites);
        } catch (const std::exception& e) {
            std::cerr << "\nArret (pads fantomes) : " << e.what() << "\n";
            return 1;
        }
        std::cout << "pads fantomes     : p = " << ghost_ptr->prob() << " par pad sans signal";
        if (R.ghost_prob < 0.0) std::cout << " (calibree, +- " << ghost_ptr->prob_err() << ")";
        else                    std::cout << " (imposee)";
        std::cout << ", soit au plus ~" << ghost_ptr->prob() * (double)plane.pads().size()
                  << " fantomes par evenement (p x " << plane.pads().size() << " pads, moins les pads du signal) ; fenetre ecrite = "
                  << ghost_ptr->window_samples() << " echantillons\n";
        if (cap_ptr) {
            // p du pad de reference ci-dessus ; p de chaque pad selon son ENC (seuil global)
            double ps = 0.0, pmin = 1.0, pmax = 0.0;
            for (size_t ip = 0; ip < pad_enc_scale.size(); ++ip) {
                const double pp = ghost_ptr->prob_for_scale(pad_enc_scale[ip]);
                ps += pp; if (pp < pmin) pmin = pp; if (pp > pmax) pmax = pp;
            }
            ghost_p_mean = ps / (double)pad_enc_scale.size(); ghost_p_min = pmin; ghost_p_max = pmax;
            std::cout << "                    p par pad selon son ENC : moyenne " << ghost_p_mean << ", de " << pmin << " a " << pmax
                      << " (p de reference = pad a c_ref_pf ; calibree pour chaque ENC distinct)\n";
        }
    } else if (R.noise_enable == 1) {
        std::cout << "pads fantomes     : desactives (ghost_enable = 0)\n";
    }

    try {
        check_table(plane.table(), D, checks, topt.acc_tol, topt.reach_tol);
        checks.finish();
    } catch (const std::exception& e) {
        std::cerr << "\nArret : " << e.what() << "\n";
        return 1;
    }

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
    writer.write_attribute_string("/metadata", "config_file", std::string(argv[1]));
    writer.write_attribute_string("/metadata", "config_text", conf.text());
    writer.write_attribute_string("/metadata", "git_version", T2F_GIT_VERSION);
    writer.write_attribute_string("/metadata", "git_branch",  T2F_GIT_BRANCH);
    writer.write_attribute_string("/metadata", "emission_axis", cfg.emission_axis == EmissionAxis::Y ? "Y" : "Z");

    writer.write_attribute<double>("/metadata", "step_size_in",  step_size_in);
    writer.write_attribute<double>("/metadata", "step_size_eff", step_size_eff);
    writer.write_attribute<int>   ("/metadata", "n_merge",       n_merge);
    writer.write_attribute<int>   ("/metadata", "n_events",      (int)N);
    writer.write_attribute<int>   ("/metadata", "frag_mask",       frag_mask);

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
    writer.write_attribute<double>("/metadata", "seuil_qt",      cfg.seuil_qt_pos);
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
    writer.write_attribute<double>("/metadata", "tau_tail_ns",    cfg.tau_tail_ns);
    writer.write_attribute<double>("/metadata", "dtau",           D.dtau);
    writer.write_attribute<int>   ("/metadata", "n_h",            D.n_h);
    writer.write_attribute<int>   ("/metadata", "table_R",        D.R);
    writer.write_attribute<double>("/metadata", "table_tau_tol",  topt.tau_tol);
    writer.write_attribute<double>("/metadata", "table_tau_dmin", topt.tau_dmin);
    writer.write_attribute<int>   ("/metadata", "table_tau_Rref", topt.tau_Rref);
    writer.write_attribute<double>("/metadata", "T_gap_last_step_ns", D.T_frac_ns);
    writer.write_attribute<int>   ("/metadata", "table_n_out",    D.tp.n_out);
    writer.write_attribute<double>("/metadata", "table_step_out", D.tp.step_out);
    writer.write_attribute<int>   ("/metadata", "table_adaptive", D.tp.adaptive);
    writer.write_attribute<double>("/metadata", "table_kappa",    topt.kappa);
    writer.write_attribute<double>("/metadata", "table_r_src",    D.tp.r_src);
    writer.write_attribute<int>   ("/metadata", "table_m_cells",  D.tp.m_cells);
    writer.write_attribute<int>   ("/metadata", "table_n_img",    D.tp.n_img);
    writer.write_attribute<int>   ("/metadata", "table_n_d",      D.tp.n_d);
    writer.write_attribute<double>("/metadata", "reach_cm",       D.reach_cm);
    writer.write_attribute_string("/metadata", "ignored_checks",  checks.ignored_text());
    writer.write_attribute<int>("/metadata", "n_pads", (int)plane.pads().size());
    writer.write_attribute<double>("/metadata", "eps_qt", cfg.eps_qt);

    // bruit electronique et lecture
    writer.write_attribute<int>   ("/metadata", "noise_enable",     R.noise_enable);
    writer.write_attribute_string ("/metadata", "preamp_response",  response->name());
    writer.write_attribute_string ("/metadata", "preamp_response_info", response->describe());
    writer.write_attribute<double>("/metadata", "enc_total",        noise.params().enc_total);
    writer.write_attribute_string ("/metadata", "noise_model",      R.noise_model);
    if (cap_ptr) {
        writer.write_attribute<double>("/metadata", "enc0",             R.enc0);
        writer.write_attribute<double>("/metadata", "enc_ref",          R.enc_ref);
        writer.write_attribute<double>("/metadata", "c_ref_pf",         R.c_ref_pf);
        writer.write_attribute<double>("/metadata", "c_cable_pf",       R.c_cable_pf);
        writer.write_attribute<double>("/metadata", "c_routing_pf",     R.c_routing_pf);
        writer.write_attribute<double>("/metadata", "enc_slope_per_pf", cap_ptr->slope());
        writer.write_attribute_string ("/metadata", "pad_capacitance_file", R.pad_capacitance_file);
        writer.write_attribute_string ("/metadata", "noise_model_description",
            "ENC(C)^2 = enc0^2 + (b C)^2, b = enc_slope_per_pf ; seul le bruit serie depend de C ; enc_total, enc_par, enc_ser, enc_can, "
            "sigma_par, sigma_ser, ghost_p et read_thr_e sont ceux du pad de capacite de reference c_ref_pf ; capacite et ENC de chaque pad : "
            "/pads/c_pF et /pads/enc ; seuil de lecture global = read_thr_sigma * enc_ref");
        if (ghosts_on) {
            writer.write_attribute<double>("/metadata", "ghost_p_mean", ghost_p_mean);
            writer.write_attribute<double>("/metadata", "ghost_p_min",  ghost_p_min);
            writer.write_attribute<double>("/metadata", "ghost_p_max",  ghost_p_max);
        }
    }
    writer.write_attribute<double>("/metadata", "f_par",            R.f_par);
    writer.write_attribute<double>("/metadata", "f_ser",            R.f_ser);
    writer.write_attribute<double>("/metadata", "f_can",            R.f_can);
    writer.write_attribute<double>("/metadata", "enc_par",          noise.enc_par());
    writer.write_attribute<double>("/metadata", "enc_ser",          noise.enc_ser());
    writer.write_attribute<double>("/metadata", "enc_can",          noise.enc_can());
    writer.write_attribute<double>("/metadata", "sigma_par",        noise.sigma_par());
    writer.write_attribute<double>("/metadata", "sigma_ser",        noise.sigma_ser());
    writer.write_attribute<double>("/metadata", "adc_period_ns",    R.adc_period_ns);
    writer.write_attribute<int>   ("/metadata", "adc_ratio",        noise.params().adc_ratio);
    writer.write_attribute<double>("/metadata", "read_thr_sigma",   R.read_thr_sigma);
    writer.write_attribute<double>("/metadata", "read_thr_e",       readout.threshold());
    writer.write_attribute<int>   ("/metadata", "read_n_consec",    R.read_n_consec);
    writer.write_attribute<double>("/metadata", "window_margin_ns", R.window_margin_ns);
    writer.write_attribute<int>   ("/metadata", "noise_seed",       R.noise_seed);
    writer.write_attribute<int>   ("/metadata", "response_n_taps",  noise.n_taps());
    writer.write_attribute<int>   ("/metadata", "window_tail_bins", readout.tail_bins());
    writer.write_attribute<int>   ("/metadata", "readout_grid_bins", readout.grid_bins());
    writer.write_attribute<int>   ("/metadata", "window_margin_bins", readout.margin_bins());
    writer.write_attribute<int>   ("/metadata", "ghost_enable",     ghosts_on ? 1 : 0);
    writer.write_attribute<double>("/metadata", "ghost_p",          ghosts_on ? ghost_ptr->prob() : 0.0);
    writer.write_attribute<double>("/metadata", "ghost_p_err",      ghosts_on ? ghost_ptr->prob_err() : 0.0);
    writer.write_attribute<int>   ("/metadata", "ghost_window_samples", ghosts_on ? ghost_ptr->window_samples() : 0);
    writer.write_attribute<int>   ("/metadata", "ghost_calib_trials", R.ghost_calib_trials);
    writer.write_attribute_string ("/metadata", "P_description",
        "liste unique de pads par evenement (groupe /P) ; pflag : 1 = H seul, 2 = L seul, 3 = H et L, 0 = fantome ; "
        "pqn : echantillons en e- equivalents ; temps de l'echantillon j (compte depuis le debut de la fenetre complete) : "
        "t = (phase + j * adc_ratio) * delta_t ns ; pk0 = indice j du premier echantillon ecrit ; pnt = nombre d'echantillons ecrits ; "
        "pq = charge vraie du pad (H+L) ; un fantome (pflag 0) a pq = 0, une fenetre de longueur fixe centree sur son premier "
        "franchissement du seuil, et n_ghost = leur nombre dans l'evenement ; "
        "la position (x, y) du centre d'un pad n'est pas ecrite par pad : la lire dans /pads (id, x, y) a partir de pid");
    writer.write_attribute_string ("/metadata", "pads_description",
        std::string("groupe /pads : un element par pad du detecteur (meme contenu que my_pad.txt, en double) : id, x et y (centre, cm), "
        "size (cote du pad carre, cm ; 0 pour un anneau), r_in et r_out (anneaux, cm), sx et sy (quadrant, +1 ou -1) ; "
        "retrouver un pad par son id (pas forcement egal a l'indice)") +
        std::string(cap_ptr ? " ; modele capacitance : c_pF = capacite totale d'entree du pad (pF), enc = ENC du pad (e-)" : ""));
    writer.write_attribute_string ("/metadata", "vertex_description",
        "groupe /vertex (x, y, z, un par evenement, en cm) : vertex de fission, commun a H et L");
    writer.write_attribute_string ("/metadata", "trace_xyz_description",
        "les points (x, y, z) de la trace de H ou L ne sont pas ecrits ; pour i = 0 .. len(dE)-1 : s = step_size_eff * (i + 0.5) ; "
        "ux = sin(theta) cos(phi) ; uy = cos(theta) si emission_axis = Y, sinon sin(theta) sin(phi) ; "
        "uz = cos(theta) si emission_axis = Z, sinon sin(theta) sin(phi) ; "
        "x = vertex_x + s ux ; y = s uy ; z = vertex_z + s uz (theta, phi du groupe /H ou /L)");
    
    for (const std::string &frag : {"/H", "/L"}) {
      writer.add_dataset<float>     (frag + "/KE");
      writer.add_dataset<float>     (frag + "/theta");
      writer.add_dataset<float>     (frag + "/phi");
      writer.add_vlen_dataset<float>(frag + "/dE");
    }
    // vertex de fission, commun a H et L (voir vertex_description)
    writer.add_dataset<float>("/vertex/x");
    writer.add_dataset<float>("/vertex/y");
    writer.add_dataset<float>("/vertex/z");

    // geometrie de tous les pads du detecteur (meme contenu que my_pad.txt, en double) : le fichier est autonome
    {
        const std::vector<Pad>& pl = plane.pads();
        const hsize_t np = (hsize_t)pl.size();
        writer.add_dataset<int32_t>("/pads/id", np);
        writer.add_dataset<double> ("/pads/x", np);
        writer.add_dataset<double> ("/pads/y", np);
        writer.add_dataset<double> ("/pads/size", np);
        writer.add_dataset<double> ("/pads/r_in", np);
        writer.add_dataset<double> ("/pads/r_out", np);
        writer.add_dataset<int32_t>("/pads/sx", np);
        writer.add_dataset<int32_t>("/pads/sy", np);
        if (cap_ptr) {
            writer.add_dataset<double>("/pads/c_pF", np);     // capacite totale d'entree du pad (pF)
            writer.add_dataset<double>("/pads/enc", np);      // ENC du pad (e-)
        }
        std::vector<int32_t> g_id(np), g_sx(np), g_sy(np);
        std::vector<double>  g_x(np), g_y(np), g_size(np), g_rin(np), g_rout(np);
        for (size_t ip = 0; ip < pl.size(); ++ip) {
            g_id[ip] = (int32_t)pl[ip].id;   g_sx[ip] = (int32_t)pl[ip].sx;   g_sy[ip] = (int32_t)pl[ip].sy;
            g_x[ip] = pl[ip].x;              g_y[ip] = pl[ip].y;              g_size[ip] = pl[ip].size;
            g_rin[ip] = pl[ip].r_in;         g_rout[ip] = pl[ip].r_out;
        }
        writer.write_all<int32_t>("/pads/id",    g_id);
        writer.write_all<double> ("/pads/x",     g_x);
        writer.write_all<double> ("/pads/y",     g_y);
        writer.write_all<double> ("/pads/size",  g_size);
        writer.write_all<double> ("/pads/r_in",  g_rin);
        writer.write_all<double> ("/pads/r_out", g_rout);
        writer.write_all<int32_t>("/pads/sx",    g_sx);
        writer.write_all<int32_t>("/pads/sy",    g_sy);
        if (cap_ptr) {
            writer.write_all<double>("/pads/c_pF", pad_c_pf);
            writer.write_all<double>("/pads/enc",  pad_enc);
        }
    }
    
    // liste unique de pads (voir readout.hpp) et etiquette d'evenement (indice dans le fichier d'input)
    writer.add_dataset<int32_t>("/event_id");
    writer.add_dataset<int32_t>("/P/phase");
    writer.add_dataset<int32_t>("/P/n_ghost");
    writer.add_vlen_dataset<int32_t>("/P/pid");
    writer.add_vlen_dataset<float>  ("/P/pq");
    writer.add_vlen_dataset<int32_t>("/P/pflag");
    writer.add_vlen_dataset<int32_t>("/P/pk0");
    writer.add_vlen_dataset<int32_t>("/P/pnt");
    writer.add_vlen_dataset<float>  ("/P/pqn");

    std::vector<Processor> procs;
    procs.reserve(nthreads);
    for (int t = 0; t < nthreads; ++t)
        procs.emplace_back(cfg, base_seed + t);

    const int n_bins = cfg.n_bins() ;
    std::vector<std::vector<std::vector<double>>> qt_bufs(
        nthreads,
        std::vector<std::vector<double>>(plane.pads().size(),
                                         std::vector<double>(n_bins, 0.0)));

    long n_cathode_total = 0;      // traces (fragments) qui depassent la cathode (z < 0)
    long n_grid_total    = 0;      // traces (fragments) coupees a la grille (z >= dkg)

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
        std::vector<float> V_x(count), V_y(count), V_z(count);
        std::vector<float> L_KEf(count);
        std::vector<float> L_theta(count), L_phi(count);

        std::vector<std::vector<float>> H_dE(count);
        std::vector<int32_t> P_event_id(count), P_phase(count), P_nghost(count);
        std::vector<std::vector<float>> P_pq(count), P_pqn(count);
        std::vector<std::vector<int32_t>> P_pid(count), P_pflag(count), P_pk0(count), P_pnt(count);
	
        std::vector<std::vector<float>> L_dE(count);

        std::vector<std::vector<PadCharge>> H_pads(count), L_pads(count);

        long n_cathode = 0, n_grid = 0;
#pragma omp parallel for num_threads(nthreads) schedule(dynamic, 16) reduction(+:n_cathode, n_grid)
        for (hsize_t i = 0; i < count; ++i) {
          int tid = omp_get_thread_num();
	  procs[tid].reseed(mix_seed(base_seed, start + i));

            EventInput in;
            in.H = { merge_trace(H_trace[i], n_merge), 0, 0, 0 };
            in.L = { merge_trace(L_trace[i], n_merge), 0, 0, 0 };

            EventOutput out;
            procs[tid].process(in, out);

            // trace hors du volume de derive : cathode (z < 0) ou grille (la trace est alors coupee a z = dkg)
            if (frag_mask & 1) {
                if (out.H.z.size() < in.H.raw_trace.size()) ++n_grid;
                for (size_t k = 0; k < out.H.z.size(); ++k) if (out.H.z[k] < 0.0) { ++n_cathode; break; }
            }
            if (frag_mask & 2) {
                if (out.L.z.size() < in.L.raw_trace.size()) ++n_grid;
                for (size_t k = 0; k < out.L.z.size(); ++k) if (out.L.z[k] < 0.0) { ++n_cathode; break; }
            }
	    
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
            // le vertex est commun a H et L (garantie du Processor) ; on le verifie avant de n'en ecrire qu'un
            if (out.H.vx != out.L.vx || out.H.vy != out.L.vy || out.H.vz != out.L.vz) {
#pragma omp critical
                { std::cerr << "Erreur : vertex different pour H et L, evenement " << start + i << "\n"; }
                std::exit(1);
            }
            V_x[i] = (float)out.H.vx; V_y[i] = (float)out.H.vy; V_z[i] = (float)out.H.vz;
            H_dE[i] = to_float(out.H.dE);
            

	    L_KEf[i] = (float)L_KE[i];
            L_theta[i] = (float)out.L.dir.theta; L_phi[i] = (float)out.L.dir.phi;
            L_dE[i] = to_float(out.L.dE);

            // liste unique de pads : mise en forme, bruit, CAN, seuil de lecture (voir readout.hpp)
            ReadoutEvent rev;
            try {
                readout.process(H_pads[i], L_pads[i], mix_seed(noise_seed, start + i), rev);
            } catch (const std::exception& e) {
#pragma omp critical
                { std::cerr << "Erreur de lecture, evenement " << start + i << " : " << e.what() << "\n"; }
                std::exit(1);
            }
            P_event_id[i] = (int32_t)(start + i);
            P_phase[i]    = (int32_t)rev.phase;
            P_nghost[i]   = (int32_t)rev.n_ghost;
            for (size_t ip = 0; ip < rev.pads.size(); ++ip) {
                const ReadoutPad& rp = rev.pads[ip];
                P_pid[i].push_back((int32_t)rp.pad_id);
                P_pq[i].push_back((float)rp.q);
                P_pflag[i].push_back((int32_t)rp.flag);
                P_pk0[i].push_back((int32_t)rp.j0);
                P_pnt[i].push_back((int32_t)rp.samples.size());
                for (size_t js = 0; js < rp.samples.size(); ++js) P_pqn[i].push_back((float)rp.samples[js]);
            }
        }

        auto t2 = std::chrono::steady_clock::now();

        n_cathode_total += n_cathode;
        n_grid_total    += n_grid;
        if (n_cathode > 0 || n_grid > 0) {
            std::cout << "\033[1;31mATTENTION (evenements " << start << " a " << start + count - 1 << ") : "
                      << n_cathode << " trace(s) depassent la cathode (z < 0 ; voxels conserves, d limite a dkg dans la table), "
                      << n_grid << " trace(s) coupee(s) a la grille (z >= dkg ; voxels suivants ignores)\033[0m\n";
        }

	writer.write_chunk<float>("/H/KE", H_KEf, start);
        writer.write_chunk<float> ("/H/theta", H_theta, start);
        writer.write_chunk<float> ("/H/phi",   H_phi,   start);
        writer.write_chunk<float> ("/vertex/x", V_x, start);
        writer.write_chunk<float> ("/vertex/y", V_y, start);
        writer.write_chunk<float> ("/vertex/z", V_z, start);

        timed_vlen<float> (writer, "/H/dE",  H_dE,  start);

	writer.write_chunk<float>("/L/KE", L_KEf, start);        
        writer.write_chunk<float> ("/L/theta", L_theta, start);
        writer.write_chunk<float> ("/L/phi",   L_phi,   start);
        timed_vlen<float>(writer, "/L/dE",  L_dE,  start);
	
        writer.write_chunk<int32_t>("/event_id",   P_event_id, start);
        writer.write_chunk<int32_t>("/P/phase",    P_phase,    start);
        writer.write_chunk<int32_t>("/P/n_ghost",  P_nghost,   start);
        timed_vlen<int32_t>(writer, "/P/pid",   P_pid,   start);
        timed_vlen<float>  (writer, "/P/pq",    P_pq,    start);
        timed_vlen<int32_t>(writer, "/P/pflag", P_pflag, start);
        timed_vlen<int32_t>(writer, "/P/pk0",   P_pk0,   start);
        timed_vlen<int32_t>(writer, "/P/pnt",   P_pnt,   start);
        timed_vlen<float>  (writer, "/P/pqn",   P_pqn,   start);

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

    if (n_cathode_total > 0 || n_grid_total > 0) {
        std::cout << "\033[1;31mATTENTION : " << n_cathode_total << " trace(s) ont depasse la cathode et "
                  << n_grid_total << " ont ete coupees a la grille sur l'ensemble du run. "
                  << "Verifier dkg et la position du vertex.\033[0m\n";
    }

    std::cout << "\nTemps réel total : " << omp_get_wtime() - wall0 << " s\n";
    for (std::map<std::string,double>::const_iterator it = g_wt.begin(); it != g_wt.end(); ++it)
      std::cout << "write " << it->first << " : " << it->second << " s\n";
    
    std::cout << "\nTerminé → " << out_path << "\n";
    return 0;
}
