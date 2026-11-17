// ---------------------------------------------------------------------------
// profile_basis : reponse EXACTE de la chaine (induction + preampli + CAN, sans bruit) a des fonctions
// de base du profil de charge le long des deux demi-droites d'un evenement.
//
// Pour chaque evenement, on connait (estime ou vrai) : vertex, direction de H, direction de L.
// Le profil inconnu q(l) (electrons par cm, l = longueur parcourue depuis le vertex) est decrit par des
// fonctions "chapeau" de pas Delta, centrees en l_m = m * Delta, m = 0 .. M-1, M = ceil(L / Delta) + 1.
// Pour CHAQUE chapeau, ce programme calcule le signal qu'il produirait sur les echantillons demandes,
// avec exactement la meme chaine que ./process :
//     voxels (n_e) -> PadPlane::project_pair (table d'induction) -> h (*) qt -> echantillons CAN
//     d'indice phase + j * adc_ratio.
// Le signal total est lineaire en les coefficients des chapeaux : signal = somme_m c_m * reponse_m.
//
// Usage :   ./profile_basis drift.cfg requete.txt reponse.bin [cle=valeur ...]
//   drift.cfg : le MEME fichier que pour ./process (seules les cles utiles sont lues, les autres sont ignorees)
//
// Mode serveur (pour les ajustements iteratifs : la table n'est construite qu'une fois) :
//           ./profile_basis drift.cfg --serve [cle=valeur ...]
//   Apres la construction de la table, ecrit la ligne  @@READY  ; puis lit sur l'entree standard des lignes
//   "requete.txt reponse.bin" (une requete par ligne), traite chacune et ecrit la ligne  @@DONE  (ou
//   @@ERROR message). La ligne "quit" ou la fin de l'entree arrete le programme.
//
// Fichier de requete (texte, mots separes par des blancs) :
//   step_eff  delta  n_events                       step_eff : pas des voxels (cm) = /metadata step_size_eff
//                                                   delta    : pas des chapeaux (cm)
//   puis, pour chaque evenement :
//     event_id  phase                               phase = /P/phase
//     vx vy vz                                      vertex (cm)
//     uxH uyH uzH  LH                               direction unitaire de H, longueur modelisee (cm)
//     uxL uyL uzL  LL                               idem pour L
//     n_pads                                        nombre de pads demandes
//     pad_id j0 nt          (n_pads lignes)         echantillons j0 .. j0+nt-1 du pad (indices CAN, comme /P/pk0)
//
// Fichier de sortie (binaire, petit-boutiste de la machine), pour chaque evenement, dans l'ordre :
//   int32 event_id, int32 M_H, int32 M_L, int32 n_rows
//   double reponse[(M_H + M_L) * n_rows]       ligne b = chapeau b (H d'abord, puis L), colonne r = echantillon
//                                              demande (pads dans l'ordre de la requete, puis j croissant)
//   Unite : electrons equivalents par (electron/cm) de coefficient.
//
// Les voxels sont aux positions s = step_eff * (i + 0.5), comme dans /metadata trace_xyz_description, et un
// voxel qui atteint la grille (z >= dkg) arrete la trace, comme dans ./process.
// Pas de seuil de lecture ni de coupure eps_qt : tous les pads a portee de la trace sont calcules.
//
// C++ classique.
// ---------------------------------------------------------------------------

#include "detector.hpp"
#include "pad_geometry.hpp"
#include "config_file.hpp"
#include "derived_params.hpp"
#include "noise.hpp"
#include "preamp_response.hpp"

#include <omp.h>
#include <stdint.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct PadWin {
    int pad_id;
    int j0;
    int nt;
};

struct EventReq {
    int    id;
    int    phase;
    double vx, vy, vz;
    double ux[2], uy[2], uz[2], len[2];     // 0 = H, 1 = L
    std::vector<PadWin> rows;
};

struct EventRes {
    int M[2];
    int n_rows;
    std::vector<double> data;               // (M[0] + M[1]) * n_rows
};

// ---------------------------------------------------------------------------
// Lecture de la requete
// ---------------------------------------------------------------------------
static void read_request(const std::string& path, double& step_eff, double& delta, std::vector<EventReq>& evs)
{
    std::ifstream in(path.c_str());
    if (!in) throw std::runtime_error("requete illisible : " + path);
    int n = 0;
    if (!(in >> step_eff >> delta >> n)) throw std::runtime_error("requete : entete step_eff delta n_events illisible");
    if (step_eff <= 0.0 || delta <= 0.0 || n < 0) throw std::runtime_error("requete : step_eff, delta > 0 et n_events >= 0 requis");
    evs.resize(n);
    for (int e = 0; e < n; ++e) {
        EventReq& r = evs[e];
        if (!(in >> r.id >> r.phase >> r.vx >> r.vy >> r.vz))
            throw std::runtime_error("requete : entete d'evenement illisible");
        for (int f = 0; f < 2; ++f)
            if (!(in >> r.ux[f] >> r.uy[f] >> r.uz[f] >> r.len[f]))
                throw std::runtime_error("requete : direction ou longueur illisible");
        int np = 0;
        if (!(in >> np) || np < 0) throw std::runtime_error("requete : n_pads illisible");
        r.rows.resize(np);
        for (int k = 0; k < np; ++k)
            if (!(in >> r.rows[k].pad_id >> r.rows[k].j0 >> r.rows[k].nt))
                throw std::runtime_error("requete : ligne de pad illisible");
    }
}

static int n_nodes(double len, double delta)
{
    int M = (int)std::ceil(len / delta - 1.0e-9) + 1;
    return (M < 1) ? 1 : M;
}

// ---------------------------------------------------------------------------
// Voxels du chapeau m de la demi-droite f : n_e = w(s) * step_eff, w = 1 - |s - m delta| / delta
// ---------------------------------------------------------------------------
static void build_voxels(const EventReq& ev, int f, int m, double delta, double step,
                         const DetectorConfig& cfg,
                         std::vector<double>& x, std::vector<double>& y, std::vector<double>& z,
                         std::vector<double>& ne, std::vector<double>& sg)
{
    x.clear(); y.clear(); z.clear(); ne.clear(); sg.clear();
    const double lm = m * delta;
    int i_lo = (int)std::ceil((lm - delta) / step - 0.5);
    int i_hi = (int)std::floor((lm + delta) / step - 0.5);
    if (i_lo < 0) i_lo = 0;
    for (int i = i_lo; i <= i_hi; ++i) {
        const double s = step * (i + 0.5);
        const double zz = ev.vz + s * ev.uz[f];
        if (zz >= cfg.dkg) break;                       // la trace est coupee a la grille, comme dans ./process
        const double w = 1.0 - std::fabs(s - lm) / delta;
        if (w <= 0.0) continue;
        x.push_back(ev.vx + s * ev.ux[f]);
        y.push_back(ev.vy + s * ev.uy[f]);
        z.push_back(zz);
        ne.push_back(w * step);
        double st = cfg.sigma_T(zz);                    // ne sert qu'a l'avertissement de bord de project_pair
        if (!(st > 1.0e-3)) st = 1.0e-3;
        sg.push_back(st);
    }
}

// ---------------------------------------------------------------------------
// Traitement d'un fichier de requete : ecrit le fichier de reponse. Renvoie 0 si tout va bien.
// ---------------------------------------------------------------------------
static int run_batch(const std::string& req_path, const std::string& out_path,
                     const DetectorConfig& cfg, const PadPlane& plane, const NoiseModel& noise,
                     int adc_ratio, int n_threads)
{
    double step_eff = 0.0, delta = 0.0;
    std::vector<EventReq> evs;
    try {
        read_request(req_path, step_eff, delta, evs);
    } catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << "\n";
        return 1;
    }

    const int n_bins = cfg.n_bins();
    const int grid   = n_bins + noise.n_taps();          // >= n_bins + n_taps - 1 : la convolution n'est jamais coupee
    const std::vector<double> empty_vec;

    std::cout << "requete : " << evs.size() << " evenements, step_eff = " << step_eff << " cm, delta = " << delta
              << " cm, n_threads = " << n_threads << std::endl;

    std::vector<EventRes> res(evs.size());
    std::vector<std::vector<std::vector<double> > > qt_bufs(n_threads);
    bool failed = false;
    double t0 = omp_get_wtime();

#pragma omp parallel for num_threads(n_threads) schedule(dynamic, 1)
    for (int e = 0; e < (int)evs.size(); ++e) {
        if (failed) continue;
        try {
            const int tid = omp_get_thread_num();
            if (qt_bufs[tid].empty())
                qt_bufs[tid].assign(plane.pads().size(), std::vector<double>(n_bins, 0.0));
            const EventReq& ev = evs[e];
            EventRes& rs = res[e];

            // lignes demandees : decalage de chaque pad dans les colonnes
            std::map<int, int> row0, nt_of, j0_of;       // pad_id -> premiere colonne, nombre d'echantillons, j0
            int n_rows = 0;
            for (size_t k = 0; k < ev.rows.size(); ++k) {
                row0[ev.rows[k].pad_id] = n_rows;
                nt_of[ev.rows[k].pad_id] = ev.rows[k].nt;
                j0_of[ev.rows[k].pad_id] = ev.rows[k].j0;
                n_rows += ev.rows[k].nt;
            }

            rs.M[0] = n_nodes(ev.len[0], delta);
            rs.M[1] = n_nodes(ev.len[1], delta);
            rs.n_rows = n_rows;
            rs.data.assign((size_t)(rs.M[0] + rs.M[1]) * (size_t)n_rows, 0.0);

            std::vector<double> x, y, z, ne, sg, sig, shaped;
            std::vector<PadCharge> resA, resB;
            int b = 0;
            for (int f = 0; f < 2; ++f) {
                for (int m = 0; m < rs.M[f]; ++m, ++b) {
                    build_voxels(ev, f, m, delta, step_eff, cfg, x, y, z, ne, sg);
                    if (x.empty()) continue;
                    FragTrace fa = { &x, &y, &z, &ne, &sg };
                    FragTrace fe = { &empty_vec, &empty_vec, &empty_vec, &empty_vec, &empty_vec };
                    plane.project_pair(fa, fe, cfg, qt_bufs[tid], resA, resB);
                    for (size_t p = 0; p < resA.size(); ++p) {
                        const PadCharge& pc = resA[p];
                        std::map<int, int>::const_iterator it = row0.find(pc.pad_id);
                        if (it == row0.end()) continue;          // pad non demande
                        const int nq = (int)pc.qt.size();
                        if (nq == 0) continue;
                        sig.assign(grid, 0.0);
                        for (int k = 0; k < nq; ++k) sig[pc.k0 + k] = pc.qt[k];
                        noise.shape(sig, grid, shaped);
                        const int off = it->second;
                        const int nt  = nt_of[pc.pad_id];
                        const int j0  = j0_of[pc.pad_id];
                        double* out = &rs.data[(size_t)b * (size_t)n_rows + off];
                        for (int jj = 0; jj < nt; ++jj) {
                            const long idx = (long)ev.phase + (long)(j0 + jj) * adc_ratio;   // bin de dt
                            if (idx >= 0 && idx < grid) out[jj] = shaped[idx];
                        }
                    }
                }
            }
        } catch (const std::exception& ex) {
#pragma omp critical
            { std::cerr << "Erreur, evenement " << evs[e].id << " : " << ex.what() << "\n"; failed = true; }
        }
    }
    if (failed) return 1;
    std::cout << "calcul : " << omp_get_wtime() - t0 << " s" << std::endl;

    std::ofstream out(out_path.c_str(), std::ios::binary);
    if (!out) { std::cerr << "Erreur : fichier de sortie illisible : " << out_path << "\n"; return 1; }
    for (size_t e = 0; e < evs.size(); ++e) {
        int32_t hdr[4];
        hdr[0] = (int32_t)evs[e].id;
        hdr[1] = (int32_t)res[e].M[0];
        hdr[2] = (int32_t)res[e].M[1];
        hdr[3] = (int32_t)res[e].n_rows;
        out.write((const char*)hdr, sizeof(hdr));
        if (!res[e].data.empty())
            out.write((const char*)&res[e].data[0], (std::streamsize)(res[e].data.size() * sizeof(double)));
    }
    out.close();
    std::cout << "ecrit : " << out_path << std::endl;
    return 0;
}

// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    const bool serve = (argc >= 3 && std::string(argv[2]) == "--serve");
    if (argc < 3 || (!serve && argc < 4)) {
        std::cerr << "Usage : " << argv[0] << " drift.cfg requete.txt reponse.bin [cle=valeur ...]\n"
                  << "        " << argv[0] << " drift.cfg --serve [cle=valeur ...]\n";
        return 1;
    }
    const int first_override = serve ? 3 : 4;

    // --- cles utiles du fichier .cfg (memes noms et memes defauts que main.cpp) ---
    ConfigFile     conf;
    DetectorConfig cfg;
    PadPlaneParams pParam;
    TableOptions   topt;
    int         n_threads = 8;
    std::string preamp_response;
    double      gauss_fwhm = 190.0, minos_peak = 0.0, adc_period_ns = 20.0;
    try {
        conf.load(argv[1]);
        for (int a = first_override; a < argc; ++a) conf.set_override(argv[a]);

        n_threads = conf.get_int("n_threads", 8);
        preamp_response = conf.get_string("preamp_response", "test_gauss");
        gauss_fwhm      = conf.get_double("preamp_gauss_fwhm", 190.0);
        minos_peak      = conf.get_double("preamp_minos_peak_ns", 0.0);
        adc_period_ns   = conf.get_double("adc_period_ns", 20.0);

        cfg.dkg          = conf.get_double("dkg",          cfg.dkg);
        cfg.dga          = conf.get_double("dga",          cfg.dga);
        cfg.vertex_r_max = conf.get_double("vertex_r_max", cfg.vertex_r_max);
        cfg.Fano         = conf.get_double("Fano",         cfg.Fano);
        cfg.W            = conf.get_double("W",            cfg.W);
        cfg.DT_kg        = conf.get_double("DT_kg",        cfg.DT_kg);
        cfg.DT_ga        = conf.get_double("DT_ga",        cfg.DT_ga);
        cfg.DL_kg        = conf.get_double("DL_kg",        cfg.DL_kg);
        cfg.DL_ga        = conf.get_double("DL_ga",        cfg.DL_ga);
        cfg.ve_kg        = conf.get_double("ve_kg",        cfg.ve_kg);
        cfg.ve_ga        = conf.get_double("ve_ga",        cfg.ve_ga);
        cfg.delta_t      = conf.get_double("delta_t",      cfg.delta_t);

        pParam.pad_size     = conf.get_double("pad_size",     pParam.pad_size);
        pParam.inter_pad    = conf.get_double("inter_pad",    pParam.inter_pad);
        pParam.r_quinconce  = conf.get_double("r_quinconce",  pParam.r_quinconce);
        pParam.r_transition = conf.get_double("r_transition", pParam.r_transition);
        pParam.r_active     = conf.get_double("r_active",     pParam.r_active);
        pParam.n_rings      = conf.get_int   ("n_rings",      pParam.n_rings);
        pParam.y_offset     = conf.get_double("y_offset",     pParam.y_offset);
        pParam.reach_pitch  = conf.get_double("reach_pitch",  pParam.reach_pitch);
        pParam.bound_margin = conf.get_double("bound_margin", pParam.bound_margin);

        topt.n_d          = conf.get_int   ("table_n_d",      20);
        topt.n_img        = conf.get_int   ("table_n_img",    20);
        topt.m_cells      = conf.get_int   ("table_m_cells",  0);
        topt.step_out     = conf.get_double("table_step_out", 0.0);
        topt.r_src        = conf.get_double("table_r_src",    0.0);
        topt.kappa        = conf.get_double("table_kappa",    3.0);
        topt.step_cap     = conf.get_double("table_step_cap", 0.0);
        topt.acc_tol      = conf.get_double("table_acc_tol",  1.0e-3);
        topt.reach_tol    = conf.get_double("reach_tol",      1.0e-5);
        topt.max_table_mb = conf.get_double("max_table_mb",   500.0);
        topt.R            = conf.get_int   ("table_R",        0);
        topt.tau_tol      = conf.get_double("table_tau_tol",  0.005);
        topt.tau_dmin     = conf.get_double("table_tau_dmin", 2.0);
        topt.tau_Rref     = conf.get_int   ("table_tau_Rref", 12);
        topt.ignore       = conf.get_string("ignore",         "");

        if (preamp_response != "test_gauss" && preamp_response != "minos")
            throw std::runtime_error("preamp_response doit valoir test_gauss ou minos");
        if (preamp_response == "minos" &&
            (minos_peak < MinosResponse::peak_min_ns() || minos_peak > MinosResponse::peak_max_ns()))
            throw std::runtime_error("preamp_minos_peak_ns hors de la plage autorisee");
        if (n_threads < 1) throw std::runtime_error("n_threads doit etre >= 1");
    } catch (const std::exception& e) {
        std::cerr << "Erreur de configuration : " << e.what() << "\n";
        return 1;
    }

    // --- parametres derives, table d'induction (comme main.cpp) ---
    CheckList     checks;
    DerivedParams D;
    try {
        checks.set_ignored(topt.ignore);
        topt.R = resolve_table_R(cfg, pParam, topt, checks);
        D = derive_params(cfg, pParam, topt, checks);
        checks.finish();
    } catch (const std::exception& e) {
        std::cerr << "\nArret : " << e.what() << "\n";
        return 1;
    }
    cfg.tau_tail_ns = D.tau_tail_ns;

    // pas de seuil de lecture : tous les pads touches sont rendus, signal exact (eps_qt = 0)
    cfg.seuil_qt_pos = -1.0e300;
    cfg.seuil_qt_neg = -1.0e300;
    cfg.eps_qt       = 0.0;

    // --- reponse du preampli et mise en forme (meme code que main.cpp) ---
    GaussTestResponse gauss_resp(gauss_fwhm);
    const PreampResponse* response = &gauss_resp;
    MinosResponse* minos_ptr = 0;
    if (preamp_response == "minos") {
        minos_ptr = new MinosResponse(minos_peak);
        response  = minos_ptr;
    }
    NoiseModel* noise_ptr = 0;
    int adc_ratio = 1;
    try {
        adc_ratio = (int)std::floor(adc_period_ns / cfg.delta_t + 0.5);
        if (adc_ratio < 1 || std::fabs(adc_ratio * cfg.delta_t - adc_period_ns) > 1.0e-6 * cfg.delta_t)
            throw std::runtime_error("adc_period_ns doit etre un multiple entier de delta_t");
        NoiseParams nparams;                 // le bruit n'est pas utilise ici : seule la mise en forme (shape) l'est
        nparams.dt = cfg.delta_t;
        nparams.adc_ratio = adc_ratio;
        std::vector<double> h_resp;
        response->sample(cfg.delta_t, h_resp);
        noise_ptr = new NoiseModel(nparams, h_resp);
    } catch (const std::exception& e) {
        std::cerr << "\nArret (reponse) : " << e.what() << "\n";
        return 1;
    }
    const NoiseModel& noise = *noise_ptr;
    std::cout << "reponse : " << response->name() << " : " << response->describe() << "\n";

    PadPlane plane(pParam, cfg, D.tp, D.reach_cm);

    if (!serve)
        return run_batch(argv[2], argv[3], cfg, plane, noise, adc_ratio, n_threads);

    // --- mode serveur ---
    std::cout << "@@READY" << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream ls(line);
        std::string rq, ou;
        if (!(ls >> rq)) continue;
        if (rq == "quit") break;
        if (!(ls >> ou)) { std::cout << "@@ERROR ligne de commande incomplete" << std::endl; continue; }
        const int rc = run_batch(rq, ou, cfg, plane, noise, adc_ratio, n_threads);
        if (rc == 0) std::cout << "@@DONE" << std::endl;
        else         std::cout << "@@ERROR voir la sortie d'erreur" << std::endl;
    }
    return 0;
}
