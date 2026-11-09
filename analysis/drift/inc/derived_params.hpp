#pragma once

// ---------------------------------------------------------------------------
// Parametres derives et controles de coherence.
//
//   derive_params(cfg, pp, opt, chk) calcule ce qui etait ecrit en dur (n_h, tau_max,
//   n_tau, n_bins, step_out, n_out, r_src, m_cells) et verifie les contraintes.
//
//   Un controle en echec est enregistre dans la CheckList avec un identifiant.
//   chk.finish() affiche tous les echecs puis ARRETE le programme, sauf pour les
//   controles listes dans   ignore = id1,id2   (fichier d'input ou ligne de commande).
//
// C++ classique, header seul.
// ---------------------------------------------------------------------------

#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include "detector.hpp"
#include "pad_geometry.hpp"
#include "psi_table.hpp"

// Options de la table lues dans le fichier d'input (0 = automatique)
struct TableOptions {
    int         n_d;           // noeuds en d
    int         n_img;         // images de la serie du potentiel de ponderation
    int         m_cells;       // cellules par demi-pad de la source (0 = auto)
    double      step_out;      // pas en u et v, cm (0 = auto)
    double      r_src;         // rayon de la source, cm (0 = auto)
    double      max_table_mb;  // plafond de memoire de la table
    int         R;             // pas de table = delta_t / R ; 0 = automatique parmi 1..6 (voir resolve_table_R)
    double      tau_tol;       // erreur relative toleree sur le signal (interpolation en tau) ; 0 = pas de controle
    double      tau_dmin;      // distance de derive minimale (cm) des voxels pour ce controle
    int         tau_Rref;      // R de la table de reference du controle (>= 8 pour le choix automatique)
    std::string ignore;        // identifiants de controles a ignorer, separes par des virgules

    TableOptions()
        : n_d(10), n_img(10), m_cells(0), step_out(0.0), r_src(0.0), max_table_mb(500.0), R(0), tau_tol(0.005), tau_dmin(2.0), tau_Rref(12) {}
};

struct DerivedParams {
    double pitch, a;
    double T_gap;          // temps de collecte dans le gap (ns)
    double dtau;           // pas de la table en tau (ns) = delta_t / R
    int    R;              // pas de table par bin de lecture
    double T_frac_ns;      // duree du dernier pas, partiel (ns) : T_gap - (n_h - 1) * dtau
    double tau_max_ns;
    double sigma_t_max;    // etalement temporel maximal (ns)
    int    n_h, m_min, m_max, n_tau;
    double reach_cm;
    double table_mb;
    double tau_tail_ns;    // queue du signal apres l'arrivee a la grille
    int    n_bins;
    TableParams tp;        // parametres passes a PsiTable
};

// ---------------------------------------------------------------------------
// Liste de controles
// ---------------------------------------------------------------------------

inline const std::vector<std::string>& check_ids() {
    static std::vector<std::string> ids;
    if (ids.empty()) {
                ids.push_back("n_h_small");
        ids.push_back("tau_resolution");
        ids.push_back("step_out_coarse");
        ids.push_back("reach_small");
        ids.push_back("table_memory");
        ids.push_back("ring_geometry");
        ids.push_back("phi_edge");
    }
    return ids;
}

class CheckList {
public:
    CheckList() : m_printed(0) {}

    // liste separee par des virgules ou des espaces ; un identifiant inconnu est une erreur
    void set_ignored(const std::string& list) {
        m_ignored.clear();
        std::string cur;
        for (size_t i = 0; i <= list.size(); ++i) {
            char ch = (i < list.size()) ? list[i] : ',';
            if (ch == ',' || ch == ' ' || ch == '\t') {
                if (!cur.empty()) { m_ignored.push_back(cur); cur.clear(); }
            } else {
                cur += ch;
            }
        }
        for (size_t i = 0; i < m_ignored.size(); ++i) {
            const std::vector<std::string>& ids = check_ids();
            if (std::find(ids.begin(), ids.end(), m_ignored[i]) == ids.end()) {
                std::string all;
                for (size_t k = 0; k < ids.size(); ++k) all += (k ? ", " : "") + ids[k];
                throw std::runtime_error("ignore : identifiant inconnu '" + m_ignored[i] +
                                         "'. Identifiants valides : " + all);
            }
        }
    }

    void fail(const std::string& id, const std::string& message) {
        Item it;
        it.id = id;
        it.msg = message;
        it.ignored = (std::find(m_ignored.begin(), m_ignored.end(), id) != m_ignored.end());
        m_items.push_back(it);
    }

    // Affiche les echecs apparus depuis le dernier appel ; leve une erreur si l'un d'eux
    // n'est pas ignore.
    void finish() {
        std::string blocking;
        for (size_t i = m_printed; i < m_items.size(); ++i) {
            const Item& it = m_items[i];
            std::cerr << (it.ignored ? "IGNORE  [" : "ERREUR  [") << it.id << "] " << it.msg << "\n";
            if (!it.ignored) blocking += (blocking.empty() ? "" : ",") + it.id;
        }
        m_printed = m_items.size();
        if (!blocking.empty())
            throw std::runtime_error("controle(s) en echec. Pour continuer malgre tout, ajouter  ignore = "
                                     + blocking + "  (fichier d'input ou argument ligne de commande)");
    }

    // pour les metadonnees de sortie
    std::string ignored_text() const {
        if (m_ignored.empty()) return "(aucun)";
        std::string s;
        for (size_t i = 0; i < m_ignored.size(); ++i) s += (i ? "," : "") + m_ignored[i];
        return s;
    }

private:
    struct Item {
        std::string id, msg;
        bool ignored;
    };
    std::vector<std::string> m_ignored;
    std::vector<Item>        m_items;
    size_t                   m_printed;
};

// ---------------------------------------------------------------------------
// derive_params
// ---------------------------------------------------------------------------

inline DerivedParams derive_params(const DetectorConfig& cfg, const PadPlaneParams& pp,
                                   const TableOptions& opt, CheckList& chk)
{
    DerivedParams D;
    const double g = cfg.dga;

    D.pitch = pp.pad_size + pp.inter_pad;
    D.a     = 0.5 * D.pitch;

    // --- pas de la table : dtau = delta_t / R, n_h = ceil(T_gap / dtau) pas dans le gap ---
    // La vitesse est la vraie : le dernier pas (le plus bas) n'est que partiel, voir PhiSource::height.
    D.T_gap = g / cfg.ve_ga;
    D.R     = opt.R;
    D.dtau  = cfg.delta_t / D.R;
    int n_h = (int)std::ceil(D.T_gap / D.dtau - 1.0e-9);
    if (n_h < 4) {
        std::ostringstream s;
        s << "le gap est decrit par moins de 4 pas de table (T_gap = " << D.T_gap << " ns, pas = "
          << D.dtau << " ns). Augmenter table_R.";
        chk.fail("n_h_small", s.str());
    }
    if (n_h < 2) n_h = 2;
    D.n_h       = n_h;
    D.T_frac_ns = D.T_gap - (n_h - 1) * D.dtau;

    // --- fin de la plage en tau (meme formule que l'ancien default_table_params) ---
    const double s_kg = cfg.DL_kg * std::sqrt(cfg.dkg) / cfg.ve_kg;
    const double s_ga = cfg.DL_ga * std::sqrt(g) / cfg.ve_ga;
    D.sigma_t_max = std::sqrt(s_kg * s_kg + s_ga * s_ga);
    D.tau_max_ns  = D.T_gap + 5.0 * D.sigma_t_max;

    // --- parametres de la table ---
    TableParams& tp = D.tp;
    tp.a     = D.a;
    tp.g     = g;
    tp.ve_kg = cfg.ve_kg;  tp.ve_ga = cfg.ve_ga;
    tp.DT_kg = cfg.DT_kg;  tp.DT_ga = cfg.DT_ga;
    tp.DL_kg = cfg.DL_kg;  tp.DL_ga = cfg.DL_ga;
    tp.d_max = cfg.dkg;
    tp.n_d   = opt.n_d;
    tp.n_img = opt.n_img;
    tp.n_h   = n_h;
    tp.R     = D.R;
    tp.dtau  = D.dtau;
    tp.tau_max_ns = D.tau_max_ns;

    // pas des noeuds en u, v : au plus pitch/16 et au plus g/15 (la structure de Psi varie sur ~g)
    const double step_auto = std::min(D.pitch / 16.0, g / 15.0);
    tp.step_out = (opt.step_out > 0.0) ? opt.step_out : step_auto;
    if (tp.step_out > g / 10.0) {
        std::ostringstream s;
        s << "table_step_out = " << tp.step_out << " cm est trop grossier pour dga = " << g
          << " cm (limite : dga/10 = " << g / 10.0 << " cm).";
        chk.fail("step_out_coarse", s.str());
    }

    // cellules de la source : pas de source a/m_cells <= g/15, au moins 8 cellules par demi-pad
    int mc_auto = (int)std::ceil(15.0 * D.a / g);
    if (mc_auto < 8) mc_auto = 8;
    tp.m_cells = (opt.m_cells > 0) ? opt.m_cells : mc_auto;

    // rayon de la source : phi decroit comme exp(-pi r / g) ; 6 g donne ~1e-8 (verifie apres coup : phi_edge)
    tp.r_src = (opt.r_src > 0.0) ? opt.r_src : std::sqrt(2.0) * D.a + 6.0 * g;

    // portee d'induction et taille de la table : le stencil cubique lit jusqu'au noeud i0 + 2
    D.reach_cm = pp.reach_pitch * D.pitch;
    tp.n_out   = (int)std::ceil(D.reach_cm / tp.step_out) + 3;

    table_time_range(tp, D.dtau, D.m_min, D.m_max);
    D.n_tau    = D.m_max - D.m_min + 1;
    D.table_mb = (double)tp.n_d * tp.n_out * tp.n_out * D.n_tau * 4.0 / 1.0e6;

    if (D.table_mb > opt.max_table_mb) {
        std::ostringstream s;
        s << "la table d'induction ferait ~" << D.table_mb << " Mo (n_out = " << tp.n_out << ", n_tau = "
          << D.n_tau << ", n_d = " << tp.n_d << "), plafond max_table_mb = " << opt.max_table_mb << ".";
        chk.fail("table_memory", s.str());
    }

    // --- portee : doit couvrir la tuile + le nuage d'electrons a la grille ---
    const double sigma_max = std::sqrt(cfg.DT_kg * cfg.DT_kg * cfg.dkg + cfg.DT_ga * cfg.DT_ga * g);
    const double reach_min = D.a + 4.0 * sigma_max;
    if (D.reach_cm < reach_min) {
        std::ostringstream s;
        s << "portee d'induction " << D.reach_cm << " cm (reach_pitch = " << pp.reach_pitch
          << ") < demi-pad + 4 sigma_transverse = " << reach_min << " cm (sigma = " << sigma_max
          << " cm a dkg). Charge perdue probable : augmenter reach_pitch (>= "
          << reach_min / D.pitch << ").";
        chk.fail("reach_small", s.str());
    }

    // --- geometrie des anneaux (n_rings = 0 : anode entierement en pads carres) ---
    if (pp.n_rings < 0 ||
        (pp.n_rings > 0 && !(pp.r_quinconce < pp.r_transition && pp.r_transition < pp.r_active))) {
        std::ostringstream s;
        s << "il faut r_quinconce < r_transition < r_active et n_rings >= 0 (ici " << pp.r_quinconce << " < "
          << pp.r_transition << " < " << pp.r_active << ", n_rings = " << pp.n_rings << ").";
        chk.fail("ring_geometry", s.str());
    }

    // --- lecture : le signal d'un voxel finit a t_grid + tau_max de la table ---
    D.tau_tail_ns = D.m_max * D.dtau + cfg.delta_t;
    D.n_bins = (int)std::ceil((cfg.dkg / cfg.ve_kg + D.tau_tail_ns) / cfg.delta_t) + 1;

    return D;
}

// ---------------------------------------------------------------------------
// Resolution en tau : erreur d'interpolation mesuree entre deux tables (pas dtau et dtau/2)
// ---------------------------------------------------------------------------

inline double tau_rnd(unsigned long& seed) {          // generateur congruentiel simple, dans [0, 1)
    seed = (seed * 1103515245UL + 12345UL) & 0x7fffffffUL;
    return (double)seed / 2147483648.0;
}

// Signaux de voxels isoles (charge unitaire) integres sur des bins de delta_t, calcules avec la
// table A (pas dtau) et la table B (pas dtau/2), aux memes instants. Renvoie le maximum, sur les
// signaux dont le pic atteint au moins 5 % du plus grand pic, de  max|qA - qB| / pic.
// La distance de derive d est tiree uniformement dans [d_min, dkg] : l'erreur est la plus grande
// pres de la grille (etalement temporel faible), donc le maximum est atteint vers d_min.
inline double tau_resolution_error(const PsiTable& A, const PsiTable& B, const DetectorConfig& cfg,
                                   const DerivedParams& D, double d_min, int n_samples)
{
    const int ntA = A.n_tau(), ntB = B.n_tau();
    std::vector<double> vecA(ntA), vecB(ntB), padA(ntA + 2), padB(ntB + 2);
    const double tau_lo = std::min(A.tau_min(), B.tau_min());
    const double tau_hi = std::max(A.tau_max(), B.tau_max());
    const int ne = (int)std::ceil((tau_hi - tau_lo) / cfg.delta_t) + 3;
    std::vector<int>    idxA(ne), idxB(ne);
    std::vector<double> edA(ne), edB(ne);
    double wA[4], wB[4];
    const double uv_max = std::min(2.0 * D.pitch, D.reach_cm);

    std::vector<double> errs, amps;
    double amp_max = 0.0;
    unsigned long seed = 20241109UL;
    for (int s = 0; s < n_samples; ++s) {
        const double u = (2.0 * tau_rnd(seed) - 1.0) * uv_max;
        const double v = (2.0 * tau_rnd(seed) - 1.0) * uv_max;
        const double d = d_min + (cfg.dkg - d_min) * tau_rnd(seed);
        const double tau0 = tau_lo + tau_rnd(seed) * cfg.delta_t;

        A.psi_vector(u, v, d, &vecA[0]);
        B.psi_vector(u, v, d, &vecB[0]);
        padA[0] = vecA[0];  padB[0] = vecB[0];
        for (int n = 0; n < ntA; ++n) padA[n + 1] = vecA[n];
        for (int n = 0; n < ntB; ++n) padB[n + 1] = vecB[n];
        padA[ntA + 1] = vecA[ntA - 1];  padB[ntB + 1] = vecB[ntB - 1];

        if (!A.edges_prepare(tau0, cfg.delta_t, ne, &idxA[0], wA) ||
            !B.edges_prepare(tau0, cfg.delta_t, ne, &idxB[0], wB))
            throw std::logic_error("tau_resolution_error : pas de table incompatible avec delta_t (bug)");
        A.edges_values(&padA[0], &idxA[0], wA, ne, &edA[0]);
        B.edges_values(&padB[0], &idxB[0], wB, ne, &edB[0]);

        double amp = 0.0, err = 0.0;
        for (int j = 0; j + 1 < ne; ++j) {
            const double qa = edA[j + 1] - edA[j];
            const double qb = edB[j + 1] - edB[j];
            amp = std::max(amp, std::fabs(qa));
            err = std::max(err, std::fabs(qa - qb));
        }
        errs.push_back(err);  amps.push_back(amp);
        amp_max = std::max(amp_max, amp);
    }
    double worst = 0.0;
    for (size_t s = 0; s < errs.size(); ++s)
        if (amps[s] >= 0.05 * amp_max && amps[s] > 0.0) worst = std::max(worst, errs[s] / amps[s]);
    return worst;
}

// Parametres d'une table du controle de resolution en tau. Les points tires ne sont qu'a moins de
// 2 pitchs du pad : on reduit n_out a cette zone (memoire et temps), sans toucher a la table de production.
inline TableParams tau_check_params(const DetectorConfig& cfg, const PadPlaneParams& pp,
                                    const TableOptions& opt, int R, DerivedParams& D, double& mb)
{
    TableOptions o = opt;  o.R = R;
    CheckList tmp;                                   // les echecs de ces calculs d'essai ne comptent pas
    D = derive_params(cfg, pp, o, tmp);
    TableParams tp = D.tp;
    const double uv_max = std::min(2.0 * D.pitch, D.reach_cm);
    const int n = (int)std::ceil(uv_max / tp.step_out) + 3;
    if (n < tp.n_out) tp.n_out = n;
    mb = (double)tp.n_d * tp.n_out * tp.n_out * D.n_tau * 4.0 / 1.0e6;
    return tp;
}

// Choisit (table_R = 0) ou verifie (table_R > 0) le pas de table delta_t / R.
// Reference : table de R = tau_Rref (ou 2R si R >= tau_Rref). Pour chaque candidat R = 1, ..., 6,
// on construit sa table et on mesure l'ecart a la reference (tau_resolution_error) ; on retient
// le plus petit R dont l'ecart est <= tau_tol. On controle aussi que la reference est assez fine :
// l'ecart entre R_ref/2 et R_ref, divise par ~4 (convergence en dtau^2.3), estime son biais.
inline int resolve_table_R(const DetectorConfig& cfg, const PadPlaneParams& pp,
                           const TableOptions& opt, CheckList& chk)
{
    if (opt.tau_tol <= 0.0) {
        if (opt.R < 1) throw std::runtime_error("table_R = 0 (automatique) demande table_tau_tol > 0");
        return opt.R;
    }
    if (opt.tau_dmin < 0.0 || opt.tau_dmin >= cfg.dkg)
        throw std::runtime_error("table_tau_dmin doit etre dans [0, dkg[");
    const bool automatique = (opt.R < 1);
    if (automatique && opt.tau_Rref < 8)
        throw std::runtime_error("table_tau_Rref doit etre >= 8 pour le choix automatique de R");
    std::vector<int> cand;
    if (automatique) { for (int r = 1; r <= 6; ++r) cand.push_back(r); }
    else             { cand.push_back(opt.R); }
    const int Rref = (cand.back() < opt.tau_Rref) ? opt.tau_Rref : 2 * cand.back();
    const int Rh   = Rref / 2;

    DerivedParams DR, DH;
    double mb_ref = 0.0, mb_h = 0.0;
    TableParams tpR = tau_check_params(cfg, pp, opt, Rref, DR, mb_ref);
    TableParams tpH = tau_check_params(cfg, pp, opt, Rh,   DH, mb_h);
    if (mb_ref > opt.max_table_mb) {
        std::ostringstream s;
        s << "la table de reference du controle (R = " << Rref << ") ferait " << mb_ref << " Mo (plafond "
          << opt.max_table_mb << ") : resolution en tau non verifiable. Augmenter max_table_mb, diminuer "
          << "table_tau_Rref, ou imposer table_R avec table_tau_tol = 0.";
        chk.fail("tau_resolution", s.str());
        return automatique ? 1 : opt.R;
    }
    std::cout << "Controle de la resolution en tau (tolerance " << 100.0 * opt.tau_tol << " %, voxels a d >= "
              << opt.tau_dmin << " cm, reference R = " << Rref << ") :\n";
    PsiTable ref(tpR);
    PsiTable half(tpH);
    const double conv = tau_resolution_error(half, ref, cfg, DR, opt.tau_dmin, 2000);
    const double bias = conv / 4.0;
    std::cout << "  reference : R = " << Rh << " contre R = " << Rref << " : ecart = " << 100.0 * conv
              << " %  (biais estime de la reference ~ " << 100.0 * bias << " %)\n";
    if (bias > 0.5 * opt.tau_tol) {
        std::ostringstream s;
        s << "la table de reference (R = " << Rref << ") n'est pas assez fine : biais estime ~ " << 100.0 * bias
          << " % pour une tolerance de " << 100.0 * opt.tau_tol << " %. Augmenter table_tau_Rref.";
        chk.fail("tau_resolution", s.str());
    }

    double err = 0.0;
    for (size_t k = 0; k < cand.size(); ++k) {
        const int R = cand[k];
        if (R == Rh) {
            err = conv;                              // deja calculee
        } else {
            DerivedParams DC;  double mbc = 0.0;
            TableParams tpc = tau_check_params(cfg, pp, opt, R, DC, mbc);
            PsiTable T(tpc);
            err = tau_resolution_error(T, ref, cfg, DC, opt.tau_dmin, 2000);
        }
        std::cout << "  R = " << R << " : ecart a R = " << Rref << " : " << 100.0 * err << " %\n";
        if (err <= opt.tau_tol) return R;
    }
    std::ostringstream s;
    if (automatique) {
        s << "aucun R <= 6 n'atteint la tolerance " << 100.0 * opt.tau_tol << " % (R = 6 : ecart " << 100.0 * err
          << " %). R = 6 est retenu. Augmenter table_tau_tol, ou imposer table_R = 7 ou plus avec "
          << "table_tau_tol = 0 (et table_tau_Rref plus grand). (Cause probable : diffusion longitudinale "
          << "faible devant delta_t.)";
    } else {
        s << "table_R = " << opt.R << " : ecart a R = " << Rref << " de " << 100.0 * err << " % (> tolerance "
          << 100.0 * opt.tau_tol << " %). Essayer un R plus grand, ou table_R = 0 (automatique), ou augmenter "
          << "table_tau_tol.";
    }
    chk.fail("tau_resolution", s.str());
    return automatique ? 6 : opt.R;
}

inline void print_derived(const DerivedParams& D, const DetectorConfig& cfg) {
    std::cout << "\n--- Parametres derives ---\n"
              << "pitch              = " << D.pitch * 10.0 << " mm    gap = " << cfg.dga * 10.0 << " mm\n"
              << "T_gap              = " << D.T_gap << " ns    n_h = " << D.n_h
              << "    dtau = " << D.dtau << " ns (delta_t/" << D.R << ")\n"
              << "dernier pas (partiel) = " << D.T_frac_ns << " ns (sur " << D.dtau << " ns)\n"
              << "plage en tau       = [" << D.m_min * D.dtau << ", " << D.m_max * D.dtau
              << "] ns    n_tau = " << D.n_tau << "\n"
              << "noeuds u,v         : pas = " << D.tp.step_out * 10.0 << " mm    n_out = " << D.tp.n_out << "\n"
              << "source phi         : m_cells = " << D.tp.m_cells << "    r_src = " << D.tp.r_src
              << " cm    n_img = " << D.tp.n_img << "    n_d = " << D.tp.n_d << "\n"
              << "portee d'induction = " << D.reach_cm << " cm\n"
              << "table estimee      = " << D.table_mb << " Mo\n"
              << "n_bins             = " << D.n_bins << "    (queue du signal : " << D.tau_tail_ns << " ns)\n\n";
}

// Controles faits apres construction de la table
inline void check_table(const PsiTable& t, const DerivedParams& D, CheckList& chk) {
    if (t.n_tau() != D.n_tau || t.n_out() != D.tp.n_out)
        throw std::logic_error("derive_params et PsiTable ne sont pas d'accord (n_tau ou n_out) : bug");

    if (t.phi_edge_max() > 1.0e-6) {
        std::ostringstream s;
        s << "phi vaut " << t.phi_edge_max() << " (> 1e-6) au bord de la source r_src = " << D.tp.r_src
          << " cm : source tronquee trop tot. Augmenter table_r_src (par exemple " << 1.5 * D.tp.r_src
          << " cm).";
        chk.fail("phi_edge", s.str());
    }
}
