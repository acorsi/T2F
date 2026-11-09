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
    std::string ignore;        // identifiants de controles a ignorer, separes par des virgules

    TableOptions()
        : n_d(10), n_img(10), m_cells(0), step_out(0.0), r_src(0.0), max_table_mb(500.0) {}
};

struct DerivedParams {
    double pitch, a;
    double T_gap;          // temps de collecte dans le gap (ns)
    double dtau;           // pas de la table en tau (ns)
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
        ids.push_back("T_gap_non_multiple");
        ids.push_back("n_h_small");
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

    // --- temps dans le gap : la table impose dtau = delta_t, donc T_gap multiple de delta_t ---
    D.T_gap = g / cfg.ve_ga;
    const double ratio = D.T_gap / cfg.delta_t;
    int n_h = (int)(ratio + 0.5);

    if (std::fabs(ratio - n_h) > 1.0e-6) {
        int lo = (int)std::floor(ratio);
        if (lo < 1) lo = 1;
        int hi = lo + 1;
        std::ostringstream s;
        s << "dga/ve_ga = " << D.T_gap << " ns n'est pas un multiple de delta_t = " << cfg.delta_t
          << " ns (rapport " << ratio << "). Le calcul rapide exige un rapport entier. "
          << "Valeurs voisines : dga = " << lo * cfg.delta_t * cfg.ve_ga << " cm ou "
          << hi * cfg.delta_t * cfg.ve_ga << " cm (pour ve_ga = " << cfg.ve_ga << "), ou ve_ga = "
          << g / (hi * cfg.delta_t) << " ou " << g / (lo * cfg.delta_t) << " cm/ns (pour dga = " << g
          << "). Si on continue quand meme : voie lente et borne de selection fausse.";
        chk.fail("T_gap_non_multiple", s.str());
    }
    if (n_h < 4) {
        std::ostringstream s;
        s << "dga/(ve_ga*delta_t) = " << ratio << " : le signal d'induction dans le gap est decrit par moins de "
          << "4 pas de table (T_gap = " << D.T_gap << " ns pour delta_t = " << cfg.delta_t << " ns).";
        chk.fail("n_h_small", s.str());
    }
    if (n_h < 2) n_h = 2;
    D.n_h  = n_h;
    D.dtau = g / (cfg.ve_ga * n_h);

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

    // rayon de la source : phi decroit comme exp(-pi r / g) ; 5 g donne ~1e-7 (verifie apres coup : phi_edge)
    tp.r_src = (opt.r_src > 0.0) ? opt.r_src : std::sqrt(2.0) * D.a + 5.0 * g;

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

inline void print_derived(const DerivedParams& D, const DetectorConfig& cfg) {
    std::cout << "\n--- Parametres derives ---\n"
              << "pitch              = " << D.pitch * 10.0 << " mm    gap = " << cfg.dga * 10.0 << " mm\n"
              << "T_gap              = " << D.T_gap << " ns    n_h = " << D.n_h
              << "    dtau = " << D.dtau << " ns\n"
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
