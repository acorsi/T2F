#pragma once

// ---------------------------------------------------------------------------
// Lecture d'un evenement : liste UNIQUE de pads, mise en forme par le preampli, bruit, CAN, seuil.
//
// Entree : les listes H et L de PadCharge produites par PadPlane::project_pair (un pad touche par
//          les deux fragments est dans les deux listes, avec la meme somme H+L).
// Sortie : un ReadoutEvent : une seule liste de pads, avec un flag
//          1 = H seul, 2 = L seul, 3 = H et L (induction comprise), 0 = pad fantome (voir ghost.hpp).
//
// Chaine, par pad :
//   signal sur la grille de dt (n_bins bins, zeros hors de la zone ecrite par project_pair)
//   -> h (*) signal -> [+ bruit parallele et serie filtres] -> echantillons du CAN (phase aleatoire
//   commune a l'evenement) -> [+ bruit du CAN] -> [seuil de lecture]
//
// Pads fantomes (avec bruit seulement, si set_ghosts a ete appele) : chaque pad du detecteur absent de la
// liste vraie (H, L) est lu avec la probabilite p (ghost.hpp) ; sa courbe est tiree sous la condition
// "lu". Fenetre ecrite d'un fantome : longueur fixe ghost_window_ns, centree sur le premier
// franchissement du seuil (decalee, pas raccourcie, au bord de la fenetre complete). Les fantomes
// utilisent un generateur a part (graine derivee de celle de l'evenement) : les pads du signal sont
// strictement identiques avec ou sans fantomes.
//
// noise_on = false : meme chaine, sans bruit ni seuil de lecture (tous les pads de la liste sont
// gardes). La phase est tiree dans les deux cas, en premier, dans le meme generateur : deux runs
// (avec / sans bruit) ont donc la meme phase et des courbes comparables echantillon par echantillon.
//
// Fenetre ecrite pour chaque pad : zone du signal vrai (premier a dernier bin garde par eps_qt dans
// project_pair), allongee de la queue complete de la reponse (n_taps - 1 bins : la convolution du signal
// par h est exacte et finie), et elargie de margin_ns de chaque cote. La grille de lecture est rallongee
// en consequence (grid_bins()) : la fenetre n'est donc jamais tronquee par la fin de la grille physique. Elle ne depend que du signal vrai : elle est identique dans
// un run avec et sans bruit.
//
// Temps d'un echantillon j de la fenetre complete : t = (phase + j * adc_ratio) * dt (ns, depuis t = 0).
//
// C++ classique.
// ---------------------------------------------------------------------------

#include <stdint.h>
#include <map>
#include <vector>
#include "noise.hpp"
#include "ghost.hpp"
#include "pad_geometry.hpp"     // PadCharge

struct ReadoutParams {
    bool   noise_on;       // false : pas de bruit, pas de seuil de lecture
    double thr_sigma;      // seuil de lecture = thr_sigma * enc_total (e- equivalents), sortie du CAN
    int    n_consec;       // nombre d'echantillons consecutifs au-dessus du seuil (>= 1)
    double margin_ns;      // marge de fenetre de chaque cote (ns)

    ReadoutParams() : noise_on(false), thr_sigma(4.0), n_consec(1), margin_ns(100.0) {}
};

// un pad du detecteur (liste complete, pour tirer les fantomes)
struct PadSite {
    int    pad_id;
    double x, y;
};

struct ReadoutPad {
    int    pad_id;
    double x, y;
    double q;                       // charge vraie du pad (H+L), e-
    int    flag;                    // 1 = H, 2 = L, 3 = H et L, 0 = fantome (alors q = 0)
    int    j0;                      // indice (echantillons CAN) du premier echantillon ecrit
    std::vector<double> samples;    // echantillons ecrits, e- equivalents
};

struct ReadoutEvent {
    int phase;                      // decalage de l'horloge CAN, en bins de dt (0 .. adc_ratio-1)
    int n_ghost;                    // nombre de pads fantomes (flag 0)
    std::vector<ReadoutPad> pads;   // classes par pad_id croissant
};

class Readout {
public:
    // n_bins : nombre de bins de la grille physique (cfg.n_bins()) ; la grille de lecture est plus longue
    Readout(const ReadoutParams& rp, const NoiseModel& noise, int n_bins);

    // Active les pads fantomes : g (calibre, voir GhostModel::calibrate) et la liste de tous les pads du
    // detecteur ; g doit vivre aussi longtemps que le Readout. A appeler avant le traitement.
    void set_ghosts(const GhostModel* g, const std::vector<PadSite>& sites);

    // Bruit par pad (modele "capacitance", voir cap_noise.hpp) : pour chaque pad du detecteur, ser_scale = facteur
    // du bruit serie et enc_scale = ENC du pad / ENC de reference (pour les fantomes). Les trois tableaux ont
    // la meme taille. Une fois la table donnee, tout pad lu doit s'y trouver (exception sinon). Sans appel :
    // tous les pads ont le bruit de reference (ser_scale = enc_scale = 1). A appeler avant le traitement.
    void set_pad_noise(const std::vector<int>& pad_ids, const std::vector<double>& ser_scale,
                       const std::vector<double>& enc_scale);

    // seed : graine de l'evenement (le meme tirage de phase doit etre obtenu avec et sans bruit)
    // thread-safe : n'ecrit que dans out
    void process(const std::vector<PadCharge>& H, const std::vector<PadCharge>& L,
                 uint64_t seed, ReadoutEvent& out) const;

    double threshold() const { return m_thr; }          // e-
    int    n_bins() const { return m_n_bins; }          // grille physique
    int    grid_bins() const { return m_grid; }         // grille de lecture = n_bins + tail_bins + margin_bins
    int    tail_bins() const { return m_tail; }         // n_taps de h - 1
    int    margin_bins() const { return m_margin; }
    const ReadoutParams& params() const { return m_rp; }

private:
    ReadoutParams m_rp;
    const NoiseModel* m_noise;
    int    m_n_bins;
    int    m_grid;      // grille de lecture (bins)
    double m_thr;
    int    m_tail;      // queue complete de h (bins) : n_taps - 1
    int    m_margin;    // marge en bins
    const GhostModel*    m_ghost;   // 0 : pas de fantomes
    std::vector<PadSite> m_sites;
    struct PadNoise { double ser_scale; double enc_scale; };
    std::map<int, PadNoise> m_pn;   // vide : tous les pads au bruit de reference
};
