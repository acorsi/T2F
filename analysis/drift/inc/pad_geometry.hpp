#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include "detector.hpp"
#include "psi_table.hpp"

// ---------------------------------------------------------------------------
// Géométrie du plan de pads
// ---------------------------------------------------------------------------

struct Pad {
    double x, y;     // centre du pad (cm)
    double size;     // côté du pad (cm), carré (0 pour les anneaux)
    double r_in;     // rayon intérieur (anneaux seulement)
    double r_out;    // rayon extérieur (anneaux seulement)
    int    sx;       // quadrant : +1 (x>0) ou -1 (x<0)
    int    sy;       // quadrant : +1 (y>0) ou -1 (y<0)
    int    id;
};

// Signal d'un pad pour un fragment. Les tableaux t, qt, Q ont la même taille :
// un élément par bin de delta_t gardé par les seuils de lecture.
//   t  : début du bin, en ns depuis la fission (t = k * delta_t)
//   qt : charge induite pendant le bin [t, t + delta_t], en électrons (signé)
//   Q  : charge cumulée à la FIN du bin (somme de tous les qt depuis le début,
//        y compris les bins non gardés), en électrons
struct PadCharge {
    int    pad_id;
    double x, y;
    double q;                  // charge finale du pad, en électrons (= dernier Q)
    std::vector<double> t;
    std::vector<double> qt;
    std::vector<double> Q;
};

// ---------------------------------------------------------------------------
// Paramètres géométriques
// ---------------------------------------------------------------------------

struct PadPlaneParams {
  //Paramètres de géométrie du plan de pad
    double pad_size     = 0.3;   // côté des pads cartésiens (cm)
    double inter_pad    = 0.015; // inter-pad (cm)
    double r_quinconce  = 8.1;   // rayon coins < r → pad central (cm)
    double r_transition = 8.2;   // séparation transition/anneaux (cm)
    double r_active     = 10.0;  // rayon externe (cm)
    int    n_rings      = 6;     // nombre de demi-anneaux périphériques
    double y_offset     = 0.0;   // doit rester 0 : la table suppose des tuiles jointives en y = 0

  //Paramètres liés à l'induction - Ne pas changer sans connaissance.
    double reach_pitch  = 4.5;  // portée de l induction autour du voxel, en pitchs (<= 6.1)
    double bound_margin = 1.3;  // marge sur la borne de |qt| par cellule ; <= 0 : pas de sélection par le seuil de lecture
};

// ---------------------------------------------------------------------------
// PadPlane
//
// Le plan est pavé de cellules p x p (p = pad_size + inter_pad) en briques : c'est le
// pavage des pads de la zone quinconce, prolongé sur tout le plan. Une cellule est
//   - réelle   : c'est un pad de la zone quinconce ;
//   - virtuelle : elle recouvre le pad de transition et les anneaux. Son signal est
//     réparti sur ces électrodes avec des fractions de recouvrement précalculées.
// Le signal de chaque cellule vient de la table PsiTable (induction de Ramo-Shockley).
// ---------------------------------------------------------------------------

class PadPlane {
public:
    PadPlane(const PadPlaneParams& params, const DetectorConfig& cfg);

    const std::vector<Pad>& pads() const { return m_pads; }

    std::vector<const Pad*> neighbors(double x0, double y0, double r) const;

    // x, y, z : position de chaque voxel (cm) ; n_electrons : charge du voxel.
    // sigma_L n est plus utilisé ; sigma_T ne sert qu à l avertissement de bord (la table recalcule les étalements à
    // partir de d = dkg - z) : ils restent dans la signature pour ne pas changer main.
    // qt_buf : tampon de travail propre au thread, n_pads x n_bins, rempli de zéros
    // à l'entrée et laissé à zéro en sortie.
    // Thread-safe : la table et la géométrie sont en lecture seule.
    std::vector<PadCharge> project(const std::vector<double>& x,
                                   const std::vector<double>& y,
                                   const std::vector<double>& z,
                                   const std::vector<double>& n_electrons,
                                   const std::vector<double>& sigma_T,
                                   const std::vector<double>& sigma_L,
                                   const DetectorConfig& cfg,
                                   std::vector<std::vector<double>>& qt_buf) const;

    // Pour les tests : nombre de cellules, et charge perdue au-delà de r_active.
    int n_cells() const { return (int)m_cells.size(); }
    const PsiTable& table() const { return m_table; }

private:
    // Une contribution d'une cellule vers une électrode : fraction de l'aire de la cellule.
    struct Contrib {
        int    pad;
        double frac;
    };
    struct Cell {
        double x, y;       // centre de la cellule (cm)
        int    first;      // indice de la première contribution dans m_contribs
        int count;         // nombre de contributions
    };

    PadPlaneParams  m_params;
    double          m_pitch;
    double          m_reach;           // portée en cm
    std::vector<Pad>      m_pads;
    std::vector<int> m_ring_ids, m_trans_ids ;

    PsiTable            m_table;
    std::vector<Cell>   m_cells;
    std::vector<Contrib> m_contribs;
    // grille de recherche des cellules : m_grid[(row + m_nrows) * (2*m_ncols + 1) + (col + m_ncols)]
    int                 m_nrows, m_ncols;
    std::vector<int>    m_grid;

    void build_anode();
    void build_cells();
    int  find_cell(int row, int col) const;
};

long pad_beyond_active_count() ;
long pad_traces_beyond_count();   // fragments avec au moins un voxel au-delà de r_active
long pad_traces_count();          // fragments traités
