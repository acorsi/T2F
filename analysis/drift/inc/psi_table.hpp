#pragma once

#include <vector>
#include <cstddef>
#include "detector.hpp"

// ---------------------------------------------------------------------------
// Table Psi(u, v, tau, d) construite une fois à l'initialisation, puis lue
// (en lecture seule : aucune donnée modifiée après le constructeur, donc
// utilisable en parallèle sans précaution).
//
//   (u, v) : décalage entre le centre du nuage et le centre du pad (cm)
//   tau    : temps depuis l'entrée du centre du nuage dans le gap (ns)
//   d      : distance de dérive cathode-grille du voxel, d = dkg - z (cm)
//
// Pour un nuage de N électrons, la charge induite sur le pad vaut N * Psi.
//
// Unités : cm et ns. La table est stockée en float, calculée en double.
// ---------------------------------------------------------------------------

struct TableParams {
    // --- détecteur ---
    double a;          // demi-côté de la tuile = pitch / 2
    double g;          // épaisseur du gap grille-anode (dga)
    double ve_kg, ve_ga;
    double DT_kg, DT_ga, DL_kg, DL_ga;
    double d_max;      // plus grande distance de dérive (dkg)

    // --- grille de la table ---
    int    n_d;        // nombre de noeuds en d (uniformes en ln sigma_anode(d), voir PsiTable)
    int    n_out;      // nombre de noeuds en u et en v (quart de plan)
    double step_out;   // pas en u et v (cm)
    int    n_h;        // nombre de niveaux de hauteur : pas en tau = g / (ve_ga * n_h)
    double tau_max_ns; // fin de la plage en tau (agrandie si besoin pour atteindre le plateau)

    // --- calcul de phi ---
    int    n_img;      // images de la série
    int    m_cells;    // cellules par demi-pad
    double r_src;      // rayon de la source (cm)
};

// Valeurs par défaut tirées de la configuration du détecteur.
// pitch = pad_size + inter_pad (cm).
TableParams default_table_params(const DetectorConfig& cfg, double pitch);

class PsiTable {
public:
    // Construction : quelques secondes.
    explicit PsiTable(const TableParams& tp);

    // --- description de la table ---
    int    n_tau()   const { return m_max - m_min + 1; }
    double dtau()    const { return m_dtau; }
    double tau_min() const { return m_min * m_dtau; }
    double tau_max() const { return m_max * m_dtau; }
    int    m_min_index() const { return m_min; }
    int    m_max_index() const { return m_max; }
    // Au-delà de u_max ou v_max, Psi est pris nul : à ne pas utiliser plus loin.
    double u_max()   const { return (m_tp.n_out - 1) * m_tp.step_out; }
    size_t memory_bytes() const { return m_data.size() * sizeof(float); }

    // sigma transverse et étalement temporel du nuage à la grille, pour la distance d
    double sigma_T(double d) const;
    double sigma_t(double d) const;

    // Les noeuds en d ne sont pas équidistants en d : ils sont équidistants en
    // x = ln(sigma_anode), avec sigma_anode^2 = DT_kg^2 d + DT_ga^2 g (+ un petit plancher).
    // C'est ce qui rend Psi lisse près de d = 0, où sigma est dominé par le plancher
    // de diffusion du gap. d_node(k) donne la distance d du noeud k (d_node(0) = 0).
    double d_node(int k) const;

    // Saut des cellules négligeables (voir PadPlane::project)
    int    d_cell(double d) const;                     // noeud en d juste en dessous de d
    double dpsi_max(double u, double v, int kd) const;   // majorant de max_tau |Psi(tau+dtau) - Psi(tau)|
    
    // Bords de bins d'un voxel : nedges bords aux temps tau0 + j*step.
    // Renvoie false si step n'est pas un multiple entier du pas de la table.
    // Sinon remplit idx[j] (-1 : avant la plage, -2 : après, sinon indice de noeud)
    // et w[0..3], poids de Lagrange communs à tous les bords.
    bool edges_prepare(double tau0, double step, int nedges, int* idx, double* w) const;

    // Valeurs aux bords. p pointe sur un vecteur "rembourré" de taille n_tau()+2 :
    // p[1+n] = Psi(tau_n), p[0] = p[1], p[n_tau()+1] = p[n_tau()].
    void edges_values(const double* p, const int* idx, const double* w,
                      int nedges, double* out) const;

    // Versions en float (voir pad_geometry.cpp) : même résultat à ~1e-6 près sur Psi.
    void psi_vector_f(double u, double v, double d, float* out) const;
    void edges_values_f(const float* p, const int* idx, const double* w,
                        int nedges, double* out) const; 
  
    // --- lecture ---
    // Remplit out[0 .. n_tau()-1] avec Psi aux noeuds en tau (tau_n = (m_min+n)*dtau),
    // pour le décalage (u,v) et la distance d.
    // Interpolation cubique (Lagrange 4 points) en u, v et d.
    // Coût : environ 64 * n_tau multiplications-additions.
    void psi_vector(double u, double v, double d, double* out) const;

    // Interpolation cubique en tau d'un vecteur produit par psi_vector.
    // tau < tau_min : 0 ; tau > tau_max : valeur du plateau (collecte terminée).
    double psi_at(const double* vec, double tau) const;

private:
    TableParams m_tp;
    double m_dtau;
    int    m_min, m_max;
    double m_c0;                 // DT_ga^2 g + plancher^2 : valeur de sigma_anode^2 en d = 0
    double m_x0, m_dx;           // x = ln(sigma_anode) : x du noeud 0 et pas entre noeuds
    std::vector<float> m_data; // [ ((k * n_out + i) * n_out + j) * n_tau + n ]
    std::vector<float> m_dmax;
    
};
