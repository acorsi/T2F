#pragma once

#include <vector>

// ---------------------------------------------------------------------------
// Construction de Psi(u, v, tau) pour UNE valeur de la distance de dérive d.
//
// Psi = charge induite (en fraction de la charge du nuage) sur un pad dont le
// centre est décalé de (u, v) par rapport au centre du nuage, au temps tau
// compté depuis l'entrée du CENTRE du nuage dans le gap.
//
// Deux étapes :
//   1. lissage latéral : phi (ponctuel) convolué avec une gaussienne de largeur
//      sigma(h)^2 = sigma0^2 + DT_ga^2 * (g - h), pour chaque hauteur h ;
//   2. convolution temporelle : étalement gaussien sigma_t du retard d'arrivée
//      à la grille, avec h = g - ve_ga * (tau - delta).
//
// Unités : toutes les longueurs en cm, les temps en ns.
// ---------------------------------------------------------------------------

// phi ponctuel échantillonné sur une grille de cellules carrées, pour les
// hauteurs h_l = l * g / n_h (l = 0 ... n_h).
//
// Le pad est centré à l'origine. Les cellules ont pour côté  step = a / m_cells,
// donc les bords du pad tombent exactement sur des frontières de cellules
// (important : en h = 0, phi vaut 1 sur le pad et 0 ailleurs, et le lissage
// redonne alors exactement la fraction collectée fx*fy).
//
// Par symétrie on ne stocke que le quart de plan x >= 0, y >= 0. La cellule
// (j, k) a pour centre ((j+0.5)*step, (k+0.5)*step).
class PhiSource {
public:
    // a       : demi-côté du pad (la tuile p x p, soit p/2)
    // g       : épaisseur du gap
    // n_img   : nombre d'images de la série (10 suffit : erreur ~ 1e-7)
    // m_cells : nombre de cellules par demi-pad (8 -> pas = p/16)
    // r_src   : rayon au-delà duquel phi est pris nul
    // n_h     : nombre d'intervalles en h ; dh : pas en h (= ve_ga * dtau), voir height()
    PhiSource(double a, double g, int n_img, int m_cells, double r_src, int n_h, double dh);

    double a()       const { return m_a; }
    double g()       const { return m_g; }
    double step()    const { return m_step; }
    int    n_cells() const { return m_nc; }
    int    n_h()     const { return m_nh; }
    double dh()      const { return m_dh; }
    // hauteur du niveau l : h_l = max(0, g - (n_h - l) * dh)  (l = n_h : grille ; l = 0 : anode, collecte)
    double height(int l) const { double h = m_g - (m_nh - l) * m_dh; return (h > 0.0) ? h : 0.0; }
    // max de |phi| sur les cellules du bord (r_src - 2 pas < r <= r_src), tous niveaux de hauteur
    double edge_max() const;

    // pointeur sur la ligne (niveau l, colonne j), de longueur n_cells()
    const double* row(int l, int j) const { return &m_data[(l * m_nc + j) * m_nc]; }

private:
    double m_a, m_g, m_step, m_rsrc, m_dh;
    int    m_nc, m_nh;
    std::vector<double> m_data;
};

struct PsiParams {
    double sigma0;    // sigma transverse à la grille = sigma_T(d)  (cm)
    double DT_ga;     // diffusion transverse dans le gap (cm/sqrt(cm)) ; 0 = sigma constant
    double sigma_t;   // étalement temporel des arrivées à la grille (ns) ; 0 = pas de convolution
    double dtau;      // pas en tau (ns)
    double ve_ga;     // vitesse de dérive dans le gap (cm/ns)
    int    m_min;     // tau = m * dtau, de m_min (<= 0) ...
    int    m_max;     //                 ... à m_max
    int    n_out;     // nombre de noeuds en u et en v (quart de plan)
    double step_out;  // pas des noeuds de sortie (cm) : u_i = i*step_out
};

// Nombre de points en tau
inline int psi_n_tau(const PsiParams& P) { return P.m_max - P.m_min + 1; }

// Remplit psi, de taille n_out * n_out * n_tau, avec l'ordre
//     psi[ (i * n_out + j) * n_tau + n ]   ,   tau = (m_min + n) * dtau
// (u = i*step_out, v = j*step_out, tau est l'indice le plus rapide).
//
// Condition : dh = ve_ga * dtau (un pas en tau = un niveau en h), avec n_h = ceil(g / dh) :
// le niveau le plus bas (l = 1) peut avoir une hauteur partielle, l = 0 est la collecte.
void build_psi(const PhiSource& src, const PsiParams& P, std::vector<double>& psi);
