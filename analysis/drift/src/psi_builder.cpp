#include "psi_builder.hpp"
#include "weighting_potential.hpp"
#include <cmath>
#include <stdexcept>

namespace {

// Fonction de répartition de la gaussienne réduite
double gauss_cdf(double t) {
    return 0.5 * (1.0 + std::erf(t * 0.70710678118654752));
}

// Les cellules sources d'indice j < 0 sont les images miroir de celles
// d'indice -j-1 (phi est symétrique en x -> -x et en y -> -y).
int fold(int j) {
    return (j >= 0) ? j : -j - 1;
}

// Poids de la convolution gaussienne 1D.
// Pour chaque noeud de sortie i (position u_i = i*step_out), on garde la liste
// des cellules sources j (indices complets, positifs et négatifs) situées à
// moins de 5 sigma, avec pour poids l'INTEGRALE de la gaussienne sur la cellule
// (différence de deux erf). Ainsi une fonction constante par cellule est
// intégrée exactement, y compris le créneau phi(h=0).
struct Weights1D {
    std::vector<int> jmin;
    std::vector< std::vector<double> > w;
};

void make_weights(double sigma, double step_src, double step_out,
                  int n_out, int nc, Weights1D& W)
{
    W.jmin.assign(n_out, 0);
    W.w.assign(n_out, std::vector<double>());
    for (int i = 0; i < n_out; ++i) {
        double u = i * step_out;
        int jlo = (int)std::floor((u - 5.0 * sigma) / step_src);
        int jhi = (int)std::floor((u + 5.0 * sigma) / step_src);
        if (jlo < -nc)     jlo = -nc;
        if (jhi > nc - 1)  jhi = nc - 1;
        W.jmin[i] = jlo;
        for (int j = jlo; j <= jhi; ++j) {
            double bas  = j * step_src;
            double haut = (j + 1) * step_src;
            W.w[i].push_back(gauss_cdf((haut - u) / sigma) - gauss_cdf((bas - u) / sigma));
        }
    }
}

// Lisse le niveau l de la source, d'abord en u puis en v (la gaussienne est
// séparable). Résultat : out[i * n_out + j].
// tmp est un tableau de travail (réutilisé d'un appel à l'autre).
void smooth_level(const PhiSource& src, int l, const Weights1D& W,
                  int n_out, std::vector<double>& tmp, double* out)
{
    const int nc = src.n_cells();
    tmp.assign(n_out * nc, 0.0);

    // passe 1 : lissage en u, pour chaque ligne source k >= 0
    for (int i = 0; i < n_out; ++i) {
        double* t = &tmp[i * nc];
        int nw = (int)W.w[i].size();
        for (int q = 0; q < nw; ++q) {
            double w = W.w[i][q];
            const double* ligne = src.row(l, fold(W.jmin[i] + q));
            for (int k = 0; k < nc; ++k) t[k] += w * ligne[k];
        }
    }

    // passe 2 : lissage en v
    for (int i = 0; i < n_out; ++i) {
        const double* t = &tmp[i * nc];
        for (int j = 0; j < n_out; ++j) {
            int nw = (int)W.w[j].size();
            double s = 0.0;
            for (int q = 0; q < nw; ++q) {
                s += W.w[j][q] * t[fold(W.jmin[j] + q)];
            }
            out[i * n_out + j] = s;
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// PhiSource
// ---------------------------------------------------------------------------

PhiSource::PhiSource(double a, double g, int n_img, int m_cells, double r_src, int n_h, double dh)
    : m_a(a), m_g(g), m_step(a / m_cells), m_rsrc(r_src), m_dh(dh), m_nc(0), m_nh(n_h)
{
    m_nc = (int)std::ceil(r_src / m_step);
    m_data.assign((size_t)(n_h + 1) * m_nc * m_nc, 0.0);

    for (int l = 0; l < n_h; ++l) {          // l = n_h (h = g) : phi = 0, déjà à zéro
        double h = height(l);
        for (int j = 0; j < m_nc; ++j) {
            double x = (j + 0.5) * m_step;
            for (int k = 0; k <= j; ++k) {   // phi(x,y) = phi(y,x) : on ne calcule que k <= j
                double y = (k + 0.5) * m_step;
                if (x * x + y * y > r_src * r_src) continue;
                double v;
                if (l == 0) {
                    // h = 0 : indicatrice du pad (les centres de cellule ne sont jamais sur le bord)
                    v = (x < a && y < a) ? 1.0 : 0.0;
                } else {
                    v = phi_point(x, y, h, a, g, n_img);
                }
                m_data[((size_t)l * m_nc + j) * m_nc + k] = v;
                m_data[((size_t)l * m_nc + k) * m_nc + j] = v;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// build_psi
// ---------------------------------------------------------------------------

void build_psi(const PhiSource& src, const PsiParams& P, std::vector<double>& psi)
{
    const int    n_h   = src.n_h();
    const double g     = src.g();
    const double dh    = src.dh();
    const int    n_out = P.n_out;
    const int    n_tau = P.m_max - P.m_min + 1;

    if (std::fabs(dh - P.ve_ga * P.dtau) > 1.0e-9 * g) {
        throw std::runtime_error("build_psi : dh doit valoir ve_ga*dtau");
    }
    if (!(n_h * dh >= g * (1.0 - 1.0e-9)) || !((n_h - 1) * dh < g)) {
        throw std::runtime_error("build_psi : n_h doit valoir ceil(g / (ve_ga*dtau))");
    }

    // --- Etape 1 : lissage latéral de chaque niveau de hauteur --------------
    // lat[(l * n_out + i) * n_out + j]
    std::vector<double> lat((size_t)(n_h + 1) * n_out * n_out, 0.0);
    std::vector<double> tmp;
    Weights1D W;

    // Plancher sur sigma : quand sigma devient plus petit qu'une demi-cellule
    // (très près de la grille, h -> g, pour d ~ 0), la gaussienne n'est plus
    // résolue. Ce n'est pas gênant : phi y est lisse et petit (il tend vers 0).
    const double sigma_min = 0.5 * src.step();

    for (int l = 0; l <= n_h; ++l) {
        double h  = src.height(l);
        double s2 = P.sigma0 * P.sigma0 + P.DT_ga * P.DT_ga * (g - h);
        double sigma = std::sqrt(s2);
        if (sigma < sigma_min) sigma = sigma_min;
        make_weights(sigma, src.step(), P.step_out, n_out, src.n_cells(), W);
        smooth_level(src, l, W, n_out, tmp, &lat[(size_t)l * n_out * n_out]);
    }

    // --- Remise en ordre : pour chaque (i,j), les niveaux l contigus ---------
    // latT[(i * n_out + j) * (n_h + 1) + l]
    std::vector<double> latT((size_t)n_out * n_out * (n_h + 1), 0.0);
    for (int l = 0; l <= n_h; ++l) {
        for (int ij = 0; ij < n_out * n_out; ++ij) {
            latT[(size_t)ij * (n_h + 1) + l] = lat[(size_t)l * n_out * n_out + ij];
        }
    }

    // --- Etape 2 : poids de la convolution temporelle -----------------------
    // Poids d'une cellule de temps [(j-1/2) dtau, (j+1/2) dtau].
    std::vector<double> wt;
    int jmax = 0;
    if (P.sigma_t < 0.3 * P.dtau) {
        wt.push_back(1.0);                         // pas d'étalement
    } else {
        jmax = (int)std::ceil(5.0 * P.sigma_t / P.dtau);
        for (int j = -jmax; j <= jmax; ++j) {
            double hi = (j + 0.5) * P.dtau / P.sigma_t;
            double lo = (j - 0.5) * P.dtau / P.sigma_t;
            wt.push_back(gauss_cdf(hi) - gauss_cdf(lo));
        }
    }

    // --- Convolution --------------------------------------------------------
    // Un électron qui arrive à la grille avec le retard delta = j*dtau a, au
    // temps tau = m*dtau, passé q = m - j pas dans le gap. Sa hauteur est
    //   h = g - q*dh   -> niveau l = n_h - q (h_l = src.height(l), partielle pour l = 1).
    //   q <= 0   : pas encore entré dans le gap  -> 0
    //   q >= n_h : collecté                      -> valeur à l'anode (l = 0)
    psi.assign((size_t)n_out * n_out * n_tau, 0.0);
    for (int ij = 0; ij < n_out * n_out; ++ij) {
        const double* L   = &latT[(size_t)ij * (n_h + 1)];
        double*       sor = &psi[(size_t)ij * n_tau];
        for (int n = 0; n < n_tau; ++n) {
            int m = P.m_min + n;
            double s = 0.0;
            for (int jj = -jmax; jj <= jmax; ++jj) {
                int q = m - jj;
                double val;
                if (q <= 0)         val = 0.0;
                else if (q >= n_h)  val = L[0];
                else                val = L[n_h - q];
                s += wt[jj + jmax] * val;
            }
            sor[n] = s;
        }
    }
}

// ---------------------------------------------------------------------------
// edge_max : controle de la troncature de la source
// ---------------------------------------------------------------------------

double PhiSource::edge_max() const
{
    const double r_lo = m_rsrc - 2.0 * m_step;
    double mx = 0.0;
    for (int l = 0; l <= m_nh; ++l) {
        for (int j = 0; j < m_nc; ++j) {
            double x = (j + 0.5) * m_step;
            for (int k = 0; k < m_nc; ++k) {
                double y = (k + 0.5) * m_step;
                double r2 = x * x + y * y;
                if (r2 > m_rsrc * m_rsrc || r2 < r_lo * r_lo) continue;
                double v = std::fabs(m_data[((size_t)l * m_nc + j) * m_nc + k]);
                if (v > mx) mx = v;
            }
        }
    }
    return mx;
}
