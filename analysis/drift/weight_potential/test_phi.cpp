// Tests de phi_point contre les valeurs de référence du cours (unités : mm)
#include "weighting_potential.hpp"
#include <cstdio>
#include <cmath>

static int n_fail = 0;

static void check(const char* nom, double calcule, double attendu, double tol) {
    double ecart = std::fabs(calcule - attendu);
    const char* etat = (ecart <= tol) ? "OK " : "ECHEC";
    if (ecart > tol) n_fail++;
    printf("[%s] %-34s calcule = %.6f  attendu = %.6f  ecart = %.2e\n",
           etat, nom, calcule, attendu, ecart);
}

int main() {
    const double p = 3.15;       // pitch (mm)
    const double a = 0.5 * p;    // demi-côté
    const double g = 5.0;        // gap (mm)
    const int    N = 50;         // images

    // --- Table de référence (cours, tableau 1), tolérance 1e-4 (arrondi 4 chiffres)
    check("phi_demi(0,0,h=1)",     phi_halfspace(0, 0, 1.0, a),        0.5051, 1e-4);
    check("phi(0,0,h=1)",          phi_point(0, 0, 1.0, a, g, N),      0.4977, 1e-4);
    check("phi(0,0,h=2.5)",        phi_point(0, 0, 2.5, a, g, N),      0.1632, 1e-4);
    check("phi(p,0,h=1.5)",        phi_point(p, 0, 1.5, a, g, N),      0.0579, 1e-4);
    check("phi(p/2,p,h=1.5)",      phi_point(0.5*p, p, 1.5, a, g, N),  0.0431, 1e-4);
    check("phi(3p/2,p,h=2.5)",     phi_point(1.5*p, p, 2.5, a, g, N),  0.0085, 1e-4);
    check("phi(0,2p,h=2.5)",       phi_point(0, 2*p, 2.5, a, g, N),    0.0054, 1e-4);
    check("phi(0,0,h=4.5)",        phi_point(0, 0, 4.5, a, g, N),      0.0226, 1e-4);

    // --- Conditions aux limites
    check("phi sur le pad, h->0",  phi_point(0.3, -0.2, 1e-6, a, g, N), 1.0, 1e-4);
    check("phi hors pad, h->0",    phi_point(p, 0.0, 1e-6, a, g, N),    0.0, 1e-4);
    check("phi en h = g",          phi_point(0.3, 0.2, g, a, g, N),     0.0, 1e-12);
    check("phi en h = g-1e-9",     phi_point(0.3, 0.2, g-1e-9, a, g, N), 0.0, 1e-8);

    // --- Symétries
    check("symetrie u<->v",        phi_point(1.1, 2.3, 2.0, a, g, N),
                                   phi_point(2.3, 1.1, 2.0, a, g, N), 1e-12);
    check("symetrie u -> -u",      phi_point(1.1, 2.3, 2.0, a, g, N),
                                   phi_point(-1.1, 2.3, 2.0, a, g, N), 1e-12);

    // --- Convergence en nombre d'images
    check("N=10 contre N=200",     phi_point(0, 0, 2.5, a, g, 10),
                                   phi_point(0, 0, 2.5, a, g, 200), 1e-5);
    check("N=50 contre N=200",     phi_point(0, 0, 2.5, a, g, 50),
                                   phi_point(0, 0, 2.5, a, g, 200), 1e-6);

    // --- Règle de somme sur un pavage carré (la tuile pave le plan) :
    //     somme_P phi_P(x,y,h) = 1 - h/g, pour un point quelconque.
    {
        double x0 = 0.37, y0 = -0.81;
        double hs[3] = {0.5, 2.5, 4.5};
        for (int ih = 0; ih < 3; ++ih) {
            double h = hs[ih];
            double somme = 0.0;
            for (int i = -30; i <= 30; ++i) {
                for (int j = -30; j <= 30; ++j) {
                    somme += phi_point(x0 - i*p, y0 - j*p, h, a, g, N);
                }
            }
            char nom[64];
            snprintf(nom, sizeof(nom), "somme sur pavage, h=%.1f", h);
            // 3721 pads x ~5e-10 residuel par pad (queue des images) => ~2e-6
            check(nom, somme, 1.0 - h/g, 5e-6);
        }
    }

    // --- Maximum du voisin de même rangée (p,0) : 5.80 % vers h = 1.60 mm
    {
        double best = 0.0, hbest = 0.0;
        for (int k = 1; k < 5000; ++k) {
            double h = g * k / 5000.0;
            double v = phi_point(p, 0, h, a, g, N);
            if (v > best) { best = v; hbest = h; }
        }
        check("max phi(p,0)",       best,  0.0580, 5e-5);
        check("h du max phi(p,0)",  hbest, 1.60,   0.01);
    }

    printf("\n%s (%d echec(s))\n", n_fail == 0 ? "TOUS LES TESTS PASSENT" : "ECHECS", n_fail);
    return n_fail == 0 ? 0 : 1;
}
