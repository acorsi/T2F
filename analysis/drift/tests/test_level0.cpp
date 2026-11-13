// ---------------------------------------------------------------------------
// Niveau 0 : potentiel de ponderation (weighting_potential.cpp)
//
// Ce programme ne fait AUCUNE comparaison : il ecrit les valeurs du code dans des CSV
// (out/l0_*.csv). Les references exactes sont calculees separement, en Python
// (test_level0.py), par une methode independante (transformee de Fourier 2D).
// Seules exceptions : quelques formules fermees triviales (misc), ecrites ici.
//
// Compilation et lancement : voir run_level0.sh
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include "weighting_potential.hpp"

static const double PI = 3.14159265358979323846;

struct Geo {
    const char* name;
    double a;   // demi-cote de la tuile = pitch / 2   (cm)
    double g;   // epaisseur du gap = dga              (cm)
};

int main()
{
    // A : pad 5 mm + inter-pad 0.15 mm (pitch 0.515), gap 3 mm   (drift.cfg actuel)
    // B : pad 3 mm + inter-pad 0.15 mm (pitch 0.315), gap 3 mm   (pads 3 mm commentes dans drift.cfg)
    // C : pad 5 mm, gap 5 mm                                      (dga = 0.5 : valeur choisie par moi, pas dans vos fichiers)
    Geo geos[3] = { {"A", 0.2575, 0.3}, {"B", 0.1575, 0.3}, {"C", 0.2575, 0.5} };

    const int N_IMG = 20;                       // table_n_img de drift.cfg

    const double hfrac[10] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95};
    // points (u/a, v/a)
    const double pts[9][2] = { {0, 0}, {0.5, 0}, {1, 0}, {1.5, 0}, {2, 0}, {3, 0}, {1, 1}, {2, 1}, {0.5, 0.5} };

    // ------------------------------------------------------------------ phi
    FILE* f = fopen("out/l0_phi.csv", "w");
    if (!f) { fprintf(stderr, "impossible d'ecrire out/l0_phi.csv (lancer depuis tests/)\n"); return 1; }
    fprintf(f, "geo,a,g,n_img,tag,u,v,h,phi\n");

    for (int ig = 0; ig < 3; ++ig) {
        const Geo& G = geos[ig];
        // points isoles, n_img = 20
        for (int ih = 0; ih < 10; ++ih) {
            double h = hfrac[ih] * G.g;
            for (int ip = 0; ip < 9; ++ip) {
                double u = pts[ip][0] * G.a, v = pts[ip][1] * G.a;
                fprintf(f, "%s,%.10g,%.10g,%d,pt,%.10g,%.10g,%.10g,%.15g\n", G.name, G.a, G.g, N_IMG,
                        u, v, h, phi_point(u, v, h, G.a, G.g, N_IMG));
            }
        }
        // balayage en u, v = 0, trois hauteurs
        const double hs[3] = {0.2, 0.5, 0.8};
        for (int ih = 0; ih < 3; ++ih) {
            double h = hs[ih] * G.g;
            for (int iu = 0; iu <= 40; ++iu) {
                double u = 4.0 * G.a * iu / 40.0;
                fprintf(f, "%s,%.10g,%.10g,%d,scan,%.10g,%.10g,%.10g,%.15g\n", G.name, G.a, G.g, N_IMG,
                        u, 0.0, h, phi_point(u, 0.0, h, G.a, G.g, N_IMG));
            }
        }
        // convergence en nombre d'images
        const int nimgs[6] = {1, 2, 5, 10, 20, 40};
        const double hn[3] = {0.1, 0.5, 0.9};
        const double pn[3][2] = { {0, 0}, {1, 0}, {2, 1} };
        for (int ni = 0; ni < 6; ++ni)
            for (int ih = 0; ih < 3; ++ih)
                for (int ip = 0; ip < 3; ++ip) {
                    double h = hn[ih] * G.g;
                    double u = pn[ip][0] * G.a, v = pn[ip][1] * G.a;
                    fprintf(f, "%s,%.10g,%.10g,%d,nimg,%.10g,%.10g,%.10g,%.15g\n", G.name, G.a, G.g, nimgs[ni],
                            u, v, h, phi_point(u, v, h, G.a, G.g, nimgs[ni]));
                }
    }
    fclose(f);

    // ------------------------------------------------------------------ regle de somme
    // Somme de phi sur toutes les tuiles d'un pavage carre de pas p = 2a : doit valoir 1 - h/g
    // exactement (potentiel de ponderation d'une anode entiere a 1).
    f = fopen("out/l0_sum.csv", "w");
    fprintf(f, "geo,x_over_a,y_over_a,hfrac,sum\n");
    const double sp[5][2] = { {0, 0}, {0.5, 0}, {1, 0}, {1, 1}, {0.3, 0.7} };
    for (int ig = 0; ig < 3; ++ig) {
        const Geo& G = geos[ig];
        const double p = 2.0 * G.a;
        const int L = (int)std::ceil(14.0 * G.g / p) + 1;   // phi ~ exp(-pi r / g) : 14 g -> e^-44
        for (int ip = 0; ip < 5; ++ip) {
            double x = sp[ip][0] * G.a, y = sp[ip][1] * G.a;
            for (int ih = 0; ih < 25; ++ih) {
                double hf = 0.02 + 0.04 * ih;
                double h = hf * G.g, s = 0.0;
                for (int i = -L; i <= L; ++i)
                    for (int j = -L; j <= L; ++j)
                        s += phi_point(x - i * p, y - j * p, h, G.a, G.g, N_IMG);
                fprintf(f, "%s,%.10g,%.10g,%.10g,%.15g\n", G.name, sp[ip][0], sp[ip][1], hf, s);
            }
        }
    }
    fclose(f);

    // ------------------------------------------------------------------ cas triviaux
    f = fopen("out/l0_misc.csv", "w");
    fprintf(f, "name,sim,exact\n");
    {
        const double a = geos[0].a, g = geos[0].g;
        fprintf(f, "phi(h=g)=0,%.15g,0\n", phi_point(0, 0, g, a, g, N_IMG));
        fprintf(f, "phi(h=0) centre du pad=1,%.15g,1\n", phi_point(0, 0, 0.0, a, g, N_IMG));
        fprintf(f, "phi(h=0) bord du pad=1/2,%.15g,0.5\n", phi_point(a, 0, 0.0, a, g, N_IMG));
        fprintf(f, "phi(h=0) coin du pad=1/4,%.15g,0.25\n", phi_point(a, a, 0.0, a, g, N_IMG));
        fprintf(f, "phi(h=0) hors du pad=0,%.15g,0\n", phi_point(1.5 * a, 0, 0.0, a, g, N_IMG));
        fprintf(f, "phi(h=0) hors du pad (diag)=0,%.15g,0\n", phi_point(2.0 * a, 2.0 * a, 0.0, a, g, N_IMG));
        // pad tres grand devant g : condensateur plan, phi = 1 - h/g.
        // a = 1 cm (3.3 g) : l'effet de bord vaut e^(-pi a/g) ~ 3e-5 au centre. On n'ose pas a plus grand :
        // la correction de queue de phi_point suppose le pad petit devant (2 n_img + 1) g.
        const double ab = 1.0;
        const double hb[5] = {0.1, 0.3, 0.5, 0.7, 0.9};
        for (int i = 0; i < 5; ++i) {
            char name[96];
            snprintf(name, sizeof(name), "grand pad a=1cm h=%.1fg : 1-h/g", hb[i]);
            fprintf(f, "%s,%.15g,%.15g\n", name, phi_point(0, 0, hb[i] * g, ab, g, N_IMG), 1.0 - hb[i]);
        }
        // demi-espace, au centre : angle solide d'un carre vu depuis l'axe, 4 atan(a^2 / (h sqrt(2a^2+h^2)))
        const double hh[3] = {0.5, 1.0, 2.0};
        for (int i = 0; i < 3; ++i) {
            double h = hh[i] * a;
            char name[96];
            snprintf(name, sizeof(name), "phi_halfspace centre h=%.1fa", hh[i]);
            double ex = 4.0 * std::atan(a * a / (h * std::sqrt(2.0 * a * a + h * h))) / (2.0 * PI);
            fprintf(f, "%s,%.15g,%.15g\n", name, phi_halfspace(0, 0, h, a), ex);
        }
    }
    fclose(f);

    printf("niveau 0 : CSV ecrits dans out/ (l0_phi.csv, l0_sum.csv, l0_misc.csv)\n");
    return 0;
}
