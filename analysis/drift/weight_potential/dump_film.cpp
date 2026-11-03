// Ecrit dans un fichier CSV la courbe Psi(tau) complete de chaque pad autour d'un voxel,
// pour quelques valeurs de d. Sert a faire un film (plot_film.py).
//
// Usage :  ./dump_film [fichier.csv] [x0_cm] [y0_cm]
// (le voxel est en (x0, y0) ; par defaut (0, 0.4725) = centre d'un pad)
//
// Format : ligne 1  "# voxel x0 y0 (cm): x0 y0"
//          ligne 2  "# tau0_ns dtau_ns n_tau : tau0 dtau n_tau"
//          ligne 3  d,x,y,psi_0,psi_1,...,psi_(n_tau-1)     (une ligne par pad et par d)
#include "psi_table.hpp"
#include "detector.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

int main(int argc, char** argv) {
    const char* nom_fichier = (argc > 1) ? argv[1] : "film.csv";
    double x0 = (argc > 2) ? atof(argv[2]) : 0.0;
    double y0 = (argc > 3) ? atof(argv[3]) : 0.4725;

    const double PAD_SIZE  = 0.3;
    const double INTER_PAD = 0.015;
    const double PITCH     = PAD_SIZE + INTER_PAD;

    DetectorConfig cfg;
    TableParams tp = default_table_params(cfg, PITCH);
    printf("Construction de la table...\n");
    PsiTable table(tp);
    const int NT = table.n_tau();
    std::vector<double> vec(NT);

    const int N_D = 5;
    double d_liste[N_D] = {0.5, 3.0, 8.0, 14.0, 20.0};

    FILE* f = fopen(nom_fichier, "w");
    if (!f) { printf("Impossible d'ecrire %s\n", nom_fichier); return 1; }
    fprintf(f, "# voxel x0 y0 (cm): %g %g\n", x0, y0);
    fprintf(f, "# tau0_ns dtau_ns n_tau : %.6g %.6g %d\n", table.tau_min(), table.dtau(), NT);

    const double R_MAX = 3.5 * PITCH;
    int n_rows = 12;
    for (int id = 0; id < N_D; ++id) {
        double d = d_liste[id];
        int n_pads = 0;
        for (int row = -n_rows; row <= n_rows; ++row) {
            if (row == 0) continue;
            double y_base = row * PITCH;
            double y = y_base + ((y_base >= 0) ? -PITCH / 2.0 : PITCH / 2.0);
            double x_offset = (row % 2 != 0) ? PITCH / 2.0 : 0.0;
            for (int col = -n_rows; col <= n_rows; ++col) {
                double x = col * PITCH + x_offset;
                double u = x0 - x, v = y0 - y;
                if (fabs(u) > R_MAX || fabs(v) > R_MAX) continue;

                table.psi_vector(u, v, d, &vec[0]);
                fprintf(f, "%g,%g,%g", d, x, y);
                for (int n = 0; n < NT; ++n) fprintf(f, ",%.6g", vec[n]);
                fprintf(f, "\n");
                ++n_pads;
            }
        }
        printf("d = %5.1f cm : %d pads\n", d, n_pads);
    }
    fclose(f);
    printf("Ecrit : %s\n", nom_fichier);
    return 0;
}
