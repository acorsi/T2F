// Ecrit dans un fichier CSV, pour quelques valeurs de d, la réponse de chaque pad
// autour d'un voxel : Psi final (charge collectée), Psi maximal, et le plus grand
// qt positif et le plus grand qt négatif (variation de Psi sur un bin de 10 ns).
// Les pads suivent la disposition de build_anode().
//
// Usage :  ./dump_patterns [fichier.csv] [x0_cm] [y0_cm]
// (le voxel est en (x0, y0) ; par défaut (0, 0))
#include "psi_table.hpp"
#include "detector.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

int main(int argc, char** argv) {
    const char* nom_fichier = (argc > 1) ? argv[1] : "patterns.csv";
    double x0 = (argc > 2) ? atof(argv[2]) : 0.0;
    double y0 = (argc > 3) ? atof(argv[3]) : 0.0;

    const double PAD_SIZE  = 0.3;
    const double INTER_PAD = 0.015;
    const double PITCH     = PAD_SIZE + INTER_PAD;

    DetectorConfig cfg;
    TableParams tp = default_table_params(cfg, PITCH);
    printf("Construction de la table...\n");
    PsiTable table(tp);
    std::vector<double> vec(table.n_tau());

    // Valeurs de d à tracer (cm)
    const int N_D = 5;
    double d_liste[N_D] = {0.5, 3.0, 8.0, 14.0, 20.0};

    // Pads dans un rayon de 3,5 pitchs autour du voxel (même construction que build_anode)
    FILE* f = fopen(nom_fichier, "w");
    if (!f) { printf("Impossible d'ecrire %s\n", nom_fichier); return 1; }
    fprintf(f, "# voxel x0 y0 (cm): %g %g\n", x0, y0);
    fprintf(f, "d,x,y,psi_final,psi_max,qt_max,qt_min\n");

    // qt = variation de Psi sur un bin de delta_t (10 ns) : c'est le signal écrit par bin.
    int pas_bin = (int)std::floor(cfg.delta_t / table.dtau() + 0.5);

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
                double psi_max = vec[0];
                for (int n = 1; n < table.n_tau(); ++n) {
                    if (vec[n] > psi_max) psi_max = vec[n];
                }
                double qt_max = 0.0, qt_min = 0.0;
                for (int n = 0; n + pas_bin < table.n_tau(); ++n) {
                    double qt = vec[n + pas_bin] - vec[n];
                    if (qt > qt_max) qt_max = qt;
                    if (qt < qt_min) qt_min = qt;
                }
                fprintf(f, "%g,%g,%g,%.6g,%.6g,%.6g,%.6g\n", d, x, y,
                        vec[table.n_tau() - 1], psi_max, qt_max, qt_min);
                ++n_pads;
            }
        }
        printf("d = %5.1f cm : %d pads\n", d, n_pads);
    }
    fclose(f);
    printf("Ecrit : %s\n", nom_fichier);
    return 0;
}
