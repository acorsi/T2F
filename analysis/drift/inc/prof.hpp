#pragma once
#include <omp.h>

#ifdef T2F_PROF
struct ProfStats {
  double t_tot, t_vec, t_at, t_acc, t_out, sum_ne, n_cell_touched, n_cell_keep, sum_nb;   // secondes cumulées
  long   n_vox, n_cell, n_contrib, n_skip_sel, n_cell_wasted;
  char   pad[64];                            // évite le partage de ligne de cache
};
extern ProfStats g_prof[64];                   // un élément par thread
#define PROF(x) x
#else
#define PROF(x)
#endif
