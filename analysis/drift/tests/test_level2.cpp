// ---------------------------------------------------------------------------
// Niveau 2 : PadPlane::project_pair (cas A = drift.cfg) : un voxel, puis quelques traces.
//
// Usage :  ./test_level2 [R]        (R = table_R, 5 par defaut)
//
// Tests faits ici (aucune reference exterieure) :
//   T1  conservation : somme sur les pads de la charge finale et des bins = n_electrons (seuils a 0)
//   T2  symetries du pavage : miroir en x, en y, translation d'un pitch, translation (p/2, p) loin de y = 0
//   T3  linearite : A+B = A puis B ; n x k = k fois ; H+L = H puis L
//   T5  selection par seuil (bound_margin = 0 contre 1.3, seuils 200 et 2000, traces de 6.5 a 8 cm) :
//       pas de pad perdu, memes charges, solidite de la borne dpsi_max, gain en temps
// Fichiers ecrits pour test_level2.py (reference exacte, figures) :
//   out/l2_meta.txt, out/l2_ref.csv (T4 : bins par pad pour 6 voxels), out/l2_sumbins.csv, out/l2_bound.csv
//
// C++ classique.
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include <iostream>
#include <chrono>
#include "detector.hpp"
#include "derived_params.hpp"      // inclut pad_geometry.hpp et psi_table.hpp

struct Vox { double x, y, z, n; };

typedef std::pair<long, long> Key;
struct Dense { double q; std::vector<double> b; };
typedef std::map<Key, Dense> DenseMap;

static Key make_key(double x, double y) { return Key(std::lround(x * 1.0e5), std::lround(y * 1.0e5)); }

static void make_case_A(DetectorConfig& cfg, PadPlaneParams& pp)
{
    cfg = DetectorConfig();
    pp  = PadPlaneParams();
    cfg.dga   = 0.3;
    cfg.DT_kg = 0.02;  cfg.DT_ga = 0.04;
    cfg.DL_kg = 0.012; cfg.DL_ga = 0.016;
    cfg.ve_kg = 0.010; cfg.ve_ga = 0.003;
    pp.pad_size = 0.5; pp.inter_pad = 0.015;
    pp.r_quinconce = 10; pp.r_transition = 10; pp.r_active = 10; pp.n_rings = 0;
}

static void run_pair(const PadPlane& pl, const DetectorConfig& cfg,
                     const std::vector<Vox>& vh, const std::vector<Vox>& vl,
                     std::vector<std::vector<double> >& buf,
                     std::vector<PadCharge>& rh, std::vector<PadCharge>& rl)
{
    std::vector<double> xh, yh, zh, nh, sh, xl, yl, zl, nl, sl;
    for (size_t i = 0; i < vh.size(); ++i) {
        xh.push_back(vh[i].x); yh.push_back(vh[i].y); zh.push_back(vh[i].z); nh.push_back(vh[i].n);
        sh.push_back(cfg.sigma_T(vh[i].z));
    }
    for (size_t i = 0; i < vl.size(); ++i) {
        xl.push_back(vl[i].x); yl.push_back(vl[i].y); zl.push_back(vl[i].z); nl.push_back(vl[i].n);
        sl.push_back(cfg.sigma_T(vl[i].z));
    }
    FragTrace fh = { &xh, &yh, &zh, &nh, &sh };
    FragTrace fl = { &xl, &yl, &zl, &nl, &sl };
    pl.project_pair(fh, fl, cfg, buf, rh, rl);
}

static void to_dense(const std::vector<PadCharge>& r, int n_bins, DenseMap& m)
{
    m.clear();
    for (size_t i = 0; i < r.size(); ++i) {
        Dense d;
        d.q = r[i].q;
        d.b.assign(n_bins, 0.0);
        for (size_t j = 0; j < r[i].qt.size(); ++j) d.b[r[i].k0 + j] = r[i].qt[j];
        m[make_key(r[i].x, r[i].y)] = d;
    }
}

// union des pads ecrits dans resH et resL (un pad touche par un seul des fragments n'est que dans sa liste)
static void to_dense_union(const std::vector<PadCharge>& rh, const std::vector<PadCharge>& rl, int n_bins, DenseMap& m)
{
    DenseMap V;
    to_dense(rh, n_bins, m);
    to_dense(rl, n_bins, V);
    for (DenseMap::const_iterator it = V.begin(); it != V.end(); ++it)
        if (m.find(it->first) == m.end()) m[it->first] = it->second;
}

// ecart max |A(pad) - B(pad')| / norm sur q et sur tous les bins ; les pads absents d'un cote valent 0.
// (dx, dy) : pad' = pad + (sx * x + tx, sy * y + ty)
static double compare_maps(const DenseMap& A, const DenseMap& B, double sx, double sy, double tx, double ty,
                           int n_bins, double norm)
{
    double worst = 0.0;
    std::vector<double> zero(n_bins, 0.0);
    // chaque pad de A
    for (DenseMap::const_iterator it = A.begin(); it != A.end(); ++it) {
        const double x = it->first.first * 1.0e-5, y = it->first.second * 1.0e-5;
        DenseMap::const_iterator jt = B.find(make_key(sx * x + tx, sy * y + ty));
        const std::vector<double>& bb = (jt == B.end()) ? zero : jt->second.b;
        const double qb = (jt == B.end()) ? 0.0 : jt->second.q;
        worst = std::max(worst, std::fabs(it->second.q - qb) / norm);
        for (int k = 0; k < n_bins; ++k) worst = std::max(worst, std::fabs(it->second.b[k] - bb[k]) / norm);
    }
    // pads de B absents de A
    for (DenseMap::const_iterator jt = B.begin(); jt != B.end(); ++jt) {
        const double x = jt->first.first * 1.0e-5, y = jt->first.second * 1.0e-5;
        // inverse : pour sx, sy = +-1, x = sx * (x' - tx) etc.
        DenseMap::const_iterator it = A.find(make_key(sx * (x - tx), sy * (y - ty)));
        if (it == A.end()) {
            worst = std::max(worst, std::fabs(jt->second.q) / norm);
            for (int k = 0; k < n_bins; ++k) worst = std::max(worst, std::fabs(jt->second.b[k]) / norm);
        }
    }
    return worst;
}

static double sum_q(const DenseMap& A) { double s = 0.0; for (DenseMap::const_iterator it = A.begin(); it != A.end(); ++it) s += it->second.q; return s; }
static double sum_b(const DenseMap& A, int n_bins)
{
    double s = 0.0;
    for (DenseMap::const_iterator it = A.begin(); it != A.end(); ++it)
        for (int k = 0; k < n_bins; ++k) s += it->second.b[k];
    return s;
}

static unsigned long g_seed = 20261112UL;
static double rnd01() { g_seed = g_seed * 6364136223846793005ULL + 1442695040888963407ULL; return (double)((g_seed >> 11) & 0xFFFFFFFFFFFFFULL) / 4503599627370496.0; }

static int g_nfail = 0;
static void report(const char* name, double value, double tol)
{
    const bool ok = (value <= tol);
    if (!ok) ++g_nfail;
    printf("  %-78s %10.2e  %s\n", name, value, ok ? "OK" : "ECHEC");
}

int main(int argc, char* argv[])
{
    const int R = (argc >= 2) ? atoi(argv[1]) : 5;

    DetectorConfig cfg;
    PadPlaneParams pp;
    TableOptions   opt;
    make_case_A(cfg, pp);
    opt.R = R;

    CheckList     chk;
    DerivedParams D;
    try {
        D = derive_params(cfg, pp, opt, chk);
        chk.finish();
    } catch (const std::exception& e) {
        std::cerr << "\nArret : " << e.what() << "\n";
        return 1;
    }
    cfg.tau_tail_ns = D.tau_tail_ns;
    const int n_bins = cfg.n_bins();
    const double p = D.pitch, dkg = cfg.dkg;

    PadPlaneParams pp0 = pp; pp0.bound_margin = 0.0;       // pas de selection
    PadPlaneParams pp1 = pp; pp1.bound_margin = 1.3;       // valeur de drift.cfg
    PadPlane* plane0 = new PadPlane(pp0, cfg, D.tp, D.reach_cm);
    PadPlane* plane1 = new PadPlane(pp1, cfg, D.tp, D.reach_cm);
    const PsiTable& T = plane0->table();

    DetectorConfig cfg0 = cfg;                              // tout lire, rien rogner
    cfg0.seuil_qt_pos = 0.0; cfg0.seuil_qt_neg = 0.0; cfg0.eps_qt = 0.0;

    std::vector<std::vector<double> > buf0, buf1;
    std::vector<PadCharge> rh, rl;
    DenseMap A, B, C;
    std::vector<Vox> none;

    // meta
    {
        FILE* m = fopen("out/l2_meta.txt", "w");
        if (!m) { std::cerr << "impossible d'ecrire dans out/ (lancer depuis tests/)\n"; return 1; }
        fprintf(m, "case=A\nR=%d\na=%.12g\ng=%.12g\npitch=%.12g\ndkg=%.12g\n", R, D.a, cfg.dga, p, dkg);
        fprintf(m, "DT_kg=%.12g\nDT_ga=%.12g\nDL_kg=%.12g\nDL_ga=%.12g\nve_kg=%.12g\nve_ga=%.12g\n",
                cfg.DT_kg, cfg.DT_ga, cfg.DL_kg, cfg.DL_ga, cfg.ve_kg, cfg.ve_ga);
        fprintf(m, "delta_t=%.12g\nn_bins=%d\nreach=%.12g\ndtau=%.12g\n", cfg.delta_t, n_bins, D.reach_cm, T.dtau());
        fclose(m);
    }

    printf("=== T1 : conservation de la charge (seuils a 0, eps_qt = 0) ===\n");
    printf("  (n_electrons = 1e5 ; ecart = somme sur les pads / n_electrons - 1)\n");
    {
        // (x, y, d) : cinq voxels au coeur du plan, puis le bord (la charge hors des pads carres est perdue)
        const double pts[11][3] = {
            {0.0, 0.0, 10.0}, {0.13, 0.21, 5.0}, {0.2575, 0.2575, 12.5}, {1.7, -2.3, 15.0}, {-3.4, 0.9, 19.5},
            {0.0, 0.0, 0.5},  {4.27, -3.9, 7.3},
            {7.0, 0.0, 10.0}, {9.0, 0.0, 10.0}, {9.9, 0.0, 10.0}, {10.4, 0.0, 10.0} };
        for (int i = 0; i < 11; ++i) {
            std::vector<Vox> v(1);
            v[0].x = pts[i][0]; v[0].y = pts[i][1]; v[0].z = dkg - pts[i][2]; v[0].n = 1.0e5;
            run_pair(*plane0, cfg0, v, none, buf0, rh, rl);
            to_dense(rh, n_bins, A);
            const double eq = sum_q(A) / 1.0e5 - 1.0, eb = sum_b(A, n_bins) / 1.0e5 - 1.0;
            char name[160];
            if (i < 7) {
                snprintf(name, sizeof(name), "voxel (%.3g, %.3g) d = %.3g cm : charge finale", pts[i][0], pts[i][1], pts[i][2]);
                report(name, std::fabs(eq), 1.0e-3);
                snprintf(name, sizeof(name), "voxel (%.3g, %.3g) d = %.3g cm : somme des bins", pts[i][0], pts[i][1], pts[i][2]);
                report(name, std::fabs(eb), 1.0e-3);
            } else {
                printf("  (bord, a titre d'information) voxel (%.3g, %.3g) : somme/n - 1 = %.4f   (%d pads touches)\n",
                       pts[i][0], pts[i][1], eq, (int)rh.size());
            }
        }
    }

    printf("=== T2 : symetries du pavage (ecart max / n_electrons sur q et sur tous les bins) ===\n");
    {
        std::vector<Vox> v(1), w(1);
        v[0].x = 0.31; v[0].y = 0.77; v[0].z = dkg - 7.0; v[0].n = 1.0e5;
        run_pair(*plane0, cfg0, v, none, buf0, rh, rl); to_dense(rh, n_bins, A);
        w = v; w[0].x = -v[0].x;
        run_pair(*plane0, cfg0, w, none, buf0, rh, rl); to_dense(rh, n_bins, B);
        report("miroir x -> -x", compare_maps(A, B, -1.0, 1.0, 0.0, 0.0, n_bins, 1.0e5), 1.0e-5);
        w = v; w[0].y = -v[0].y;
        run_pair(*plane0, cfg0, w, none, buf0, rh, rl); to_dense(rh, n_bins, B);
        report("miroir y -> -y", compare_maps(A, B, 1.0, -1.0, 0.0, 0.0, n_bins, 1.0e5), 1.0e-5);
        w = v; w[0].x = v[0].x + p;
        run_pair(*plane0, cfg0, w, none, buf0, rh, rl); to_dense(rh, n_bins, B);
        report("translation (p, 0)", compare_maps(A, B, 1.0, 1.0, p, 0.0, n_bins, 1.0e5), 1.0e-5);
        // translation (p/2, p) : symetrie du pavage en briques seulement du meme cote de y = 0
        v[0].y = 3.0;
        run_pair(*plane0, cfg0, v, none, buf0, rh, rl); to_dense(rh, n_bins, A);
        w = v; w[0].x = v[0].x + 0.5 * p; w[0].y = v[0].y + p;
        run_pair(*plane0, cfg0, w, none, buf0, rh, rl); to_dense(rh, n_bins, B);
        report("translation (p/2, p), voxel a y = 3 cm", compare_maps(A, B, 1.0, 1.0, 0.5 * p, p, n_bins, 1.0e5), 1.0e-5);
    }

    printf("=== T3 : linearite ===\n");
    {
        std::vector<Vox> a(1), b(1), ab(2);
        a[0].x = 0.4;  a[0].y = 0.1;  a[0].z = dkg - 6.0;  a[0].n = 3.0e4;
        b[0].x = 0.9;  b[0].y = -0.3; b[0].z = dkg - 6.35; b[0].n = 7.0e4;     // proche en x et en temps : les signaux se recouvrent
        ab[0] = a[0]; ab[1] = b[0];
        run_pair(*plane0, cfg0, a, none, buf0, rh, rl); to_dense(rh, n_bins, A);
        run_pair(*plane0, cfg0, b, none, buf0, rh, rl); to_dense(rh, n_bins, B);
        // somme A + B
        DenseMap S = A;
        for (DenseMap::const_iterator it = B.begin(); it != B.end(); ++it) {
            DenseMap::iterator jt = S.find(it->first);
            if (jt == S.end()) S[it->first] = it->second;
            else { jt->second.q += it->second.q; for (int k = 0; k < n_bins; ++k) jt->second.b[k] += it->second.b[k]; }
        }
        run_pair(*plane0, cfg0, ab, none, buf0, rh, rl); to_dense(rh, n_bins, C);
        report("deux voxels dans H = somme des deux runs separes", compare_maps(S, C, 1.0, 1.0, 0.0, 0.0, n_bins, 1.0e5), 1.0e-9);
        // H + L : un voxel dans chaque fragment
        std::vector<Vox> la(1); la[0] = b[0];
        run_pair(*plane0, cfg0, a, la, buf0, rh, rl); to_dense_union(rh, rl, n_bins, C);
        report("un voxel dans H + un dans L = somme des deux runs separes", compare_maps(S, C, 1.0, 1.0, 0.0, 0.0, n_bins, 1.0e5), 1.0e-9);
        int nboth = 0, nsame = 0;
        for (size_t i = 0; i < rh.size(); ++i)
            for (size_t j = 0; j < rl.size(); ++j)
                if (rh[i].pad_id == rl[j].pad_id) { ++nboth; if (rh[i].q == rl[j].q && rh[i].k0 == rl[j].k0 && rh[i].qt == rl[j].qt) ++nsame; }
        printf("  (information, voxels volontairement proches : test artificiel ; le cas realiste est dans T5) pads dans resH ET resL : %d, contenu identique (somme H+L) : %d\n", nboth, nsame);
        // facteur d'echelle
        std::vector<Vox> a2 = a; a2[0].n *= 3.7;
        run_pair(*plane0, cfg0, a2, none, buf0, rh, rl); to_dense(rh, n_bins, B);
        DenseMap A3 = A;
        for (DenseMap::iterator it = A3.begin(); it != A3.end(); ++it) { it->second.q *= 3.7; for (int k = 0; k < n_bins; ++k) it->second.b[k] *= 3.7; }
        report("n_electrons x 3.7 = resultat x 3.7", compare_maps(A3, B, 1.0, 1.0, 0.0, 0.0, n_bins, 1.0e5), 1.0e-9);
    }

    // ---- T4 : donnees pour la reference exacte (Python) ----
    {
        const double vv[6][3] = { {0.30, 0.20, 10.0}, {-1.1, 2.6, 3.0}, {2.5, -0.4, 19.0}, {0.0, 0.0, 0.6}, {4.27, -3.9, 7.3}, {0.2575, 0.2575, 12.5} };
        FILE* fr = fopen("out/l2_ref.csv", "w");
        FILE* fs = fopen("out/l2_sumbins.csv", "w");
        fprintf(fr, "vox,x,y,d,pad_x,pad_y,k,q\n");
        fprintf(fs, "vox,d,k,qsum\n");
        for (int i = 0; i < 6; ++i) {
            std::vector<Vox> v(1);
            v[0].x = vv[i][0]; v[0].y = vv[i][1]; v[0].z = dkg - vv[i][2]; v[0].n = 1.0;      // n = 1 : on lit directement Psi
            run_pair(*plane0, cfg0, v, none, buf0, rh, rl);
            std::vector<double> sb(n_bins, 0.0);
            for (size_t m = 0; m < rh.size(); ++m) {
                const PadCharge& pc = rh[m];
                for (size_t j = 0; j < pc.qt.size(); ++j) sb[pc.k0 + j] += pc.qt[j];
                if (std::fabs(pc.x - v[0].x) <= 1.2 && std::fabs(pc.y - v[0].y) <= 1.2)
                    for (size_t j = 0; j < pc.qt.size(); ++j)
                        fprintf(fr, "%d,%.8g,%.8g,%.8g,%.8g,%.8g,%d,%.10g\n", i, v[0].x, v[0].y, vv[i][2], pc.x, pc.y, pc.k0 + (int)j, pc.qt[j]);
            }
            for (int k = 0; k < n_bins; ++k) if (sb[k] != 0.0) fprintf(fs, "%d,%.8g,%d,%.10g\n", i, vv[i][2], k, sb[k]);
        }
        fclose(fr); fclose(fs);
    }

    // ---- T5 : selection par seuil ----
    printf("=== T5 : selection par seuil, bound_margin = 0 (tout calcule) contre 1.3 ; traces de 8 cm (H) et 6.5 cm (L) ===\n");
    {
        const int NTR = 24;
        const double thrs[2] = { 200.0, 2000.0 };
        long miss[2] = { 0, 0 }, extra[2] = { 0, 0 }, nwritten[2] = { 0, 0 };
        long n_all = 0;                                       // pads touches (seuil 0), union H et L
        long n_resH[2] = { 0, 0 }, n_resL[2] = { 0, 0 }, n_both[2] = { 0, 0 }, n_both_Hlow[2] = { 0, 0 }, n_both_Llow[2] = { 0, 0 };
        double maxdiff[2] = { 0.0, 0.0 };
        double t_sel0[2] = { 0.0, 0.0 }, t_sel1[2] = { 0.0, 0.0 };
        double min_ratio = 1.0e30;                    // borne(marge 1) / max |qt| reel, sur les pads dont le max reel > 1 electron
        long n_unsound1 = 0, n_unsound13 = 0, n_ratio = 0;
        FILE* fb = fopen("out/l2_bound.csv", "w");
        fprintf(fb, "track,pad_x,pad_y,actual,bound\n");
        std::vector<double> ratios;

        for (int tr = 0; tr < NTR; ++tr) {
            // trace : vertex a y = 0, H vers +y et L vers -y (dos a dos), angle jusqu'a 0.5 rad
            const double x0 = (2.0 * rnd01() - 1.0) * 1.0, z0 = 4.0 + 12.0 * rnd01();
            const double th = 0.5 * rnd01(), ph = 2.0 * 3.14159265358979 * rnd01();
            const double dx = std::sin(th) * std::cos(ph), dy = std::cos(th), dz = std::sin(th) * std::sin(ph);
            std::vector<Vox> vh, vl;
            for (int j = 0; j < 160; ++j) { Vox v; double s = 0.05 * (j + 0.5); v.x = x0 + dx * s; v.y = dy * s;  v.z = z0 + dz * s; v.n = 14000.0; vh.push_back(v); }
            for (int j = 0; j < 130; ++j) { Vox v; double s = 0.05 * (j + 0.5); v.x = x0 - dx * s; v.y = -dy * s; v.z = z0 - dz * s; v.n = 10000.0; vl.push_back(v); }

            // reference non rognee : tout lire
            run_pair(*plane0, cfg0, vh, vl, buf0, rh, rl);
            to_dense_union(rh, rl, n_bins, A);
            n_all += (long)A.size();
            // signaux de chaque fragment seul (tout lire) : pour savoir si un pad ecrit dans les deux listes recoit vraiment les deux
            DenseMap AH, AL;
            {
                std::vector<PadCharge> r1, r2;
                run_pair(*plane0, cfg0, vh, none, buf0, r1, r2); to_dense(r1, n_bins, AH);
                run_pair(*plane0, cfg0, none, vl, buf0, r1, r2); to_dense(r2, n_bins, AL);
            }
            // borne indépendante : sum_voxels n * dpsi_max(u, v, kd), marge 1
            std::map<Key, double> bound;
            const std::vector<Pad>& pads = plane0->pads();
            for (int f = 0; f < 2; ++f) {
                const std::vector<Vox>& vx = (f == 0) ? vh : vl;
                for (size_t i = 0; i < vx.size(); ++i) {
                    const int kd = T.d_cell(cfg.drift_distance(vx[i].z));
                    for (size_t q = 0; q < pads.size(); ++q) {
                        const double u = vx[i].x - pads[q].x, v = vx[i].y - pads[q].y;
                        if (std::fabs(u) > D.reach_cm || std::fabs(v) > D.reach_cm) continue;
                        bound[make_key(pads[q].x, pads[q].y)] += vx[i].n * T.dpsi_max(u, v, kd);
                    }
                }
            }
            for (std::map<Key, double>::const_iterator it = bound.begin(); it != bound.end(); ++it) {
                double actual = 0.0;
                DenseMap::const_iterator jt = A.find(it->first);
                if (jt != A.end()) for (int k = 0; k < n_bins; ++k) actual = std::max(actual, std::fabs(jt->second.b[k]));
                fprintf(fb, "%d,%.6g,%.6g,%.8g,%.8g\n", tr, it->first.first * 1.0e-5, it->first.second * 1.0e-5, actual, it->second);
                if (actual > 1.0) {
                    const double r = it->second / actual;
                    ratios.push_back(r);
                    ++n_ratio;
                    if (r < min_ratio) min_ratio = r;
                    if (r < 1.0) ++n_unsound1;
                    if (r * 1.3 < 1.0) ++n_unsound13;
                }
            }

            for (int it = 0; it < 2; ++it) {
                DetectorConfig cf = cfg;
                cf.seuil_qt_pos = thrs[it]; cf.seuil_qt_neg = -thrs[it]; cf.eps_qt = 10.0;
                std::vector<PadCharge> h0, l0, h1, l1;
                std::chrono::steady_clock::time_point c0 = std::chrono::steady_clock::now();
                run_pair(*plane0, cf, vh, vl, buf0, h0, l0);
                std::chrono::steady_clock::time_point c1 = std::chrono::steady_clock::now();
                run_pair(*plane1, cf, vh, vl, buf1, h1, l1);
                std::chrono::steady_clock::time_point c2 = std::chrono::steady_clock::now();
                t_sel0[it] += std::chrono::duration<double>(c1 - c0).count();
                t_sel1[it] += std::chrono::duration<double>(c2 - c1).count();
                n_resH[it] += (long)h0.size();
                n_resL[it] += (long)l0.size();
                for (size_t i = 0; i < h0.size(); ++i)
                    for (size_t j = 0; j < l0.size(); ++j)
                        if (h0[i].pad_id == l0[j].pad_id) {
                            ++n_both[it];
                            double mh = 0.0, ml = 0.0;
                            DenseMap::const_iterator ah = AH.find(make_key(h0[i].x, h0[i].y));
                            DenseMap::const_iterator al = AL.find(make_key(h0[i].x, h0[i].y));
                            if (ah != AH.end()) for (int k = 0; k < n_bins; ++k) mh = std::max(mh, std::fabs(ah->second.b[k]));
                            if (al != AL.end()) for (int k = 0; k < n_bins; ++k) ml = std::max(ml, std::fabs(al->second.b[k]));
                            if (mh < thrs[it]) ++n_both_Hlow[it];
                            if (ml < thrs[it]) ++n_both_Llow[it];
                        }
                DenseMap X0, X1;
                to_dense_union(h0, l0, n_bins, X0);
                to_dense_union(h1, l1, n_bins, X1);
                nwritten[it] += (long)X0.size();
                for (DenseMap::const_iterator a0 = X0.begin(); a0 != X0.end(); ++a0) {
                    DenseMap::const_iterator a1 = X1.find(a0->first);
                    if (a1 == X1.end()) { ++miss[it]; continue; }
                    maxdiff[it] = std::max(maxdiff[it], std::fabs(a0->second.q - a1->second.q));
                    for (int k = 0; k < n_bins; ++k) maxdiff[it] = std::max(maxdiff[it], std::fabs(a0->second.b[k] - a1->second.b[k]));
                }
                for (DenseMap::const_iterator a1 = X1.begin(); a1 != X1.end(); ++a1) if (X0.find(a1->first) == X0.end()) ++extra[it];
            }
        }
        fclose(fb);

        printf("  pads touches par la trace (seuil 0, H ou L) : %.0f par trace en moyenne ; ecrits avec seuil 200 : %.0f ; avec seuil 2000 : %.0f\n",
               (double)n_all / NTR, (double)nwritten[0] / NTR, (double)nwritten[1] / NTR);
        for (int it = 0; it < 2; ++it)
            printf("  seuil %4.0f e- : par trace, %.0f pads dans resH, %.0f dans resL, %.0f dans les DEUX (dont %.0f ou H seul resterait sous le seuil, %.0f ou L seul resterait sous le seuil)\n",
                   thrs[it], (double)n_resH[it] / NTR, (double)n_resL[it] / NTR, (double)n_both[it] / NTR, (double)n_both_Hlow[it] / NTR, (double)n_both_Llow[it] / NTR);
        for (int it = 0; it < 2; ++it) {
            printf("  seuil %4.0f e- : %ld pads ecrits (marge 0), %ld perdus avec marge 1.3, %ld en trop, ecart max des charges communes %.2e e-\n",
                   thrs[it], nwritten[it], miss[it], extra[it], maxdiff[it]);
            printf("                  temps : %.2f s sans selection, %.2f s avec (gain x %.2f)\n", t_sel0[it], t_sel1[it], t_sel0[it] / t_sel1[it]);
            char name[120];
            snprintf(name, sizeof(name), "seuil %.0f : pads perdus par la selection", thrs[it]);
            report(name, (double)miss[it], 0.0);
            snprintf(name, sizeof(name), "seuil %.0f : ecart max des charges des pads communs (electrons)", thrs[it]);
            report(name, maxdiff[it], 1.0e-6);
        }
        std::vector<double> rs = ratios;
        // mediane du rapport borne/reel
        for (size_t i = 0; i < rs.size(); ++i) for (size_t j = i + 1; j < rs.size(); ++j) if (rs[j] < rs[i]) { double t = rs[i]; rs[i] = rs[j]; rs[j] = t; }
        printf("  borne dpsi_max (marge 1) / max |qt| reel sur %ld pads (max reel > 1 e-) : min %.3f, mediane %.3f\n",
               n_ratio, min_ratio, rs.empty() ? 0.0 : rs[rs.size() / 2]);
        printf("  pads ou borne(marge 1) < reel : %ld ; ou borne x 1.3 < reel : %ld\n", n_unsound1, n_unsound13);
        report("borne x 1.3 < reel (nombre de pads)", (double)n_unsound13, 0.0);
    }

    printf("\nBILAN NIVEAU 2 (partie C++) : %s\n", g_nfail == 0 ? "tout est OK" : "ATTENTION : au moins un test echoue");
    delete plane0;
    delete plane1;
    return g_nfail == 0 ? 0 : 1;
}
