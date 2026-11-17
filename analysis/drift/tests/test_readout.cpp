// ---------------------------------------------------------------------------
// Tests du module de lecture (readout.hpp / readout.cpp) : liste unique de pads, flag, fenetre,
// seuil, appariement des runs avec et sans bruit.
//
// Usage :  ./test_readout [n_essais]      (defaut : 3000)
//
//   R1  liste unique : fusion de H et L, flags 1 / 2 / 3, ordre par pad_id
//   R2  sans bruit : les echantillons ecrits = h (*) signal, echantillonne a la bonne phase
//   R3  la phase est la meme avec et sans bruit (meme graine), et couvre 0 .. adc_ratio-1
//   R4  fenetre ecrite : bornes attendues, identiques avec et sans bruit
//   R5  seuil de lecture : un gros signal passe toujours, un signal nul ne passe presque jamais
//   R6  (run avec bruit) - (run sans bruit) sur le meme pad = bruit pur, de rms enc_total
//   R7  reproductibilite ; R8 erreurs
//
// Aucune reponse MINOS : gaussienne de test. Code retour : 0 si tout passe. C++ classique.
// ---------------------------------------------------------------------------

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "noise.hpp"
#include "preamp_response.hpp"
#include "readout.hpp"

static int g_fail = 0, g_ok = 0;

static void check(bool cond, const std::string& what, double value, double expected)
{
    if (cond) { ++g_ok;   std::printf("  ok   %-70s %12.5g (attendu %.5g)\n", what.c_str(), value, expected); }
    else      { ++g_fail; std::printf("  ECHEC %-69s %12.5g (attendu %.5g)\n", what.c_str(), value, expected); }
}

static PadCharge make_pad(int id, int k0, const std::vector<double>& qt)
{
    PadCharge p;
    p.pad_id = id;
    p.x = 0.1 * id; p.y = -0.2 * id;
    p.q = 0.0;
    for (size_t i = 0; i < qt.size(); ++i) p.q += qt[i];
    p.k0 = k0;
    p.qt = qt;
    return p;
}

static const ReadoutPad* find_pad(const ReadoutEvent& e, int id)
{
    for (size_t i = 0; i < e.pads.size(); ++i) if (e.pads[i].pad_id == id) return &e.pads[i];
    return 0;
}

int main(int argc, char* argv[])
{
    const int n_try = (argc > 1) ? std::atoi(argv[1]) : 3000;
    const int n_bins = 216;
    const double dt = 10.0;

    GaussTestResponse resp(190.0);
    std::vector<double> h;
    resp.sample(dt, h);
    NoiseParams np; np.dt = dt; np.adc_ratio = 2; np.enc_total = 2000.0;
    NoiseModel model(np, h);

    ReadoutParams rp_off;                       // sans bruit
    ReadoutParams rp_on; rp_on.noise_on = true; // avec bruit, seuil 4 x ENC, 1 echantillon
    Readout R_off(rp_off, model, n_bins), R_on(rp_on, model, n_bins);

    std::printf("test_readout : %d essais ; seuil de lecture = %g e-, queue de h = %d bins, marge = %d bins\n\n",
                n_try, R_on.threshold(), R_on.tail_bins(), R_on.margin_bins());

    // ---- evenement de reference : pads 5, 7 (H), 7, 9 (L) ----
    std::vector<double> qa(4), qb(6);
    for (int i = 0; i < 4; ++i) qa[i] = 1.0e5;                 // 4 bins de 1e5 e-
    for (int i = 0; i < 6; ++i) qb[i] = 5.0e4 + 1.0e3 * i;
    std::vector<PadCharge> H, L;
    H.push_back(make_pad(7, 60, qb));      // dans l'ordre inverse expres : 7 puis 5
    H.push_back(make_pad(5, 40, qa));
    L.push_back(make_pad(7, 60, qb));
    L.push_back(make_pad(9, 100, qa));

    std::printf("R1 liste unique et flags\n");
    {
        ReadoutEvent e;
        R_off.process(H, L, 777, e);
        check(e.pads.size() == 3, "R1 3 pads distincts", (double)e.pads.size(), 3.0);
        if (e.pads.size() == 3) {
            check(e.pads[0].pad_id == 5 && e.pads[1].pad_id == 7 && e.pads[2].pad_id == 9,
                  "R1 ordre par pad_id croissant", (double)e.pads[0].pad_id, 5.0);
            check(e.pads[0].flag == 1, "R1 pad 5 : H seul (flag 1)", (double)e.pads[0].flag, 1.0);
            check(e.pads[1].flag == 3, "R1 pad 7 : H et L (flag 3)", (double)e.pads[1].flag, 3.0);
            check(e.pads[2].flag == 2, "R1 pad 9 : L seul (flag 2)", (double)e.pads[2].flag, 2.0);
            double qsum = 0.0; for (size_t i = 0; i < qb.size(); ++i) qsum += qb[i];
            check(std::fabs(e.pads[1].q - qsum) < 1e-6, "R1 pad 7 : charge vraie = somme, comptee une fois", e.pads[1].q, qsum);
        }
        check(e.n_ghost == 0, "R1 n_ghost = 0 (etape C : fantomes non encore codes)", (double)e.n_ghost, 0.0);
    }

    std::printf("\nR2 sans bruit : echantillons = h (*) signal a la phase tiree\n");
    {
        ReadoutEvent e;
        R_off.process(H, L, 4242, e);
        const ReadoutPad* p = find_pad(e, 5);
        std::vector<double> sig(R_off.grid_bins(), 0.0), shaped;
        for (size_t k = 0; k < qa.size(); ++k) sig[40 + k] = qa[k];
        model.shape(sig, R_off.grid_bins(), shaped);
        double dmax = 0.0;
        if (p) for (size_t i = 0; i < p->samples.size(); ++i) {
            const int bin = e.phase + (p->j0 + (int)i) * np.adc_ratio;
            const double d = std::fabs(p->samples[i] - shaped[bin]);
            if (d > dmax) dmax = d;
        }
        check(p != 0 && dmax < 1e-9, "R2 ecart max aux echantillons ecrits", dmax, 0.0);
    }

    std::printf("\nR3 phase\n");
    {
        int seen[2] = {0, 0};
        bool same = true;
        for (int s = 0; s < 400; ++s) {
            ReadoutEvent a, b;
            R_off.process(H, L, 1000 + s, a);
            R_on.process(H, L, 1000 + s, b);
            if (a.phase != b.phase) same = false;
            if (a.phase >= 0 && a.phase < 2) ++seen[a.phase];
        }
        check(same, "R3 meme phase avec et sans bruit, meme graine", same ? 1.0 : 0.0, 1.0);
        check(seen[0] > 150 && seen[1] > 150, "R3 les deux phases sont tirees (~50 % chacune)", (double)seen[0], 200.0);
    }

    std::printf("\nR4 fenetre ecrite (pad 5 : k0 = 40, 4 bins)\n");
    {
        ReadoutEvent a, b;
        R_off.process(H, L, 31, a);
        R_on.process(H, L, 31, b);
        const ReadoutPad* pa = find_pad(a, 5);
        const ReadoutPad* pb = find_pad(b, 5);
        const int first = 40 - R_on.margin_bins();
        const int last  = 40 + 4 - 1 + R_on.tail_bins() + R_on.margin_bins();
        const int ph = a.phase;
        const int j0 = (first - ph + 1) / 2;                       // ceil, first - ph >= 0 ici
        const int j1 = (last - ph) / 2;                            // floor
        check(pa != 0 && pa->j0 == j0, "R4 premier echantillon ecrit", pa ? (double)pa->j0 : -1.0, (double)j0);
        check(pa != 0 && (int)pa->samples.size() == j1 - j0 + 1, "R4 nombre d'echantillons ecrits",
              pa ? (double)pa->samples.size() : -1.0, (double)(j1 - j0 + 1));
        check(pa != 0 && pb != 0 && pa->j0 == pb->j0 && pa->samples.size() == pb->samples.size(),
              "R4 meme fenetre avec et sans bruit", pb ? (double)pb->samples.size() : -1.0, pa ? (double)pa->samples.size() : -1.0);
    }

    std::printf("\nR5 seuil de lecture (avec bruit)\n");
    {
        int pass_big = 0, pass_zero = 0;
        std::vector<PadCharge> Hz, Lz, Hb, Lb;
        std::vector<double> z(5, 0.0);
        z[2] = 11.0;                                  // signal de 11 e- : bien sous le bruit
        Hz.push_back(make_pad(3, 50, z));
        Hb.push_back(make_pad(3, 50, qa));
        const int n = 500;
        for (int s = 0; s < n; ++s) {
            ReadoutEvent e;
            R_on.process(Hb, Lb, 9000 + s, e); if (e.pads.size() == 1) ++pass_big;
            R_on.process(Hz, Lz, 9000 + s, e); if (e.pads.size() == 1) ++pass_zero;
        }
        check(pass_big == n, "R5 signal de 4e5 e- : toujours lu", (double)pass_big, (double)n);
        check(pass_zero < 0.02 * n, "R5 signal de 11 e- : lu dans moins de 2 % des cas (bruit seul)", (double)pass_zero, 0.0);
        // sans bruit : pas de seuil de lecture, le pad faible est garde
        ReadoutEvent e0;
        R_off.process(Hz, Lz, 1, e0);
        check(e0.pads.size() == 1, "R5 sans bruit : pas de seuil de lecture (liste = presection)", (double)e0.pads.size(), 1.0);
    }

    std::printf("\nR6 difference (avec bruit) - (sans bruit) = bruit pur de rms enc_total\n");
    {
        std::vector<PadCharge> Hb, Lb;
        Hb.push_back(make_pad(3, 50, qa));
        double s2 = 0.0; long n = 0; bool same_win = true;
        for (int s = 0; s < n_try; ++s) {
            ReadoutEvent a, b;
            R_off.process(Hb, Lb, 50000 + s, a);
            R_on.process(Hb, Lb, 50000 + s, b);
            if (a.pads.size() != 1 || b.pads.size() != 1) { same_win = false; continue; }
            if (a.pads[0].j0 != b.pads[0].j0 || a.pads[0].samples.size() != b.pads[0].samples.size()) { same_win = false; continue; }
            for (size_t i = 0; i < a.pads[0].samples.size(); ++i) {
                const double d = b.pads[0].samples[i] - a.pads[0].samples[i];
                s2 += d * d; ++n;
            }
        }
        const double rms = std::sqrt(s2 / (double)n);
        check(same_win, "R6 le pad est lu avec la meme fenetre dans les deux runs", same_win ? 1.0 : 0.0, 1.0);
        check(std::fabs(rms - 2000.0) < 0.02 * 2000.0, "R6 rms de la difference = enc_total (2 %)", rms, 2000.0);
    }

    std::printf("\nR7 reproductibilite, R8 erreurs\n");
    {
        ReadoutEvent a, b, c;
        R_on.process(H, L, 5, a);
        R_on.process(H, L, 5, b);
        R_on.process(H, L, 6, c);
        bool same = (a.pads.size() == b.pads.size()), differ = false;
        for (size_t i = 0; same && i < a.pads.size(); ++i) {
            if (a.pads[i].samples != b.pads[i].samples) same = false;
        }
        if (a.pads.size() != c.pads.size()) differ = true;
        for (size_t i = 0; !differ && i < a.pads.size() && i < c.pads.size(); ++i)
            if (a.pads[i].samples != c.pads[i].samples) differ = true;
        check(same, "R7 meme graine : meme evenement", same ? 1.0 : 0.0, 1.0);
        check(differ, "R7 autre graine : autre evenement", differ ? 1.0 : 0.0, 1.0);

        bool thrown = false;
        std::vector<PadCharge> Hx, Lx;
        Hx.push_back(make_pad(1, n_bins - 2, qa));        // depasse la fenetre
        try { ReadoutEvent e; R_off.process(Hx, Lx, 1, e); } catch (const std::exception&) { thrown = true; }
        check(thrown, "R8 signal hors de la fenetre de bins : refuse", thrown ? 1.0 : 0.0, 1.0);

        thrown = false;
        ReadoutParams bad; bad.n_consec = 0;
        try { Readout Rb(bad, model, n_bins); } catch (const std::exception&) { thrown = true; }
        check(thrown, "R8 read_n_consec = 0 : refuse", thrown ? 1.0 : 0.0, 1.0);
    }

    std::printf("\nR9 pad dont le signal finit au dernier bin de la grille physique : fenetre complete, non tronquee\n");
    {
        std::vector<double> q4(qa.begin(), qa.begin() + 4);
        const int k0 = n_bins - 4;
        std::vector<PadCharge> Hx, Lx;
        Hx.push_back(make_pad(9, k0, q4));
        ReadoutEvent e;
        R_off.process(Hx, Lx, 77, e);
        const ReadoutPad* p = find_pad(e, 9);
        const int R = np.adc_ratio;
        const int last = k0 + 4 - 1 + R_off.tail_bins() + R_off.margin_bins();   // > n_bins - 1
        const int j1 = (last - e.phase) / R;
        check(p != 0 && last > n_bins - 1, "R9 la fenetre voulue depasse la grille physique", (double)last, (double)(n_bins - 1));
        check(p != 0 && p->j0 + (int)p->samples.size() - 1 == j1, "R9 dernier echantillon ecrit = fin voulue de la fenetre",
              p ? (double)(p->j0 + (int)p->samples.size() - 1) : -1.0, (double)j1);
        std::vector<double> sig(R_off.grid_bins(), 0.0), shaped;
        for (int k = 0; k < 4; ++k) sig[k0 + k] = q4[k];
        model.shape(sig, R_off.grid_bins(), shaped);
        double dmax = 0.0;
        if (p) for (size_t i = 0; i < p->samples.size(); ++i) {
            const double d = std::fabs(p->samples[i] - shaped[e.phase + (p->j0 + (int)i) * R]);
            if (d > dmax) dmax = d;
        }
        check(dmax < 1.0e-9, "R9 echantillons = convolution complete h (*) signal", dmax, 0.0);
        // apres la fin de la fenetre, la sortie vraie est nulle (h est fini) : rien d'important n'est perdu
        double rest = 0.0;
        for (int b = last + 1; b < R_off.grid_bins(); ++b) rest = std::max(rest, std::fabs(shaped[b]));
        check(rest < 1.0e-9, "R9 sortie vraie nulle au-dela de la fenetre (h fini + marge)", rest, 0.0);
    }

    std::printf("\nR10 seuil de lecture teste sur la seule grille physique\n");
    {
        // un pad dont le signal vrai arrive au tout dernier bin physique : son signal mis en forme est quasi
        // entierement dans la queue de h, au-dela de n_bins. Il ne doit pas etre lu (sauf bruit, tres rare).
        // Le meme signal 40 bins plus tot est dans la grille physique : lu a chaque fois.
        std::vector<double> q1(1, 1.0e6);
        int late_read = 0, early_read = 0;
        const int NT = 300;
        for (int s = 0; s < NT; ++s) {
            std::vector<PadCharge> Hl, Hk, Lx;
            Hl.push_back(make_pad(9, n_bins - 1, q1));
            Hk.push_back(make_pad(9, n_bins - 41, q1));
            ReadoutEvent el, ek;
            R_on.process(Hl, Lx, 5000 + s, el);
            R_on.process(Hk, Lx, 5000 + s, ek);
            if (find_pad(el, 9) != 0) ++late_read;
            if (find_pad(ek, 9) != 0) ++early_read;
        }
        check(early_read == NT, "R10 signal dans la grille physique : toujours lu", (double)early_read, (double)NT);
        check(late_read <= 3, "R10 signal dans la queue de h seulement : non lu (hors bruit seul)", (double)late_read, 0.0);
    }

    std::printf("\n%d verifications reussies, %d echecs\n", g_ok, g_fail);
    return g_fail == 0 ? 0 : 1;
}
