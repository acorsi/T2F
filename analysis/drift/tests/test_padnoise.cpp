// ---------------------------------------------------------------------------
// Tests du bruit par pad dependant de la capacite (cap_noise.hpp, et son emploi dans noise / ghost / readout).
//
// Usage :  ./test_padnoise [facteur]      (defaut : 1 ; 0.3 pour un essai rapide)
//
//   N1  CapNoiseModel : ENC(0) = enc0, ENC(c_ref) = enc_ref, pente b, facteurs = 1 a c_ref, validations
//   N2  add_filtered_noise(ser_scale = 1) identique bit a bit a la version sans argument
//   N3  rms des echantillons du CAN = ENC(C) pour C = 0, 60, 93, 150 pF (chaine complete, modele a c_ref)
//   N4  GhostModel::prob_for_scale : (a) = frequence brute-force du bruit de reference multiplie par le facteur ;
//       (b) ecart avec le vrai bruit (serie seul mis a l'echelle) : approximation de forme, ecart AFFICHE
//   N5  read_pad_capacitances : fichier correct, commentaires, erreurs
//   N6  Readout : table tout a 1 = pas de table (bit a bit) ; pad absent de la table -> exception ;
//       bruit d'un pad plus grand quand sa capacite est plus grande
//
// Gaussienne de test (pas de reponse MINOS). Code retour : 0 si tout passe. C++ classique.
// ---------------------------------------------------------------------------

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "cap_noise.hpp"
#include "ghost.hpp"
#include "noise.hpp"
#include "preamp_response.hpp"
#include "readout.hpp"

static int g_fail = 0, g_ok = 0;

static void check(bool cond, const std::string& what, double value, double expected)
{
    if (cond) { ++g_ok;   std::printf("  ok   %-70s %12.6g (attendu %.6g)\n", what.c_str(), value, expected); }
    else      { ++g_fail; std::printf("  ECHEC %-69s %12.6g (attendu %.6g)\n", what.c_str(), value, expected); }
}

static bool throws_cap(const CapNoiseParams& p)
{
    try { CapNoiseModel m(p); } catch (const std::exception&) { return true; }
    return false;
}

static bool reads_fail(const std::string& path, const std::vector<int>& ids)
{
    std::vector<double> c;
    try { read_pad_capacitances(path, ids, c); } catch (const std::exception&) { return true; }
    return false;
}

static void write_file(const std::string& path, const std::string& txt)
{
    std::ofstream f(path.c_str()); f << txt;
}

int main(int argc, char* argv[])
{
    const double fac = (argc > 1) ? std::atof(argv[1]) : 1.0;
    const int n_bins = 216;
    const double dt = 10.0;

    GaussTestResponse resp(190.0);
    std::vector<double> h;
    resp.sample(dt, h);

    // ---- N1 ----
    std::printf("=== N1  CapNoiseModel ===\n");
    CapNoiseParams cp;                        // 1150 / 1400 / 93 pF
    CapNoiseModel cm(cp);
    check(std::fabs(cm.enc_total(0.0) - 1150.0) < 1e-9,   "ENC(0 pF) = enc0", cm.enc_total(0.0), 1150.0);
    check(std::fabs(cm.enc_total(93.0) - 1400.0) < 1e-9,  "ENC(93 pF) = enc_ref", cm.enc_total(93.0), 1400.0);
    const double b_exp = std::sqrt(1400.0 * 1400.0 - 1150.0 * 1150.0) / 93.0;
    check(std::fabs(cm.slope() - b_exp) < 1e-9,           "pente b (e-/pF)", cm.slope(), b_exp);
    const double e103 = std::sqrt(1150.0 * 1150.0 + b_exp * b_exp * 103.0 * 103.0);
    check(std::fabs(cm.enc_total(103.0) - e103) < 1e-9,   "ENC(103 pF)", cm.enc_total(103.0), e103);
    check(std::fabs(cm.ser_scale(93.0) - 1.0) < 1e-12 && std::fabs(cm.enc_scale(93.0) - 1.0) < 1e-12,
          "facteurs = 1 a c_ref", cm.ser_scale(93.0), 1.0);
    check(cm.enc_total(150.0) > cm.enc_total(93.0) && cm.enc_total(60.0) < cm.enc_total(93.0), "ENC croissante avec C",
          cm.enc_total(150.0), cm.enc_total(93.0));
    NoiseParams npp; npp.dt = dt; npp.adc_ratio = 2;
    cm.fill_noise_params(npp);
    check(std::fabs(npp.enc_total - 1400.0) < 1e-9, "fill_noise_params : enc_total = enc_ref", npp.enc_total, 1400.0);
    {
        CapNoiseParams q;
        q = cp; q.enc0 = 0.0;      check(throws_cap(q), "enc0 = 0 refuse", 0, 0);
        q = cp; q.c_ref_pf = 0.0;  check(throws_cap(q), "c_ref = 0 refuse", 0, 0);
        q = cp; q.enc_ref = 1000;  check(throws_cap(q), "enc_ref < enc0 refuse", 0, 0);
        q = cp; q.f_par = -1.0;    check(throws_cap(q), "fraction negative refusee", 0, 0);
    }

    NoiseModel model(npp, h);                 // modele au pad de reference

    // ---- N2 ----
    std::printf("\n=== N2  add_filtered_noise(ser_scale = 1) = version sans argument ===\n");
    {
        GaussRng r1(5), r2(5);
        std::vector<double> a(n_bins, 0.0), b(n_bins, 0.0);
        model.add_filtered_noise(r1, a);
        model.add_filtered_noise(r2, b, 1.0);
        bool same = true;
        for (int i = 0; i < n_bins; ++i) if (a[i] != b[i]) same = false;
        check(same, "identique bit a bit", same ? 1 : 0, 1);
        GaussRng r3(5), r4(5);
        std::vector<double> c(n_bins, 0.0), d(n_bins, 0.0);
        model.add_filtered_noise(r3, c, 0.0);
        model.add_filtered_noise(r4, d, 2.0);
        double vc = 0, vd = 0;
        for (int i = 0; i < n_bins; ++i) { vc += c[i] * c[i]; vd += d[i] * d[i]; }
        check(vd > vc, "ser_scale 2 > ser_scale 0 (energie)", vd, vc);
        bool thr = false;
        try { GaussRng r(1); std::vector<double> z(n_bins, 0.0); model.add_filtered_noise(r, z, -1.0); }
        catch (const std::exception&) { thr = true; }
        check(thr, "ser_scale < 0 refuse", thr ? 1 : 0, 1);
    }

    // ---- N3 ----
    std::printf("\n=== N3  rms des echantillons du CAN = ENC(C) ===\n");
    {
        const double cs[4] = { 0.0, 60.0, 93.0, 150.0 };
        for (int ic = 0; ic < 4; ++ic) {
            const double ss = cm.ser_scale(cs[ic]);
            GaussRng rng(900 + ic);
            const int N = (int)(1500 * fac);
            double s2 = 0.0; long n = 0;
            std::vector<double> bins, smp;
            for (int i = 0; i < N; ++i) {
                bins.assign(n_bins, 0.0);
                model.add_filtered_noise(rng, bins, ss);
                model.decimate(bins, 0, smp);
                model.add_adc_noise(rng, smp);
                for (size_t j = 0; j < smp.size(); ++j) { s2 += smp[j] * smp[j]; ++n; }
            }
            const double rms = std::sqrt(s2 / (double)n), ex = cm.enc_total(cs[ic]);
            char lab[120]; std::snprintf(lab, sizeof(lab), "rms a %.0f pF", cs[ic]);
            // les echantillons sont correles : tolerance 1 % (statistique ~ 0.1 %)
            check(std::fabs(rms / ex - 1.0) < 0.01, lab, rms, ex);
        }
    }

    // ---- N4 ----
    std::printf("\n=== N4  GhostModel::prob_for_scale ===\n");
    {
        const double enc = npp.enc_total;
        const double sc[3] = { 0.9, 1.036, 1.25 };
        GhostParams gp; gp.calib_trials = (int)(20000 * fac);
        GhostModel G(model, gp, 4.0 * enc, 1, n_bins);
        GaussRng rc(77); G.calibrate(rc);
        // formule approchee (avant calibrate_scales) : pour memoire, a comparer a la calibration par scale
        const double p_formule125 = G.prob_for_scale(1.25);
        { std::vector<double> sl(sc, sc + 3); G.calibrate_scales(rc, sl); }
        check(std::fabs(G.prob_for_scale(1.0) - G.prob()) < 1e-15, "scale = 1 : p inchangee", G.prob_for_scale(1.0), G.prob());
        for (int is = 0; is < 3; ++is) {
            // (a) bruit de reference x scale, test sur le seuil, sans re-ponderation
            GaussRng rb(300 + is);
            const int Nb = (int)(60000 * fac);
            long hit = 0;
            std::vector<double> bins, smp;
            for (int n = 0; n < Nb; ++n) {
                bins.assign(n_bins, 0.0);
                model.add_filtered_noise(rb, bins);
                model.decimate(bins, 0, smp);
                model.add_adc_noise(rb, smp);
                for (size_t j = 0; j < smp.size(); ++j) smp[j] *= sc[is];
                if (GhostModel::first_run(smp, 4.0 * enc, 1) >= 0) ++hit;
            }
            const double pb = (double)hit / Nb, seb = std::sqrt(pb * (1.0 - pb) / Nb);
            const double pm = G.prob_for_scale(sc[is]);
            const double tol = 4.0 * std::sqrt(seb * seb + (pm * G.prob_err() / G.prob()) * (pm * G.prob_err() / G.prob()));
            char lab[140]; std::snprintf(lab, sizeof(lab), "(a) p(scale %.2f) = brute-force +- %.1e", sc[is], tol);
            check(std::fabs(pm - pb) < tol, lab, pm, pb);
        }
        // (b) vrai modele : serie seul mis a l'echelle (ENC total x 1.25 -> C tel que enc_scale = 1.25)
        const double ctarget = std::sqrt(std::pow(1.25 * 1400.0, 2) - 1150.0 * 1150.0) / cm.slope();
        const double ss = cm.ser_scale(ctarget), es = cm.enc_scale(ctarget);
        GaussRng rb(400);
        const int Nb = (int)(60000 * fac);
        long hit = 0;
        std::vector<double> bins, smp;
        for (int n = 0; n < Nb; ++n) {
            bins.assign(n_bins, 0.0);
            model.add_filtered_noise(rb, bins, ss);
            model.decimate(bins, 0, smp);
            model.add_adc_noise(rb, smp);
            if (GhostModel::first_run(smp, 4.0 * enc, 1) >= 0) ++hit;
        }
        const double pb = (double)hit / Nb, pm = G.prob_for_scale(es);
        std::printf("  info (b) C = %.0f pF (enc_scale %.3f) : p modele = %.5f, p vrai bruit = %.5f, rapport = %.3f (formule approchee seule : %.5f)\n",
                    ctarget, es, pm, pb, pm / pb, p_formule125);
        check(pm / pb > 0.85 && pm / pb < 1.15, "(b) approximation de forme : rapport dans [0.85, 1.15]", pm / pb, 1.0);
    }

    // ---- N5 ----
    std::printf("\n=== N5  read_pad_capacitances ===\n");
    {
        std::vector<int> ids; ids.push_back(3); ids.push_back(5); ids.push_back(8);
        write_file("cap_ok.txt", "# commentaire\n3 60\n\n5 93.5   # fin de ligne\n  8\t150\n");
        std::vector<double> c;
        read_pad_capacitances("cap_ok.txt", ids, c);
        check(c.size() == 3 && c[0] == 60.0 && c[1] == 93.5 && c[2] == 150.0, "lecture, ordre = ordre des ids", c.size() == 3 ? c[1] : -1, 93.5);
        write_file("cap_b1.txt", "3 60\n5 70\n");           check(reads_fail("cap_b1.txt", ids), "pad manquant refuse", 0, 0);
        write_file("cap_b2.txt", "3 60\n5 70\n8 1\n9 1\n"); check(reads_fail("cap_b2.txt", ids), "pad inconnu refuse", 0, 0);
        write_file("cap_b3.txt", "3 60\n3 61\n5 70\n8 1\n"); check(reads_fail("cap_b3.txt", ids), "pad en double refuse", 0, 0);
        write_file("cap_b4.txt", "3 -1\n5 70\n8 1\n");      check(reads_fail("cap_b4.txt", ids), "capacite negative refusee", 0, 0);
        write_file("cap_b5.txt", "3 nan\n5 70\n8 1\n");     check(reads_fail("cap_b5.txt", ids), "capacite nan refusee", 0, 0);
        write_file("cap_b6.txt", "3 60 7\n5 70\n8 1\n");    check(reads_fail("cap_b6.txt", ids), "colonne en trop refusee", 0, 0);
        write_file("cap_b7.txt", "abc\n3 60\n5 70\n8 1\n"); check(reads_fail("cap_b7.txt", ids), "ligne illisible refusee", 0, 0);
        write_file("cap_b8.txt", "3\n5 70\n8 1\n");         check(reads_fail("cap_b8.txt", ids), "ligne sans capacite refusee", 0, 0);
        check(reads_fail("cap_absent.txt", ids), "fichier absent refuse", 0, 0);
    }

    // ---- N6 ----
    std::printf("\n=== N6  Readout avec table de bruit par pad ===\n");
    {
        ReadoutParams rp; rp.noise_on = true; rp.thr_sigma = 4.0;
        Readout R0(rp, model, n_bins), R1(rp, model, n_bins), R2(rp, model, n_bins);
        std::vector<int> ids; ids.push_back(5); ids.push_back(7);
        std::vector<double> ones(2, 1.0);
        R1.set_pad_noise(ids, ones, ones);
        std::vector<double> qa(4, 1.0e5);
        std::vector<PadCharge> H, L;
        PadCharge p; p.pad_id = 5; p.x = 0.5; p.y = -1.0; p.k0 = 40; p.qt = qa; p.q = 4.0e5;
        H.push_back(p);
        bool same = true;
        for (int s = 0; s < 20; ++s) {
            ReadoutEvent a, b;
            R0.process(H, L, 100 + s, a); R1.process(H, L, 100 + s, b);
            if (a.pads.size() != b.pads.size()) { same = false; continue; }
            for (size_t i = 0; i < a.pads.size(); ++i) {
                if (a.pads[i].samples.size() != b.pads[i].samples.size()) { same = false; continue; }
                for (size_t j = 0; j < a.pads[i].samples.size(); ++j)
                    if (a.pads[i].samples[j] != b.pads[i].samples[j]) same = false;
            }
        }
        check(same, "table tout a 1 = sans table (bit a bit, 20 evenements)", same ? 1 : 0, 1);

        // pad 5 absent de la table
        std::vector<int> only7(1, 7); std::vector<double> one(1, 1.0);
        R2.set_pad_noise(only7, one, one);
        bool thr = false;
        try { ReadoutEvent e; R2.process(H, L, 1, e); } catch (const std::exception&) { thr = true; }
        check(thr, "pad absent de la table : exception", thr ? 1 : 0, 1);

        // bruit plus grand a forte capacite : ecart quadratique signal bruite - signal propre sur le pad 5
        ReadoutParams rp_off; Readout Roff(rp_off, model, n_bins);
        double s2[2] = { 0.0, 0.0 }; long n2[2] = { 0, 0 };
        const double cc[2] = { 60.0, 150.0 };
        for (int ic = 0; ic < 2; ++ic) {
            Readout Rc(rp, model, n_bins);
            std::vector<double> ss(2, cm.ser_scale(cc[ic])), es(2, cm.enc_scale(cc[ic]));
            Rc.set_pad_noise(ids, ss, es);
            for (int s = 0; s < (int)(300 * fac) + 10; ++s) {
                ReadoutEvent a, b;
                Roff.process(H, L, 5000 + s, a); Rc.process(H, L, 5000 + s, b);
                if (a.pads.empty() || b.pads.empty()) continue;
                const ReadoutPad& pa = a.pads[0]; const ReadoutPad& pb = b.pads[0];
                for (size_t j = 0; j < pb.samples.size(); ++j) {
                    const int idx = pb.j0 + (int)j - pa.j0;
                    if (idx < 0 || idx >= (int)pa.samples.size()) continue;
                    const double d = pb.samples[j] - pa.samples[idx];
                    s2[ic] += d * d; ++n2[ic];
                }
            }
        }
        const double r60 = std::sqrt(s2[0] / n2[0]), r150 = std::sqrt(s2[1] / n2[1]);
        check(std::fabs(r60 / cm.enc_total(60.0) - 1.0) < 0.02, "Readout : rms du bruit a 60 pF", r60, cm.enc_total(60.0));
        check(std::fabs(r150 / cm.enc_total(150.0) - 1.0) < 0.02, "Readout : rms du bruit a 150 pF", r150, cm.enc_total(150.0));
    }

    std::printf("\n%d verifications, %d echec(s)\n", g_ok + g_fail, g_fail);
    return g_fail ? 1 : 0;
}
