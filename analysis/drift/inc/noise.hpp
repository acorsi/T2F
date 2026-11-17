#pragma once

// ---------------------------------------------------------------------------
// Bruit electronique : generateur, parametres, modele.
//
// Chaine modelisee (voir la discussion du 13-14/11/2026) :
//
//   courant d'entree (charge par bin de dt) ---> h (preamplificateur) ---> echantillonnage CAN
//        |  + bruit parallele (blanc par bin)                                 + bruit CAN (blanc,
//        |  + bruit serie (blanc sur la charge CUMULEE, donc par bin la          par echantillon)
//        |    difference n[k] - n[k-1])
//
// Normalisation (convention ENC) : une charge de Q electrons injectee d'un coup donne un
// pic de sortie Q (h est normalisee a 1 au pic). L'ENC est le bruit rms de sortie, exprime
// en electrons equivalents d'entree. Pour chaque composante :
//   parallele : sortie rms = ENC_par,  donc sigma_par = ENC_par / sqrt( sum_j h_j^2 )          (par bin)
//   serie     : sortie rms = ENC_ser,  donc sigma_ser = ENC_ser / sqrt( sum_j (h_j - h_(j-1))^2 )
//               (sur la charge cumulee : par bin de dt, courant = difference des tirages)
//   CAN       : bruit blanc de sortie, sigma_can = ENC_can (par echantillon CAN)
//   ENC_total^2 = ENC_par^2 + ENC_ser^2 + ENC_can^2, avec ENC_x^2 = f_x * ENC_total^2.
//
// Le bruit parallele et serie traversent h : leur somme en sortie est un processus
// stationnaire (les tirages sont faits sur une fenetre elargie vers les temps negatifs).
// Le bruit du CAN n'est ajoute que sur les echantillons CAN (apres decimation).
//
// Aleatoire : mt19937_64 (suite standardisee) + Box-Muller maison : les tirages ne dependent
// pas de la bibliotheque standard (std::normal_distribution n'est pas reproductible).
//
// C++ classique.
// ---------------------------------------------------------------------------

#include <stdint.h>
#include <string>
#include <vector>
#include <random>

// ---------------------------------------------------------------------------
class GaussRng {
public:
    explicit GaussRng(uint64_t seed = 1) { reseed(seed); }
    void reseed(uint64_t seed) { m_eng.seed(seed); m_has = false; m_spare = 0.0; }

    double uniform();            // loi uniforme sur ]0,1[ (jamais 0 ni 1)
    double normal();             // loi normale N(0,1), Box-Muller
    int    uniform_int(int n);   // entier uniforme dans 0 .. n-1

private:
    std::mt19937_64 m_eng;
    bool   m_has;
    double m_spare;
};

// ---------------------------------------------------------------------------
struct NoiseParams {
    double enc_total;    // e- rms (entree) ; 0 : pas de bruit
    double f_par;        // fractions d'energie (normalisees en interne : seul le rapport compte)
    double f_ser;
    double f_can;
    double dt;           // pas de simulation (ns) : celui de h et des bins
    int    adc_ratio;    // periode du CAN / dt (entier >= 1)

    NoiseParams() : enc_total(2000.0), f_par(0.475), f_ser(0.475), f_can(0.05),
                    dt(10.0), adc_ratio(2) {}
};

// ---------------------------------------------------------------------------
class NoiseModel {
public:
    // h : reponse echantillonnee au pas p.dt (voir preamp_response.hpp). Verifie : max(h) = 1
    // a 1e-6 pres et fin de tableau negligeable (|h| < 1e-3), sinon exception.
    NoiseModel(const NoiseParams& p, const std::vector<double>& h);

    // --- grandeurs calculees ---
    double enc_par() const { return m_enc_par; }
    double enc_ser() const { return m_enc_ser; }
    double enc_can() const { return m_enc_can; }
    double sigma_par() const { return m_sig_par; }   // e- par bin, sur la charge par bin
    double sigma_ser() const { return m_sig_ser; }   // e- par bin, sur la charge cumulee
    double sigma_can() const { return m_enc_can; }   // e- par echantillon CAN (sortie)
    double sum_h2() const { return m_sum_h2; }
    double sum_dh2() const { return m_sum_dh2; }
    double sum_dh() const { return m_sum_dh; }       // = 0 : le bruit serie n'a pas de composante continue
    int    n_taps() const { return (int)m_h.size(); }
    const std::vector<double>& h() const { return m_h; }
    const NoiseParams& params() const { return m_p; }

    // sortie = h (*) signal : out[k] = sum_j h[j] * sig[k-j], k = 0 .. n_out-1 ; sig[i] = 0 hors de [0, n_in).
    // out est ecrase (redimensionne a n_out).
    void shape(const std::vector<double>& sig, int n_out, std::vector<double>& out) const;

    // AJOUTE a out (taille n_bins, bins de dt, sortie du preampli) le bruit parallele + serie filtre.
    // Tire (n_bins + n_taps) gaussiennes par composante non nulle : stationnaire des le bin 0.
    void add_filtered_noise(GaussRng& rng, std::vector<double>& out) const;

    // Meme chose, avec la composante SERIE multipliee par ser_scale (>= 0) : bruit d'un pad dont la capacite
    // d'entree n'est pas celle de reference (voir cap_noise.hpp). ser_scale = 1 : identique, bit a bit, a la
    // version sans argument. Le nombre de tirages ne depend pas de ser_scale.
    void add_filtered_noise(GaussRng& rng, std::vector<double>& out, double ser_scale) const;

    // echantillons du CAN : out[j] = in[phase + j * adc_ratio] tant que l'indice est < in.size()
    void decimate(const std::vector<double>& in, int phase, std::vector<double>& out) const;

    // AJOUTE le bruit blanc du CAN (sigma_can) a chaque echantillon
    void add_adc_noise(GaussRng& rng, std::vector<double>& samples) const;

    // resume lisible (a afficher au lancement)
    std::string summary() const;

private:
    NoiseParams m_p;
    std::vector<double> m_h;       // L echantillons
    std::vector<double> m_hd;      // L+1 : h[j] - h[j-1], avec h[-1] = 0 et h[L] = 0
    double m_enc_par, m_enc_ser, m_enc_can;
    double m_sig_par, m_sig_ser;
    double m_sum_h2, m_sum_dh2, m_sum_dh;
    std::vector<double> m_th;      // sigma_par * h
    std::vector<double> m_thd;     // sigma_ser * hd
};
