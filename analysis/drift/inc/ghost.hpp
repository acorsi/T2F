#pragma once

// ---------------------------------------------------------------------------
// Pads fantomes : pads sans signal vrai dont le bruit seul franchit le seuil de lecture.
//
// Un pad sans signal a, a la sortie du CAN, une suite d'echantillons gaussiens stationnaires, de
// variance enc_total^2 et de covariance
//     c(k) = sigma_par^2 sum_j h_j h_(j+kR)  +  sigma_ser^2 sum_j hd_j hd_(j+kR)      (k >= 1)
//     c(0) = enc_total^2                       (R = adc_ratio ; hd = h[j] - h[j-1])
// Il est lu (et devient un fantome) si n_consec echantillons consecutifs de la fenetre complete
// depassent le seuil : meme critere que pour les pads du signal (voir readout.cpp).
//
// Probabilite p : un pad sans signal est lu avec la probabilite p (fenetre complete n_bins). Chaque
// pad hors de la liste vraie est un fantome avec la probabilite p, independamment : le nombre de
// fantomes suit une loi binomiale, exactement ce que donnerait un tirage du bruit sur tous les pads.
//
// Forme d'un fantome : tiree EXACTEMENT sous la condition "le pad est lu", sans rejeter 1/p tirages
// (p ~ 1e-3). Methode (conditionnement d'un vecteur gaussien, forme de Slepian) :
//   1. on tire un bruit non conditionne x (meme code que pour les pads du signal) ;
//   2. on choisit un echantillon j* au hasard, on tire sa valeur a > seuil (gaussienne tronquee) et on
//      impose x_j* = a : x_j <- x_j + (a - x_j*) c(|j - j*|) / c(0)      (exact pour un vecteur gaussien) ;
//   3. on garde le resultat avec la probabilite 1/K si le pad est lu, 0 sinon (K = nombre
//      d'echantillons au-dessus du seuil) : le choix de j* favorise les pads ayant plusieurs
//      echantillons au-dessus du seuil, K compense exactement ce biais.
// La meme proposition donne p sans tirage rare :  p = Ns q E[ 1(lu) / K ],  q = P(x_j > seuil),
// Ns = nombre d'echantillons de la fenetre (estimateur d'importance, precis meme pour 6 sigma).
//
// Les fantomes n'existent qu'avec le bruit. C++ classique.
// ---------------------------------------------------------------------------

#include <stdint.h>
#include <map>
#include <vector>
#include "noise.hpp"

struct GhostParams {
    bool   enable;          // false : pas de fantomes
    double prob;            // < 0 : p calibre par le programme ; >= 0 : p imposee
    double window_ns;       // longueur de la fenetre ecrite d'un fantome (ns)
    int    calib_trials;    // nombre de tirages pour calibrer p

    GhostParams() : enable(true), prob(-1.0), window_ns(940.0), calib_trials(20000) {}
};

class GhostModel {
public:
    // thr : seuil de lecture (e-) ; n_consec : echantillons consecutifs ; n_bins : fenetre (bins de dt)
    GhostModel(const NoiseModel& noise, const GhostParams& gp, double thr, int n_consec, int n_bins);

    // Calibre p (si gp.prob < 0) avec le generateur donne, ou prend la valeur imposee.
    // A appeler une fois avant le traitement des evenements (pas thread-safe).
    void calibrate(GaussRng& rng);

    double prob() const { return m_p; }              // p retenue
    double prob_err() const { return m_p_err; }      // erreur statistique sur p (0 si imposee)
    int    window_samples() const { return m_win; }  // longueur ecrite d'un fantome (echantillons CAN)
    int    n_samples() const { return m_ns; }        // echantillons CAN de la fenetre complete (phase 0)
    double q_tail() const { return m_q; }            // P(un echantillon > seuil)
    double cov(int k) const { return (k >= 0 && k < (int)m_c.size()) ? m_c[k] : 0.0; }
    int    n_cov() const { return (int)m_c.size(); } // nombre de lags non nuls
    double threshold() const { return m_thr; }

    // Tire la suite d'echantillons (fenetre complete, phase donnee) d'un pad lu, sous la condition
    // "lu". smp : echantillons en e- equivalents. j_first : indice du premier echantillon d'une suite
    // de n_consec echantillons consecutifs au-dessus du seuil. Retourne le nombre de propositions
    // tirees (>= 1). thread-safe : n'ecrit que dans les arguments.
    // scale : rapport ENC du pad / ENC de reference (1 pour le pad de reference). Le bruit du pad est pris egal
    // a scale fois le bruit de reference : meme forme de correlation (approximation), amplitude exacte.
    int draw(GaussRng& rng, int phase, std::vector<double>& smp, int& j_first, double scale = 1.0) const;

    // Probabilite qu'un pad d'ENC = scale * ENC de reference soit lu. scale = 1, ou p imposee : p.
    // Si calibrate_scales a ete appelee avec cette valeur de scale (arrondie a 1e-4) : p calibree pour ce scale
    // (meme estimateur que calibrate, seuil / scale). Sinon : formule approchee p * q(scale) / q(1), qui garde
    // le facteur de regroupement des franchissements du pad de reference (surestime p quand scale > 1.1 environ).
    // Lecture seule : utilisable depuis plusieurs threads.
    double prob_for_scale(double scale) const;

    // Calibre p pour chaque valeur de scale de la liste (doublons et arrondis fusionnes). A appeler apres
    // calibrate, avant le traitement (pas thread-safe). Sans effet si p est imposee.
    void calibrate_scales(GaussRng& rng, const std::vector<double>& scales);

    // Une proposition (etapes 1 et 2) : retourne K ; pass vrai si le pad est lu ; j_first comme ci-dessus.
    // scale : seuil effectif = seuil / scale ; smp reste a l'echelle du bruit de reference.
    int propose(GaussRng& rng, int phase, std::vector<double>& smp, bool& pass, int& j_first,
                double scale = 1.0) const;

    // Gaussienne tronquee : z > t (t quelconque), loi N(0,1) conditionnee a z > t.
    static double normal_above(GaussRng& rng, double t);

    // Premier indice j tel que smp[j .. j+n_consec-1] > thr ; -1 si aucun.
    static int first_run(const std::vector<double>& smp, double thr, int n_consec);

private:
    const NoiseModel* m_noise;
    GhostParams m_gp;
    double m_thr;
    int    m_n_consec;
    int    m_n_bins;
    int    m_ns;            // echantillons CAN de la fenetre, phase 0
    int    m_win;           // fenetre ecrite
    double m_c0;            // = enc_total^2
    double m_q;             // P(x > seuil), x ~ N(0, c0)
    std::vector<double> m_c;    // c(0), c(1), ... jusqu'au dernier lag non nul
    double m_p, m_p_err;
    std::map<long, double> m_ps;   // p calibree par scale (cle = round(scale * 1e4))
};
