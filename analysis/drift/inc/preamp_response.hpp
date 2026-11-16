#pragma once

// ---------------------------------------------------------------------------
// Reponse du preamplificateur : interface + reponse de TEST.
//
// Le module de bruit (noise.hpp) ne connait que l'interface PreampResponse.
// La reponse reelle (reponse MINOS) est une autre classe, ecrite par ailleurs :
// elle derive de PreampResponse et remplace GaussTestResponse dans main.cpp.
//
// Convention (a respecter par toute reponse) :
//   - h[k] = reponse a t = k * dt, pour k = 0, 1, ..., n-1 ;
//   - causale : h(t < 0) = 0 ; h[0] peut etre non nul mais doit rester petit ;
//   - normalisee a 1 au maximum : max(h) = 1 (c'est la convention de l'ENC :
//     une charge de Q electrons injectee d'un coup donne un pic de sortie Q) ;
//   - le tableau couvre toute la reponse : les derniers echantillons sont negligeables
//     (|h| < 1e-3 en fin de tableau, verifie par NoiseModel).
//
// Un electron induit dans le bin k (charge qt[k]) contribue a la sortie au bin k + j
// avec le poids h[j] : la charge est donc supposee arriver au DEBUT du bin.
//
// C++ classique.
// ---------------------------------------------------------------------------

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

class PreampResponse {
public:
    virtual ~PreampResponse() {}

    // nom court ("test_gauss", ...) et description des parametres (pour les metadonnees)
    virtual std::string name() const = 0;
    virtual std::string describe() const = 0;

    // remplit h avec la reponse echantillonnee au pas dt (ns) ; voir les conventions ci-dessus
    virtual void sample(double dt, std::vector<double>& h) const = 0;
};

// ---------------------------------------------------------------------------
// Reponse de TEST : gaussienne causale. Sans lien avec l'electronique reelle.
//   h(t) = exp( -(t - t0)^2 / (2 sigma^2) ),  sigma = fwhm / 2.3548,  t0 = 4 sigma
// (t0 = 4 sigma rend h(0) ~ 3e-4 : la coupure a t = 0 est negligeable).
// ---------------------------------------------------------------------------
class GaussTestResponse : public PreampResponse {
public:
    explicit GaussTestResponse(double fwhm_ns) : m_fwhm(fwhm_ns) {
        if (fwhm_ns <= 0.0) throw std::runtime_error("GaussTestResponse : fwhm doit etre > 0");
    }

    std::string name() const { return "test_gauss"; }

    std::string describe() const {
        std::ostringstream s;
        s << "gaussienne causale de test, fwhm = " << m_fwhm << " ns, pic a t0 = 4 sigma = "
          << 4.0 * sigma() << " ns";
        return s.str();
    }

    void sample(double dt, std::vector<double>& h) const {
        if (dt <= 0.0) throw std::runtime_error("GaussTestResponse::sample : dt doit etre > 0");
        const double s  = sigma();
        const double t0 = 4.0 * s;
        const double tmax = t0 + 4.5 * s;               // exp(-4.5^2/2) = 4e-5 : negligeable
        const int n = (int)std::ceil(tmax / dt) + 1;
        h.assign(n, 0.0);
        for (int k = 0; k < n; ++k) {
            const double x = (k * dt - t0) / s;
            h[k] = std::exp(-0.5 * x * x);
        }
        // normalisation exacte a 1 au maximum de l'echantillonnage
        double m = 0.0;
        for (int k = 0; k < n; ++k) if (h[k] > m) m = h[k];
        for (int k = 0; k < n; ++k) h[k] /= m;
    }

private:
    double m_fwhm;
    double sigma() const { return m_fwhm / 2.3548200450309493; }
};

// ---------------------------------------------------------------------------
// Reponse MINOS (equation 6 du document de reference) :
//   h(t) ~ (t/tau)^3 * exp(-3 t/tau) * sin(t/tau),   t >= 0
// normalisee a 1 au maximum.
//
// Le parametre est le TEMPS DU MAXIMUM de h (peaking time, "preamp_minos_peak_ns") ; tau en
// decoule : le maximum est a u = t/tau = 1.1664..., donc tau = peak / 1.1664...
// Proprietes utiles (peak = 1 au maximum) : integrale = 1.306 tau, largeur a mi-hauteur =
// 1.264 tau, sous-oscillation negative maximale = -0.957 %.
//
// Troncature : la queue est coupee des que l'enveloppe u^3 exp(-3u) / (valeur au maximum)
// passe sous TAIL_CUT = 1e-4 (donc |h| < 1e-4 sur le dernier echantillon ; NoiseModel
// exige moins de 1e-3).
//
// Normalisation : comme pour GaussTestResponse, a 1 exactement au maximum de
// l'ECHANTILLONNAGE (si le pic tombe entre deux echantillons, l'ecart avec le vrai maximum
// est de l'ordre de (dt/tau)^2).
// ---------------------------------------------------------------------------
class MinosResponse : public PreampResponse {
public:
    // plage de peaking time autorisee (reglages mesures de l'electronique : 76 a 919 ns)
    static double peak_min_ns() { return 76.0; }
    static double peak_max_ns() { return 919.0; }
    static double tail_cut()    { return 1.0e-4; }

    explicit MinosResponse(double peak_ns) : m_peak(peak_ns) {
        if (!(peak_ns > 0.0)) throw std::runtime_error("MinosResponse : le temps de pic doit etre > 0");
        m_tau = peak_ns / u_peak();
    }

    std::string name() const { return "minos"; }

    std::string describe() const {
        std::ostringstream s;
        s << "reponse MINOS (t/tau)^3 exp(-3t/tau) sin(t/tau), temps du maximum = " << m_peak
          << " ns, tau = " << m_tau << " ns, queue coupee a " << tail_cut();
        return s.str();
    }

    double tau() const     { return m_tau; }
    double peak_ns() const { return m_peak; }

    // position du maximum en unites de tau : solution de 3/u - 3 + cos(u)/sin(u) = 0 (1.1664...)
    static double u_peak() {
        double a = 0.5, b = 1.5;                 // la fonction change de signe sur [0.5 ; 1.5]
        for (int i = 0; i < 200; ++i) {
            const double m = 0.5 * (a + b);
            const double f = 3.0 / m - 3.0 + std::cos(m) / std::sin(m);
            if (f > 0.0) a = m; else b = m;
        }
        return 0.5 * (a + b);
    }

    void sample(double dt, std::vector<double>& h) const {
        if (dt <= 0.0) throw std::runtime_error("MinosResponse::sample : dt doit etre > 0");
        const double up   = u_peak();
        const double hpk  = form(up);
        // u de coupure : premier u > up ou l'enveloppe passe sous tail_cut (decroissante pour u > 1)
        double a = up, b = 30.0;
        for (int i = 0; i < 200; ++i) {
            const double m = 0.5 * (a + b);
            if (envelope(m) / hpk > tail_cut()) a = m; else b = m;
        }
        const double tmax = b * m_tau;
        const int n = (int)std::ceil(tmax / dt) + 1;
        h.assign(n, 0.0);
        for (int k = 0; k < n; ++k) h[k] = form(k * dt / m_tau);
        // normalisation exacte a 1 au maximum de l'echantillonnage
        double m = 0.0;
        for (int k = 0; k < n; ++k) if (h[k] > m) m = h[k];
        for (int k = 0; k < n; ++k) h[k] /= m;
    }

private:
    double m_peak;     // ns : temps du maximum
    double m_tau;      // ns

    static double envelope(double u) { return u * u * u * std::exp(-3.0 * u); }
    static double form(double u)     { return envelope(u) * std::sin(u); }
};
