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
