#pragma once

// ---------------------------------------------------------------------------
// Bruit en fonction de la capacite d'entree (modele "capacitance").
//
// L'ENC d'une voie depend de la capacite totale C vue par le preamplificateur (pad + routage + connecteurs
// + cable). Forme retenue, valable dans la gamme de l'article du Clas12 (Dream, fig. 10) :
//
//        ENC_tot(C)^2 = ENC0^2 + (b C)^2
//
//   ENC0 : ENC a C = 0 (e-), partage entre parallele / serie / CAN comme f_par : f_ser : f_can
//   b C  : terme serie lie a la capacite, b = sqrt(enc_ref^2 - enc0^2) / c_ref  (e-/pF) : un seul point
//          (enc_ref a la capacite c_ref) fixe la pente, ENC0 etant donne.
//
//   donc  ENC_par^2 = ENC0^2 f_par / fsum        (independant de C)
//         ENC_can^2 = ENC0^2 f_can / fsum        (independant de C)
//         ENC_ser(C)^2 = ENC0^2 f_ser / fsum + (b C)^2
//
// Le NoiseModel est construit a la capacite de REFERENCE c_ref (fill_noise_params). Pour un pad de capacite
// C, seul le bruit serie change : on le multiplie par ser_scale(C) = ENC_ser(C) / ENC_ser(c_ref)
// (NoiseModel::add_filtered_noise avec ser_scale). Le seuil de lecture reste global : thr_sigma * enc_ref.
//
// Ce fichier contient aussi la lecture du fichier de capacites par pad (read_pad_capacitances).
//
// Entierement dans l'en-tete : rien a ajouter au CMakeLists. C++ classique.
// ---------------------------------------------------------------------------

#include <cmath>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "noise.hpp"

struct CapNoiseParams {
    double enc0;        // e- : ENC a C = 0
    double enc_ref;     // e- : ENC a la capacite de reference
    double c_ref_pf;    // pF : capacite de reference
    double f_par, f_ser, f_can;   // partage de ENC0^2 (seul le rapport compte)

    CapNoiseParams() : enc0(1150.0), enc_ref(1400.0), c_ref_pf(93.0),
                       f_par(0.475), f_ser(0.475), f_can(0.05) {}
};

class CapNoiseModel {
public:
    explicit CapNoiseModel(const CapNoiseParams& p) : m_p(p)
    {
        if (p.enc0 <= 0.0)         throw std::runtime_error("CapNoiseModel : enc0 doit etre > 0");
        if (p.c_ref_pf <= 0.0)     throw std::runtime_error("CapNoiseModel : c_ref_pf doit etre > 0");
        if (p.enc_ref < p.enc0)    throw std::runtime_error("CapNoiseModel : enc_ref doit etre >= enc0");
        if (p.f_par < 0.0 || p.f_ser < 0.0 || p.f_can < 0.0)
                                   throw std::runtime_error("CapNoiseModel : f_par, f_ser, f_can doivent etre >= 0");
        const double fsum = p.f_par + p.f_ser + p.f_can;
        if (fsum <= 0.0)           throw std::runtime_error("CapNoiseModel : f_par + f_ser + f_can doit etre > 0");
        const double e0 = p.enc0 * p.enc0;
        m_par2  = e0 * p.f_par / fsum;
        m_can2  = e0 * p.f_can / fsum;
        m_ser02 = e0 * p.f_ser / fsum;
        m_b     = std::sqrt(p.enc_ref * p.enc_ref - e0) / p.c_ref_pf;
        const double br = m_b * p.c_ref_pf;
        m_ser_ref = std::sqrt(m_ser02 + br * br);
        if (m_ser_ref <= 0.0)
            throw std::runtime_error("CapNoiseModel : bruit serie de reference nul (f_ser = 0 et enc_ref = enc0)");
    }

    double slope() const    { return m_b; }                       // e-/pF
    double enc_par() const  { return std::sqrt(m_par2); }
    double enc_can() const  { return std::sqrt(m_can2); }
    double enc_ser_ref() const { return m_ser_ref; }
    double enc_ser(double c_pf) const { const double x = m_b * c_pf; return std::sqrt(m_ser02 + x * x); }
    double enc_total(double c_pf) const
    {
        const double s = enc_ser(c_pf);
        return std::sqrt(m_par2 + m_can2 + s * s);
    }
    double ser_scale(double c_pf) const { return enc_ser(c_pf) / m_ser_ref; }       // 1 a c_ref
    double enc_scale(double c_pf) const { return enc_total(c_pf) / m_p.enc_ref; }   // 1 a c_ref
    const CapNoiseParams& params() const { return m_p; }

    // Parametres du NoiseModel de reference : enc_total = enc_ref, et les fractions sont les carres des
    // composantes de reference (le NoiseModel les normalise : ENC_x^2 = ENC_ref^2 f_x / fsum = composante^2).
    void fill_noise_params(NoiseParams& np) const
    {
        np.enc_total = m_p.enc_ref;
        np.f_par = m_par2;
        np.f_ser = m_ser_ref * m_ser_ref;
        np.f_can = m_can2;
    }

private:
    CapNoiseParams m_p;
    double m_par2, m_can2, m_ser02, m_b, m_ser_ref;
};

// ---------------------------------------------------------------------------
// Fichier de capacites par pad : une ligne "pad_id C_pF" ; lignes vides et commentaires (#) ignores.
// pad_ids : tous les pads du detecteur. c_pf (sortie, meme ordre que pad_ids) : capacite totale en pF.
// Exceptions (message explicite) : fichier introuvable, ligne illisible, pad en double, pad inconnu,
// pad du detecteur absent du fichier, capacite negative ou non finie.
inline void read_pad_capacitances(const std::string& path, const std::vector<int>& pad_ids,
                                  std::vector<double>& c_pf)
{
    std::ifstream in(path.c_str());
    if (!in) throw std::runtime_error("pad_capacitance_file : fichier introuvable : " + path);

    std::set<int> known(pad_ids.begin(), pad_ids.end());
    std::map<int, double> cap;
    std::string line;
    int nl = 0;
    while (std::getline(in, line)) {
        ++nl;
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        std::istringstream ss(line);
        int id; double c;
        std::ostringstream where; where << path << ", ligne " << nl;
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;   // ligne vide ou commentaire seul
        if (!(ss >> id))
            throw std::runtime_error("pad_capacitance_file : ligne illisible (attendu : pad_id C_pF) : " + where.str());
        if (!(ss >> c))
            throw std::runtime_error("pad_capacitance_file : ligne illisible (attendu : pad_id C_pF) : " + where.str());
        std::string extra;
        if (ss >> extra)
            throw std::runtime_error("pad_capacitance_file : colonnes en trop (attendu : pad_id C_pF) : " + where.str());
        if (!(c >= 0.0) || !(c < 1.0e6))
            throw std::runtime_error("pad_capacitance_file : capacite negative ou non finie : " + where.str());
        if (known.find(id) == known.end()) {
            std::ostringstream m; m << "pad_capacitance_file : le pad " << id << " n'existe pas dans la geometrie ("
                                    << where.str() << ") : mauvais fichier ou mauvaise taille de pad ?";
            throw std::runtime_error(m.str());
        }
        if (cap.find(id) != cap.end()) {
            std::ostringstream m; m << "pad_capacitance_file : pad " << id << " en double (" << where.str() << ")";
            throw std::runtime_error(m.str());
        }
        cap[id] = c;
    }
    c_pf.assign(pad_ids.size(), 0.0);
    for (size_t i = 0; i < pad_ids.size(); ++i) {
        std::map<int, double>::const_iterator it = cap.find(pad_ids[i]);
        if (it == cap.end()) {
            std::ostringstream m; m << "pad_capacitance_file : le pad " << pad_ids[i]
                                    << " de la geometrie est absent du fichier " << path;
            throw std::runtime_error(m.str());
        }
        c_pf[i] = it->second;
    }
}
