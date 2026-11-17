#pragma once

// ---------------------------------------------------------------------------
// Modele de l'ENC en fonction du temps de pic tp et de la capacite d'entree C
// (utilise avec la reponse MINOS : le bruit decoule alors de preamp_minos_peak_ns).
//
//   ENC^2(tp, C) = (s0 + s1 C^2) / tp  +  p tp  +  f0              (e-^2 ; tp en ns, C en pF)
//
//   (s0 + s1 C^2) / tp : bruit SERIE (blanc en tension) ; s1 C^2 est le terme lie a la capacite
//   p tp               : bruit PARALLELE (blanc en courant)
//   f0                 : terme constant en tp, attribue au bruit du CAN (bruit blanc de sortie) ;
//                        il contient en realite aussi du bruit en 1/f, non modelise a part
//
// Origine : ajustement des 30 points lus sur la figure 10 de l'article du Dream (CLAS12), gamme
// 200 fC, six temps de pic (76, 123, 180, 388, 578, 919 ns) x cinq capacites (0, 50, 100, 160, 220 pF).
// Ecart avec les points : rms 60 e-, maximum 170 e- (a 76 ns, 100 pF), du niveau du bruit de lecture
// de la figure (+- 50 e-). Les constantes ne sont pas dans le .cfg : on ne les change pas par erreur.
//
// Validite : tp entre 76 et 919 ns (plage mesuree) ; en dehors, exception.
//
// Correspondance avec CapNoiseModel (cap_noise.hpp) :
//   ENC_par^2 = p tp ; ENC_can^2 = f0 ; ENC_ser(C)^2 = s0/tp + (b C)^2 avec b = sqrt(s1 / tp).
//
// Entierement dans l'en-tete. C++ classique.
// ---------------------------------------------------------------------------

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

struct EncComponents {
    double par2;     // e-^2 : bruit parallele (independant de C)
    double ser02;    // e-^2 : bruit serie a C = 0
    double can2;     // e-^2 : bruit du CAN (independant de C)
    double b;        // e-/pF : pente du terme serie lie a la capacite
};

class EncTpModel {
public:
    static double s0() { return 1.31514628e8; }    // e-^2 ns
    static double s1() { return 2.08341573e4; }    // e-^2 ns / pF^2
    static double p()  { return 2.10937562e3; }    // e-^2 / ns
    static double f0() { return 2.59645231e5; }    // e-^2

    static double tp_min_ns() { return 76.0; }
    static double tp_max_ns() { return 919.0; }

    static void check_tp(double tp_ns) {
        if (!(tp_ns >= tp_min_ns() && tp_ns <= tp_max_ns())) {
            std::ostringstream m;
            m << "EncTpModel : temps de pic " << tp_ns << " ns hors de la plage autorisee ["
              << tp_min_ns() << " ; " << tp_max_ns() << "] ns";
            throw std::runtime_error(m.str());
        }
    }

    static EncComponents components(double tp_ns) {
        check_tp(tp_ns);
        EncComponents e;
        e.par2  = p() * tp_ns;
        e.ser02 = s0() / tp_ns;
        e.can2  = f0();
        e.b     = std::sqrt(s1() / tp_ns);
        return e;
    }

    // ENC totale (e-) pour un temps de pic et une capacite
    static double enc(double tp_ns, double c_pf) {
        const EncComponents e = components(tp_ns);
        const double x = e.b * c_pf;
        return std::sqrt(e.par2 + e.ser02 + e.can2 + x * x);
    }

    static std::string describe() {
        std::ostringstream s;
        s << "ENC^2 = (s0 + s1 C^2)/tp + p tp + f0 avec s0 = " << s0() << " e-^2 ns, s1 = " << s1()
          << " e-^2 ns/pF^2, p = " << p() << " e-^2/ns, f0 = " << f0()
          << " e-^2 (attribue au CAN) ; ajuste sur la figure 10 Dream/CLAS12 (30 points), valable de "
          << tp_min_ns() << " a " << tp_max_ns() << " ns";
        return s.str();
    }
};
