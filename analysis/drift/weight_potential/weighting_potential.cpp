#include "weighting_potential.hpp"
#include <cmath>

namespace {

const double PI = 3.14159265358979323846;

// f(u,v;H) = arctan( u v / ( |H| sqrt(u^2+v^2+H^2) ) )
double f_corner(double u, double v, double H) {
    double absH = std::fabs(H);
    double r    = std::sqrt(u*u + v*v + H*H);
    double den  = absH * r;
    if (den == 0.0) {
        // H = 0 : arctan(+-infini) = +-pi/2, ou 0 si u*v = 0
        double uv = u * v;
        if (uv > 0.0) return  0.5 * PI;
        if (uv < 0.0) return -0.5 * PI;
        return 0.0;
    }
    return std::atan(u * v / den);
}

} // namespace

double solid_angle_rect(double u1, double u2, double v1, double v2, double H) {
    double s = f_corner(u2, v2, H) - f_corner(u1, v2, H)
             - f_corner(u2, v1, H) + f_corner(u1, v1, H);
    if (H < 0.0) s = -s;     // fonction impaire en H
    return s;
}

double phi_halfspace(double u, double v, double h, double a) {
    double omega = solid_angle_rect(-a - u, a - u, -a - v, a - v, h);
    return omega / (2.0 * PI);
}

double phi_point(double u, double v, double h, double a, double g, int n_img) {
    if (h >= g) return 0.0;

    // En dessous de l'anode (Q constant) : on prend la valeur à l'anode,
    // 1 si le point est au-dessus du pad, 0 sinon (bords : 0.5 / 0.25).
    // On évalue en fait en h tout petit pour rester continu.
    if (h <= 0.0) h = 1.0e-9 * g;

    double u1 = -a - u, u2 = a - u;
    double v1 = -a - v, v2 = a - v;

    // On somme de n = -n_img-1 a n_img : ainsi chaque terme n possede son
    // partenaire -n-1 (celui qui l'annule exactement en h = g, car Omega est
    // impaire). Une somme de -n_img a n_img laisserait un terme orphelin.
    double somme = 0.0;
    for (int n = -n_img - 1; n <= n_img; ++n) {
        double H = h + 2.0 * n * g;
        somme += solid_angle_rect(u1, u2, v1, v2, H);
    }
    // Correction de queue (images |n| > n_img). Loin du pad, Omega vaut
    // A*H/R^3 (pad vu comme un point, A = aire, R = distance au centre).
    // Les termes manquants s'apparient en (m, -m-1), m >= M = n_img+1 :
    //   Omega(h+2mg) - |Omega(2(m+1)g - h)|.
    // Somme remplacee par une integrale (point milieu), car l'integrale de
    // H/(rho^2+H^2)^(3/2) est -1/sqrt(rho^2+H^2).
    {
        double aire = 4.0 * a * a;
        double rho2 = u * u + v * v;
        double Hbase = (2.0 * (n_img + 1) - 1.0) * g;   // (2M-1) g
        double H1 = h + Hbase;
        double H2 = 2.0 * g - h + Hbase;
        double queue = aire / (2.0 * g)
                     * (1.0 / std::sqrt(rho2 + H1 * H1) - 1.0 / std::sqrt(rho2 + H2 * H2));
        somme += queue;
    }

    return somme / (2.0 * PI);
}
