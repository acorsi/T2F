#pragma once

// ---------------------------------------------------------------------------
// Potentiel de pondération d'un pad carré (Ramo-Shockley)
//
// Toutes les longueurs sont dans la MEME unité (mm ou cm, peu importe),
// il suffit d'être cohérent entre a, g, u, v, h.
//
// Modèle : tranche 0 < h < g, anode en h = 0, grille (phi = 0) en h = g,
// pad = tuile carrée de demi-côté a, centrée à l'origine.
// ---------------------------------------------------------------------------

// Angle solide (en sr) du rectangle [u1,u2] x [v1,v2], vu depuis un point
// situé à la hauteur H au-dessus du plan. Les bords u1,u2,v1,v2 sont relatifs
// à la projection du point sur le plan. Fonction impaire en H.
double solid_angle_rect(double u1, double u2, double v1, double v2, double H);

// phi sans grille (demi-espace) : Omega / (2 pi).
double phi_halfspace(double u, double v, double h, double a);

// phi avec grille en h = g (méthode des images, n = -n_img ... +n_img).
// (u, v) : décalage entre le point et le centre du pad.
// Valable pour 0 < h < g. Pour h <= 0 on renvoie la valeur à l'anode,
// pour h >= g on renvoie 0.
double phi_point(double u, double v, double h, double a, double g, int n_img);
