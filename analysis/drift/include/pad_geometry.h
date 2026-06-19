#pragma once
#include <vector>
#include <memory>
#include <string>

struct PadID {
    int row, col;
    bool operator==(const PadID& o) const { return row == o.row && col == o.col; }
};

// Interface commune à toutes les géométries de pad
class PadGeometry {
public:
    virtual ~PadGeometry() = default;

    // Retourne les pads touchés par un dépôt de charge gaussien centré en (x,y) [cm]
    // avec écart-type sigma [cm] (diffusion transverse isotrope), et la fraction de
    // charge reçue par chaque pad. Pas de renormalisation : la somme des poids
    // peut être < 1 (charge perdue sous le seuil de troncature ou hors disque).
    virtual std::vector<std::pair<PadID,double>>
        chargeSplit(double x, double y, double sigma) const = 0;
};

// Géométrie cartésienne en quinconce : pads carrés, rangées décalées d'un demi-pad
class QuincunxGeometry : public PadGeometry {
public:
    QuincunxGeometry(double pad_size_cm, double radius_cm);

    std::vector<std::pair<PadID,double>>
        chargeSplit(double x, double y, double sigma) const override;

private:
    double m_padSize;  // cm
    double m_radius;   // cm

    // Centre du pad (row,col) dans le repère (x,y)
    void padCenter(int row, int col, double& cx, double& cy) const;
    // Pad contenant le point (x,y)
    PadID  padAt(double x, double y) const;
};

// Factory : choix de la géométrie par mot-clé
std::unique_ptr<PadGeometry> makePadGeometry(const std::string& kind,
                                              double param1, double param2);
