#pragma once

#include "detector.hpp"
#include "direction.hpp"
#include "event.hpp"
#include <optional>
#include <random>

class Processor {
public:
    Processor(const DetectorConfig& cfg, uint64_t seed);

    // Traite un event complet.
    // Si dir_H / dir_L sont fournis, on les utilise directement (lecture H5).
    // Sinon, on tire aléatoirement.
    void process(const EventInput&  in,
                       EventOutput& out,
                 std::optional<Direction> dir_H = std::nullopt,
                 std::optional<Direction> dir_L = std::nullopt);

private:
    DetectorConfig  m_cfg;
    std::mt19937_64 m_rng;  // propre à cette instance → safe OpenMP

    // Tire un vertex uniforme dans le disque de rayon vertex_r_max
    // type1 : (va=vx, vb=vy), type2 : (va=vx, vb=vz)
    void draw_vertex(double& va, double& vb);

    // Construit la trace 3D d'un fragment depuis le vertex dans une direction.
    // va, vb : coordonnées du vertex selon le type (voir draw_vertex)
    void build_trace(const FragmentInput& frag_in,
                     const Direction& dir,
                     double va, double vb,
                     FragmentOutput& out);
};
