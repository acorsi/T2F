#include "processor.hpp"
#include <cmath>

Processor::Processor(const DetectorConfig& cfg, uint64_t seed)
    : m_cfg(cfg), m_rng(seed)
{}

void Processor::draw_vertex(double& va, double& vb) {
    // type1 : (vx, vy) dans le plan XY
    // type2 : (vx, vz) dans le plan XZ (vertex en y=0)
    std::uniform_real_distribution<double> dist(-m_cfg.vertex_r_max,
                                                 m_cfg.vertex_r_max);
    double r = m_cfg.vertex_r_max;
    while (true) {
        double a = dist(m_rng);
        double b = dist(m_rng);
        if (a*a + b*b <= r*r) { va = a; vb = b; return; }
    }
}

void Processor::build_trace(const FragmentInput& frag_in,
                             const Direction& dir,
                             double va, double vb,
                             FragmentOutput& out)
{
    const EmissionAxis axis = m_cfg.emission_axis;
    const double ux   = dir.ux(axis);
    const double uy   = dir.uy(axis);
    const double uz   = dir.uz(axis);
    const double step = m_cfg.step_size;
    const double z_grid = m_cfg.z_grid();

    const size_t N = frag_in.raw_trace.size();
    out.x.clear();          out.x.reserve(N);
    out.y.clear();          out.y.reserve(N);
    out.z.clear();          out.z.reserve(N);
    out.dE.clear();         out.dE.reserve(N);
    out.n_electrons.clear();out.n_electrons.reserve(N);
    out.sigma_T.clear();    out.sigma_T.reserve(N);
    out.sigma_L.clear();    out.sigma_L.reserve(N);

    for (size_t i = 0; i < N; ++i) {
        double xi, yi, zi;

	xi = va   + step * i * ux;
	yi =        step * i * uy;  // vertex en y=0
	zi = m_cfg.dkg / 2.0 + vb + step * i * uz;  // cible centrée à dkg/2
	
        // Condition d'arrêt
        if (zi >= z_grid ) break;

        out.x.push_back(xi);
        out.y.push_back(yi);
        out.z.push_back(zi);
        out.dE.push_back(frag_in.raw_trace[i]);
        out.n_electrons.push_back(frag_in.raw_trace[i] / m_cfg.W);
        out.sigma_T.push_back(m_cfg.sigma_T(zi));
        out.sigma_L.push_back(m_cfg.sigma_L(zi));
    }
}

void Processor::process(const EventInput&  in,
                               EventOutput& out,
                         std::optional<Direction> dir_H,
                         std::optional<Direction> dir_L)
{
    double va, vb;
    draw_vertex(va, vb);

    Direction dH = dir_H.value_or(draw_direction(m_rng));
    Direction dL = dir_L.value_or(dH.opposite());

    out.H.vx = va; out.H.vy = 0.0; out.H.vz = m_cfg.dkg / 2.0 + vb;
    out.L.vx = va; out.L.vy = 0.0; out.L.vz = m_cfg.dkg / 2.0 + vb;
        
    out.H.dir = dH;
    out.L.dir = dL;

    build_trace(in.H, dH, va, vb, out.H);
    build_trace(in.L, dL, va, vb, out.L);
}
