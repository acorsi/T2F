#include "process.h"
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <map>

constexpr double BIN_SIZE_CM = 0.004;  // 40 microns en cm
constexpr double EV_PER_PAIR = 26.0;   // eV nécessaires pour créer une paire électron-ion
constexpr double ELECTRONS_PER_MEV = 1.0e6 / EV_PER_PAIR;  // ~38462 e-/MeV

Processor::Processor()
{
    int nThreads = omp_get_max_threads();
    std::random_device rd;
    m_rngs.resize(nThreads);
    for (int i = 0; i < nThreads; ++i)
        m_rngs[i].seed(rd() + i);

    v_e        = 5;     // vitesse dérive électron en cm.us-1 (microsecondes)
    diff_trans = 0.03;  // Coeff diffusion transverse cm/sqrt(L)
    diff_long  = 0.015; // Coeff diffusion longitudinal cm/sqrt(L)
    dKG        = 10;    // distance cathode-Grille en cm
    dGA        = 1;     // distance Grille-Anode en cm

    noiseSigma     = 2000.0 / (2.0 * std::sqrt(2.0 * std::log(2.0)));  // FWHM=2000 -> sigma

    samplingStep_cm = 0.05;  // 500 microns par défaut (marge x8 sous la taille du pad)

    m_padGeom = makePadGeometry("quincunx", /*pad_size_cm=*/0.4, /*radius_cm=*/9.0);
}

constexpr double TIME_STEP_NS = 10.0;
constexpr double CURRENT_THRESHOLD = 500.0;  // électrons/ns, seuil de déclenchement
constexpr double PRETRIGGER_NS = 50.0;       // marge de pré-déclenchement, ns

void Processor::processFragment(Fragment& frag, double pInit[3], double v[3], std::mt19937& rng)
{
    int len = static_cast<int>(frag.raw_trace.size());

    // Accumulateur (row,col) -> charge totale, fusionne les contributions de plusieurs bins
    std::map<std::pair<int,int>, double> padCharge;

    // Accumulateur (row,col) -> liste de contributions temporelles (t_i, sigma_t_i, charge_i)
    struct TimeContrib { double t, sigma_t, charge; };
    std::map<std::pair<int,int>, std::vector<TimeContrib>> padTimeContribs;

    // Regroupement de bins consécutifs pour réduire le nombre d'appels à chargeSplit.
    // groupSize bins de 40 microns forment un paquet de samplingStep_cm.
    int groupSize = std::max(1, (int)std::round(samplingStep_cm / BIN_SIZE_CM));

    for (int i = 0; i < len; i += groupSize)
    {
        int iEnd = std::min(i + groupSize, len);
        int iMid = (i + iEnd - 1) / 2;  // bin central du paquet

        double l_mid = iMid * BIN_SIZE_CM;
        double x = pInit[0] + l_mid * v[0];
        double y = pInit[1] + l_mid * v[1];
        double z = pInit[2] + l_mid * v[2];

        double L_i = z;  // distance de dérive jusqu'à l'anode (z=0)
        if (L_i <= 0.0) break;  // le fragment a atteint/dépassé l'anode

        double sigma_T = diff_trans * std::sqrt(L_i);

        // Temps de dérive et étalement longitudinal -> temporel (v_e en cm/us -> conversion en ns)
        double t_i       = (L_i / v_e) * 1000.0;                          // ns
        double sigma_t_i = ((diff_long * std::sqrt(L_i)) / v_e) * 1000.0; // ns

        // Somme du dépôt d'énergie sur tout le paquet (MeV), converti en électrons
        double dE_MeV = 0.0;
        for (int j = i; j < iEnd; ++j) dE_MeV += frag.raw_trace[j];
        double dE = dE_MeV * ELECTRONS_PER_MEV;

        auto hits = m_padGeom->chargeSplit(x, y, sigma_T);
        for (const auto& [id, w] : hits) {
            double dE_pad = dE * w;
            auto key = std::make_pair(id.row, id.col);
            padCharge[key] += dE_pad;
            padTimeContribs[key].push_back({t_i, sigma_t_i, dE_pad});
        }
    }

    // Conversion vers le format de sortie sparse, sélection par seuil de courant
    std::normal_distribution<double> noiseDist(0.0, noiseSigma);

    frag.proc_trace.clear();
    frag.proc_trace.reserve(padTimeContribs.size());
    for (const auto& entry : padTimeContribs) {
        const auto& rc = entry.first;
        const auto& contribs = entry.second;

        double tMin = contribs[0].t - 5.0 * contribs[0].sigma_t;
        double tMax = contribs[0].t + 5.0 * contribs[0].sigma_t;
        for (const auto& c : contribs) {
            tMin = std::min(tMin, c.t - 5.0 * c.sigma_t);
            tMax = std::max(tMax, c.t + 5.0 * c.sigma_t);
        }
        tMin = std::max(0.0, tMin);

        const double inv_sqrt2pi = 1.0 / std::sqrt(2.0 * M_PI);

        // Densité de courant instantanée I(t) = dQ/dt, somme des contributions gaussiennes
        auto evalQ = [&](double t) {
            double q = 0.0;
            for (const auto& c : contribs) {
                double sigma = std::max(c.sigma_t, 1e-9);  // évite division par zéro
                double z = (t - c.t) / sigma;
                q += c.charge * std::exp(-0.5 * z * z) * inv_sqrt2pi / sigma;
            }
            return q;
        };

        // Recherche du premier franchissement du seuil de courant
        int nSamplesCoarse = std::max(1, (int)std::ceil((tMax - tMin) / TIME_STEP_NS) + 1);

        double tCross = -1.0;
        double qPrev = evalQ(tMin);
        double tPrev = tMin;
        for (int k = 1; k <= nSamplesCoarse; ++k) {
            double t = tMin + k * TIME_STEP_NS;
            double q = evalQ(t);

            if (q >= CURRENT_THRESHOLD) {
                if (q > qPrev)
                    tCross = tPrev + (CURRENT_THRESHOLD - qPrev) * (t - tPrev) / (q - qPrev);
                else
                    tCross = t;  // cas dégénéré (saut brutal), on prend ce point
                break;
            }
            qPrev = q;
            tPrev = t;
        }

        // Pad non détecté : le courant ne dépasse jamais le seuil -> rejeté
        if (tCross < 0.0) continue;

        // Application de la marge de pré-déclenchement
        double tStart = tCross - PRETRIGGER_NS;

        // Reconstruction de la grille finale à partir de tStart
        int nSamples = std::max(1, (int)std::ceil((tMax - tStart) / TIME_STEP_NS) + 1);
        std::vector<double> qt(nSamples, 0.0);
        for (int k = 0; k < nSamples; ++k) {
            double t = tStart + k * TIME_STEP_NS;
            qt[k] = evalQ(t);
        }

        PadHit hit;
        hit.row = rc.first;
        hit.col = rc.second;
        hit.charge = padCharge[rc] + noiseDist(rng);  // charge totale, bruitée une fois
        hit.t_start = tStart;
        hit.qt = qt;
        frag.proc_trace.push_back(hit);
    }
}

void Processor::processEvent(Event& event, std::mt19937& rng)
{
    double vl[3], vh[3], pos[3];

    std::uniform_real_distribution<> dist(0, 1.0);
    double phi     = 2. * acos(-1.) * dist(rng);
    double c_theta = dist(rng);
    double s_theta = sqrt(1 - c_theta * c_theta);
    double theta   = acos(c_theta);

    vl[0] = s_theta * cos(phi);
    vl[1] = s_theta * sin(phi);
    vl[2] = -c_theta;  // z décroissant : vers l'anode de ce côté

    vh[0] = -vl[0];
    vh[1] = -vl[1];
    vh[2] = -c_theta;  // z décroissant : vers l'anode de l'autre côté

    pos[2] = dKG + dGA;
    double rho = 0.5 * std::sqrt(dist(rng));
    dist = std::uniform_real_distribution<>(0, 2 * 3.14159);
    double posAngle = dist(rng);
    pos[0] = rho * cos(posAngle);
    pos[1] = rho * sin(posAngle);

    // Stockage des informations angulaires et de position pour la reconstruction
    event.L.theta = theta;       event.L.phi = phi;
    event.H.theta = theta;       event.H.phi = phi + M_PI;  // direction opposée en x,y
    for (int i = 0; i < 3; ++i) { event.L.pInit[i] = pos[i]; event.H.pInit[i] = pos[i]; }

    processFragment(event.L, pos, vl, rng);
    processFragment(event.H, pos, vh, rng);
}

void Processor::processChunk(std::vector<Event>& events)
{
    int n = static_cast<int>(events.size());
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < n; ++i)
        processEvent(events[i], m_rngs[omp_get_thread_num()]);
}
