#include "readout.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace {

struct Entry {
    const PadCharge* pc;
    int flag;
};

// ceil(a / b) et floor(a / b) pour b > 0, a de signe quelconque
int ceil_div(int a, int b)  { return (a >= 0) ? (a + b - 1) / b : -((-a) / b); }
int floor_div(int a, int b) { return (a >= 0) ? a / b : -((-a + b - 1) / b); }

// graine du generateur des fantomes, derivee de celle de l'evenement (splitmix64)
uint64_t ghost_seed(uint64_t seed)
{
    uint64_t z = seed + 0xD1B54A32D192ED03ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

bool by_pad_id(const ReadoutPad& a, const ReadoutPad& b) { return a.pad_id < b.pad_id; }

}  // namespace

Readout::Readout(const ReadoutParams& rp, const NoiseModel& noise, int n_bins)
    : m_rp(rp), m_noise(&noise), m_n_bins(n_bins), m_ghost(0)
{
    if (n_bins < 1)           throw std::runtime_error("Readout : n_bins doit etre >= 1");
    if (rp.n_consec < 1)      throw std::runtime_error("Readout : read_n_consec doit etre >= 1");
    if (rp.thr_sigma < 0.0)   throw std::runtime_error("Readout : read_thr_sigma doit etre >= 0");
    if (rp.margin_ns < 0.0)   throw std::runtime_error("Readout : window_margin_ns doit etre >= 0");
    m_thr    = rp.thr_sigma * noise.params().enc_total;
    m_tail   = (int)noise.h().size() - 1;       // la convolution signal (*) h s'etend sur nq + n_taps - 1 bins
    m_margin = (int)std::ceil(rp.margin_ns / noise.params().dt - 1.0e-9);
    // grille de lecture : la grille physique, rallongee de la queue complete de h et de la marge, pour que
    // la fenetre ecrite ne soit jamais coupee, meme pour un signal qui arrive au dernier bin de la grille
    m_grid   = n_bins + m_tail + m_margin;
}

void Readout::set_ghosts(const GhostModel* g, const std::vector<PadSite>& sites)
{
    m_ghost = g;
    m_sites = sites;
}

void Readout::set_pad_noise(const std::vector<int>& pad_ids, const std::vector<double>& ser_scale,
                            const std::vector<double>& enc_scale)
{
    if (pad_ids.size() != ser_scale.size() || pad_ids.size() != enc_scale.size())
        throw std::runtime_error("Readout::set_pad_noise : tableaux de tailles differentes");
    m_pn.clear();
    for (size_t i = 0; i < pad_ids.size(); ++i) {
        if (!(ser_scale[i] >= 0.0) || !(enc_scale[i] > 0.0))
            throw std::runtime_error("Readout::set_pad_noise : facteurs de bruit invalides");
        PadNoise pn; pn.ser_scale = ser_scale[i]; pn.enc_scale = enc_scale[i];
        m_pn[pad_ids[i]] = pn;
    }
}

void Readout::process(const std::vector<PadCharge>& H, const std::vector<PadCharge>& L,
                      uint64_t seed, ReadoutEvent& out) const
{
    const NoiseModel& M = *m_noise;
    const int R = M.params().adc_ratio;

    GaussRng rng(seed);
    out.phase   = rng.uniform_int(R);       // toujours le premier tirage, avec ou sans bruit
    out.n_ghost = 0;
    out.pads.clear();

    // liste unique des pads, par pad_id croissant ; un pad present dans H et dans L porte
    // les memes donnees (meme somme H+L) : on garde celles de H
    std::map<int, Entry> uniq;
    for (size_t i = 0; i < H.size(); ++i) {
        Entry e; e.pc = &H[i]; e.flag = 1;
        uniq[H[i].pad_id] = e;
    }
    for (size_t i = 0; i < L.size(); ++i) {
        std::map<int, Entry>::iterator it = uniq.find(L[i].pad_id);
        if (it == uniq.end()) {
            Entry e; e.pc = &L[i]; e.flag = 2;
            uniq[L[i].pad_id] = e;
        } else {
            it->second.flag |= 2;
        }
    }

    std::vector<double> sig, shaped, smp;
    for (std::map<int, Entry>::const_iterator it = uniq.begin(); it != uniq.end(); ++it) {
        const PadCharge& pc = *it->second.pc;
        const int nq = (int)pc.qt.size();
        if (nq == 0) continue;
        if (pc.k0 < 0 || pc.k0 + nq > m_n_bins)
            throw std::runtime_error("Readout::process : signal d'un pad hors de la fenetre de bins");

        // signal sur toute la grille de lecture, mise en forme
        sig.assign(m_grid, 0.0);
        for (int k = 0; k < nq; ++k) sig[pc.k0 + k] = pc.qt[k];
        M.shape(sig, m_grid, shaped);

        // bruit de la sortie du preampli, puis echantillonnage et bruit du CAN
        double ser_scale = 1.0;
        if (m_rp.noise_on && !m_pn.empty()) {
            std::map<int, PadNoise>::const_iterator pn = m_pn.find(pc.pad_id);
            if (pn == m_pn.end()) throw std::runtime_error("Readout::process : pad absent de la table de bruit par pad");
            ser_scale = pn->second.ser_scale;
        }
        if (m_rp.noise_on) M.add_filtered_noise(rng, shaped, ser_scale);
        M.decimate(shaped, out.phase, smp);
        if (m_rp.noise_on) M.add_adc_noise(rng, smp);
        const int ns = (int)smp.size();

        // seuil de lecture : n_consec echantillons consecutifs au-dessus du seuil, testes sur la seule
        // grille physique (bin = phase + j * R < n_bins) : la queue de h et la marge de la grille de
        // lecture ne sont pas du temps de derive, elles ne servent qu'a ne pas couper la fenetre ecrite
        if (m_rp.noise_on) {
            int ns_thr = ceil_div(m_n_bins - out.phase, R);
            if (ns_thr > ns) ns_thr = ns;
            int run = 0;
            bool pass = false;
            for (int j = 0; j < ns_thr; ++j) {
                if (smp[j] > m_thr) { if (++run >= m_rp.n_consec) { pass = true; break; } }
                else run = 0;
            }
            if (!pass) continue;
        }

        // fenetre ecrite : signal vrai + queue complete de h + marge, en bins puis en echantillons CAN
        const int first_bin = pc.k0 - m_margin;
        const int last_bin  = pc.k0 + nq - 1 + m_tail + m_margin;
        int j_start = ceil_div(first_bin - out.phase, R);
        int j_end   = floor_div(last_bin - out.phase, R);
        if (j_start < 0)     j_start = 0;
        if (j_end > ns - 1)  j_end = ns - 1;
        if (j_end < j_start) continue;

        ReadoutPad rp;
        rp.pad_id = pc.pad_id;
        rp.x = pc.x;
        rp.y = pc.y;
        rp.q = pc.q;
        rp.flag = it->second.flag;
        rp.j0 = j_start;
        rp.samples.assign(smp.begin() + j_start, smp.begin() + j_end + 1);
        out.pads.push_back(rp);
    }

    // pads fantomes : chaque pad absent de la liste vraie est lu avec la probabilite p
    if (m_rp.noise_on && m_ghost != 0 && m_ghost->prob() > 0.0) {
        GaussRng grng(ghost_seed(seed));
        const double p = m_ghost->prob();
        const int win = m_ghost->window_samples();
        std::vector<double> gs;
        for (size_t is = 0; is < m_sites.size(); ++is) {
            if (uniq.find(m_sites[is].pad_id) != uniq.end()) continue;   // pad du signal : deja traite
            double gscale = 1.0;
            double p_pad = p;
            if (!m_pn.empty()) {
                std::map<int, PadNoise>::const_iterator pn = m_pn.find(m_sites[is].pad_id);
                if (pn == m_pn.end()) throw std::runtime_error("Readout::process : pad absent de la table de bruit par pad");
                gscale = pn->second.enc_scale;
                p_pad  = m_ghost->prob_for_scale(gscale);
            }
            if (grng.uniform() >= p_pad) continue;

            int jf = 0;
            m_ghost->draw(grng, out.phase, gs, jf, gscale);
            const int ns = (int)gs.size();
            int j_start = jf - win / 2;
            if (j_start + win > ns) j_start = ns - win;
            if (j_start < 0) j_start = 0;
            int j_end = j_start + win - 1;
            if (j_end > ns - 1) j_end = ns - 1;

            ReadoutPad rp;
            rp.pad_id = m_sites[is].pad_id;
            rp.x = m_sites[is].x;
            rp.y = m_sites[is].y;
            rp.q = 0.0;
            rp.flag = 0;
            rp.j0 = j_start;
            rp.samples.assign(gs.begin() + j_start, gs.begin() + j_end + 1);
            out.pads.push_back(rp);
            ++out.n_ghost;
        }
        if (out.n_ghost > 0) std::sort(out.pads.begin(), out.pads.end(), by_pad_id);
    }
}
