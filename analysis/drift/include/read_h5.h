#pragma once
#include "event.h"
#include <hdf5.h>
#include <string>
#include <vector>

class ReaderH5 {
public:
    ReaderH5(const std::string& filepath);
    ~ReaderH5();

    hsize_t nEvents() const { return m_nEvents; }

    // Lit [offset, offset+count[ événements, retourne le vecteur rempli
    // Ajuste count si on dépasse la fin du fichier
    std::vector<Event> readChunk(hsize_t offset, hsize_t count);

private:
    hid_t   m_file;
    hsize_t m_nEvents;

    // Helpers lecture VL pour un fragment (group "L" ou "H")
    void readFragmentChunk(const std::string& grp,
                           hsize_t offset, hsize_t count,
                           std::vector<Fragment>& frags);
};
