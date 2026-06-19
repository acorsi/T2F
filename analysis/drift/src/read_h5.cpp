#include "read_h5.h"
#include <stdexcept>
#include <iostream>

ReaderH5::ReaderH5(const std::string& filepath) {
    m_file = H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (m_file < 0)
        throw std::runtime_error("Cannot open HDF5 file: " + filepath);

    hid_t dset  = H5Dopen2(m_file, "/L/A", H5P_DEFAULT);
    hid_t space = H5Dget_space(dset);
    hsize_t dims[1];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    m_nEvents = dims[0];
    H5Sclose(space);
    H5Dclose(dset);

    std::cout << "Opened: " << filepath
              << "  [" << m_nEvents << " events]\n";
}

ReaderH5::~ReaderH5() {
    H5Fclose(m_file);
}

void ReaderH5::readFragmentChunk(const std::string& grp,
                                  hsize_t offset, hsize_t count,
                                  std::vector<Fragment>& frags)
{
    frags.resize(count);

    // --- Scalaires entiers : A, Z, n ---
    auto readInt = [&](const std::string& name, int Fragment::* field) {
        std::string path = "/" + grp + "/" + name;
        hid_t dset  = H5Dopen2(m_file, path.c_str(), H5P_DEFAULT);
        hid_t space = H5Dget_space(dset);
        hsize_t start[1] = { offset }, cnt[1] = { count };
        H5Sselect_hyperslab(space, H5S_SELECT_SET, start, nullptr, cnt, nullptr);
        hid_t memspace = H5Screate_simple(1, cnt, nullptr);
        std::vector<int> buf(count);
        H5Dread(dset, H5T_NATIVE_INT, memspace, space, H5P_DEFAULT, buf.data());
        for (hsize_t i = 0; i < count; ++i) frags[i].*field = buf[i];
        H5Sclose(memspace); H5Sclose(space); H5Dclose(dset);
    };

    readInt("A", &Fragment::A);
    readInt("Z", &Fragment::Z);
    readInt("n", &Fragment::n);

    // --- KE : double ---
    {
        std::string path = "/" + grp + "/KE";
        hid_t dset  = H5Dopen2(m_file, path.c_str(), H5P_DEFAULT);
        hid_t space = H5Dget_space(dset);
        hsize_t start[1] = { offset }, cnt[1] = { count };
        H5Sselect_hyperslab(space, H5S_SELECT_SET, start, nullptr, cnt, nullptr);
        hid_t memspace = H5Screate_simple(1, cnt, nullptr);
        std::vector<double> buf(count);
        H5Dread(dset, H5T_NATIVE_DOUBLE, memspace, space, H5P_DEFAULT, buf.data());
        for (hsize_t i = 0; i < count; ++i) frags[i].KE = buf[i];
        H5Sclose(memspace); H5Sclose(space); H5Dclose(dset);
    }

    // --- raw_trace : VL de doubles ---
    {
        std::string path = "/" + grp + "/raw_trace";
        hid_t dset  = H5Dopen2(m_file, path.c_str(), H5P_DEFAULT);
        hid_t space = H5Dget_space(dset);
        hid_t vtype = H5Dget_type(dset);
        hsize_t start[1] = { offset }, cnt[1] = { count };
        H5Sselect_hyperslab(space, H5S_SELECT_SET, start, nullptr, cnt, nullptr);
        hid_t memspace = H5Screate_simple(1, cnt, nullptr);
        std::vector<hvl_t> vl(count);
        H5Dread(dset, vtype, memspace, space, H5P_DEFAULT, vl.data());
        for (hsize_t i = 0; i < count; ++i) {
            double* ptr = static_cast<double*>(vl[i].p);
            frags[i].raw_trace.assign(ptr, ptr + vl[i].len);
        }
        H5Dvlen_reclaim(vtype, memspace, H5P_DEFAULT, vl.data());
        H5Sclose(memspace); H5Tclose(vtype); H5Sclose(space); H5Dclose(dset);
    }
}

std::vector<Event> ReaderH5::readChunk(hsize_t offset, hsize_t count)
{
    if (offset >= m_nEvents) return {};
    if (offset + count > m_nEvents) count = m_nEvents - offset;

    std::vector<Fragment> fragsL, fragsH;
    readFragmentChunk("L", offset, count, fragsL);
    readFragmentChunk("H", offset, count, fragsH);

    std::vector<Event> events(count);
    for (hsize_t i = 0; i < count; ++i) {
        events[i].L = std::move(fragsL[i]);
        events[i].H = std::move(fragsH[i]);
    }
    return events;
}
