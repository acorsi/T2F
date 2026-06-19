#include "write_h5.h"
#include <stdexcept>
#include <iostream>
#include <vector>
#include <limits>

WriterH5::WriterH5(const std::string& filepath, const std::string& inputPath, hsize_t nEvents)
{
    m_file = H5Fcreate(filepath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (m_file < 0)
        throw std::runtime_error("Cannot create HDF5 file: " + filepath);

    createFragmentDatasets("L", nEvents);
    createFragmentDatasets("H", nEvents);

    copyMetadata(inputPath);

    std::cout << "Created output: " << filepath << "  [" << nEvents << " events]\n";
}

WriterH5::~WriterH5()
{
    H5Fclose(m_file);
}

// ---------------------------------------------------------------------------
void WriterH5::createFragmentDatasets(const std::string& grp, hsize_t nEvents)
{
    hid_t group = H5Gcreate2(m_file, ("/" + grp).c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    auto createScalarInt = [&](const std::string& name) {
        hsize_t dims[1] = { nEvents };
        hid_t space = H5Screate_simple(1, dims, nullptr);
        hid_t dset  = H5Dcreate2(group, name.c_str(), H5T_NATIVE_INT, space,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dclose(dset);
        H5Sclose(space);
    };
    auto createScalarDouble = [&](const std::string& name) {
        hsize_t dims[1] = { nEvents };
        hid_t space = H5Screate_simple(1, dims, nullptr);
        hid_t dset  = H5Dcreate2(group, name.c_str(), H5T_NATIVE_DOUBLE, space,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dclose(dset);
        H5Sclose(space);
    };
    auto create2DInt = [&](const std::string& name, hsize_t dim2) {
        hsize_t dims[2] = { nEvents, dim2 };
        hid_t space = H5Screate_simple(2, dims, nullptr);
        hid_t dset  = H5Dcreate2(group, name.c_str(), H5T_NATIVE_INT, space,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dclose(dset);
        H5Sclose(space);
    };
    auto create2DDouble = [&](const std::string& name, hsize_t dim2) {
        hsize_t dims[2] = { nEvents, dim2 };
        hid_t space = H5Screate_simple(2, dims, nullptr);
        hid_t dset  = H5Dcreate2(group, name.c_str(), H5T_NATIVE_DOUBLE, space,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dclose(dset);
        H5Sclose(space);
    };

    // Scalaires copiés de l'entrée
    createScalarInt("A");
    createScalarInt("Z");
    createScalarInt("n");
    createScalarDouble("KE");

    // Informations angulaires et position d'émission (vérité simulation)
    createScalarDouble("theta");
    createScalarDouble("phi");
    createScalarDouble("pInit_x");
    createScalarDouble("pInit_y");
    createScalarDouble("pInit_z");

    // raw_trace : paddé à MAX_TRACE_LEN
    create2DDouble("raw_trace", MAX_TRACE_LEN);
    createScalarInt("raw_trace_len");

    // proc_trace sparse : pads touchés, paddé à MAX_HITS
    create2DInt("pad_row", MAX_HITS);
    create2DInt("pad_col", MAX_HITS);
    create2DDouble("pad_charge", MAX_HITS);
    createScalarInt("n_hits");

    // Signal temporel Q(t) par pad : longueur variable, format à plat avec offset explicite
    create2DInt("qt_length", MAX_HITS);      // nb d'échantillons (padding 0)
    create2DDouble("qt_tstart", MAX_HITS);   // temps de départ du signal, ns (padding 0)
    create2DInt("qt_offset", MAX_HITS);      // offset de début dans qt_data (padding -1)

    // qt_data : dataset extensible, taille totale inconnue à l'avance
    {
        hsize_t dims[1]    = { 0 };
        hsize_t maxdims[1] = { H5S_UNLIMITED };
        hid_t space = H5Screate_simple(1, dims, maxdims);

        hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
        hsize_t chunkDims[1] = { 1 << 20 };  // chunk de 1M doubles (~8MB)
        H5Pset_chunk(plist, 1, chunkDims);

        hid_t dset = H5Dcreate2(group, "qt_data", H5T_NATIVE_DOUBLE, space,
                                 H5P_DEFAULT, plist, H5P_DEFAULT);
        H5Dclose(dset);
        H5Pclose(plist);
        H5Sclose(space);
    }

    H5Gclose(group);
}

// ---------------------------------------------------------------------------
hsize_t WriterH5::appendQtData(const std::string& grp, const std::vector<double>& data)
{
    if (data.empty()) return 0;

    std::string path = "/" + grp + "/qt_data";
    hid_t dset = H5Dopen2(m_file, path.c_str(), H5P_DEFAULT);

    hid_t space = H5Dget_space(dset);
    hsize_t curDims[1];
    H5Sget_simple_extent_dims(space, curDims, nullptr);
    H5Sclose(space);

    hsize_t startOffset = curDims[0];
    hsize_t newDims[1]  = { curDims[0] + data.size() };
    H5Dset_extent(dset, newDims);

    hid_t newSpace = H5Dget_space(dset);
    hsize_t start[1] = { startOffset };
    hsize_t cnt[1]   = { data.size() };
    H5Sselect_hyperslab(newSpace, H5S_SELECT_SET, start, nullptr, cnt, nullptr);

    hid_t memspace = H5Screate_simple(1, cnt, nullptr);
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, memspace, newSpace, H5P_DEFAULT, data.data());

    H5Sclose(memspace);
    H5Sclose(newSpace);
    H5Dclose(dset);

    return startOffset;
}

// ---------------------------------------------------------------------------
void WriterH5::writeFragmentChunk(const std::string& grp,
                                   const std::vector<Event>& events,
                                   bool isL, hsize_t offset)
{
    hsize_t count = events.size();
    if (count == 0) return;

    auto writeScalarInt = [&](const std::string& name, auto getter) {
        std::string path = "/" + grp + "/" + name;
        hid_t dset  = H5Dopen2(m_file, path.c_str(), H5P_DEFAULT);
        hid_t space = H5Dget_space(dset);
        hsize_t start[1] = { offset }, cnt[1] = { count };
        H5Sselect_hyperslab(space, H5S_SELECT_SET, start, nullptr, cnt, nullptr);
        hid_t memspace = H5Screate_simple(1, cnt, nullptr);

        std::vector<int> buf(count);
        for (hsize_t i = 0; i < count; ++i) {
            const Fragment& f = isL ? events[i].L : events[i].H;
            buf[i] = getter(f);
        }
        H5Dwrite(dset, H5T_NATIVE_INT, memspace, space, H5P_DEFAULT, buf.data());

        H5Sclose(memspace); H5Sclose(space); H5Dclose(dset);
    };

    auto writeScalarDouble = [&](const std::string& name, auto getter) {
        std::string path = "/" + grp + "/" + name;
        hid_t dset  = H5Dopen2(m_file, path.c_str(), H5P_DEFAULT);
        hid_t space = H5Dget_space(dset);
        hsize_t start[1] = { offset }, cnt[1] = { count };
        H5Sselect_hyperslab(space, H5S_SELECT_SET, start, nullptr, cnt, nullptr);
        hid_t memspace = H5Screate_simple(1, cnt, nullptr);

        std::vector<double> buf(count);
        for (hsize_t i = 0; i < count; ++i) {
            const Fragment& f = isL ? events[i].L : events[i].H;
            buf[i] = getter(f);
        }
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, memspace, space, H5P_DEFAULT, buf.data());

        H5Sclose(memspace); H5Sclose(space); H5Dclose(dset);
    };

    writeScalarInt("A", [](const Fragment& f) { return f.A; });
    writeScalarInt("Z", [](const Fragment& f) { return f.Z; });
    writeScalarInt("n", [](const Fragment& f) { return f.n; });
    writeScalarDouble("KE", [](const Fragment& f) { return f.KE; });

    writeScalarDouble("theta",   [](const Fragment& f) { return f.theta; });
    writeScalarDouble("phi",     [](const Fragment& f) { return f.phi; });
    writeScalarDouble("pInit_x", [](const Fragment& f) { return f.pInit[0]; });
    writeScalarDouble("pInit_y", [](const Fragment& f) { return f.pInit[1]; });
    writeScalarDouble("pInit_z", [](const Fragment& f) { return f.pInit[2]; });

    // raw_trace : paddé à MAX_TRACE_LEN
    {
        std::string path = "/" + grp + "/raw_trace";
        hid_t dset  = H5Dopen2(m_file, path.c_str(), H5P_DEFAULT);
        hid_t space = H5Dget_space(dset);
        hsize_t start[2] = { offset, 0 }, cnt[2] = { count, MAX_TRACE_LEN };
        H5Sselect_hyperslab(space, H5S_SELECT_SET, start, nullptr, cnt, nullptr);
        hid_t memspace = H5Screate_simple(2, cnt, nullptr);

        std::vector<double> buf(count * MAX_TRACE_LEN, 0.0);
        for (hsize_t i = 0; i < count; ++i) {
            const Fragment& f = isL ? events[i].L : events[i].H;
            hsize_t len = std::min((hsize_t)f.raw_trace.size(), (hsize_t)MAX_TRACE_LEN);
            for (hsize_t j = 0; j < len; ++j)
                buf[i * MAX_TRACE_LEN + j] = f.raw_trace[j];
        }
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, memspace, space, H5P_DEFAULT, buf.data());

        H5Sclose(memspace); H5Sclose(space); H5Dclose(dset);
    }

    writeScalarInt("raw_trace_len", [](const Fragment& f) {
        return (int)std::min((size_t)MAX_TRACE_LEN, f.raw_trace.size());
    });

    // pad_row, pad_col, pad_charge : paddés à MAX_HITS
    {
        std::string pathRow = "/" + grp + "/pad_row";
        std::string pathCol = "/" + grp + "/pad_col";
        std::string pathChg = "/" + grp + "/pad_charge";

        hid_t dRow = H5Dopen2(m_file, pathRow.c_str(), H5P_DEFAULT);
        hid_t dCol = H5Dopen2(m_file, pathCol.c_str(), H5P_DEFAULT);
        hid_t dChg = H5Dopen2(m_file, pathChg.c_str(), H5P_DEFAULT);

        hid_t sRow = H5Dget_space(dRow);
        hid_t sCol = H5Dget_space(dCol);
        hid_t sChg = H5Dget_space(dChg);

        hsize_t start[2] = { offset, 0 }, cnt[2] = { count, MAX_HITS };
        H5Sselect_hyperslab(sRow, H5S_SELECT_SET, start, nullptr, cnt, nullptr);
        H5Sselect_hyperslab(sCol, H5S_SELECT_SET, start, nullptr, cnt, nullptr);
        H5Sselect_hyperslab(sChg, H5S_SELECT_SET, start, nullptr, cnt, nullptr);

        hid_t memspace = H5Screate_simple(2, cnt, nullptr);

        std::vector<int>    bufRow(count * MAX_HITS, -9999);
        std::vector<int>    bufCol(count * MAX_HITS, -9999);
        std::vector<double> bufChg(count * MAX_HITS, 0.0);

        for (hsize_t i = 0; i < count; ++i) {
            const Fragment& f = isL ? events[i].L : events[i].H;
            hsize_t nHits = std::min((hsize_t)f.proc_trace.size(), (hsize_t)MAX_HITS);
            for (hsize_t j = 0; j < nHits; ++j) {
                bufRow[i * MAX_HITS + j] = f.proc_trace[j].row;
                bufCol[i * MAX_HITS + j] = f.proc_trace[j].col;
                bufChg[i * MAX_HITS + j] = f.proc_trace[j].charge;
            }
        }

        H5Dwrite(dRow, H5T_NATIVE_INT,    memspace, sRow, H5P_DEFAULT, bufRow.data());
        H5Dwrite(dCol, H5T_NATIVE_INT,    memspace, sCol, H5P_DEFAULT, bufCol.data());
        H5Dwrite(dChg, H5T_NATIVE_DOUBLE, memspace, sChg, H5P_DEFAULT, bufChg.data());

        H5Sclose(memspace);
        H5Sclose(sRow); H5Sclose(sCol); H5Sclose(sChg);
        H5Dclose(dRow); H5Dclose(dCol); H5Dclose(dChg);
    }

    writeScalarInt("n_hits", [](const Fragment& f) {
        return (int)std::min((size_t)MAX_HITS, f.proc_trace.size());
    });

    // qt_length, qt_tstart, qt_offset, qt_data : signal temporel par pad
    {
        std::string pathLen    = "/" + grp + "/qt_length";
        std::string pathTstart = "/" + grp + "/qt_tstart";
        std::string pathOffset = "/" + grp + "/qt_offset";
        std::string pathData   = "/" + grp + "/qt_data";

        hid_t dLen = H5Dopen2(m_file, pathLen.c_str(), H5P_DEFAULT);
        hid_t dTs  = H5Dopen2(m_file, pathTstart.c_str(), H5P_DEFAULT);
        hid_t dOff = H5Dopen2(m_file, pathOffset.c_str(), H5P_DEFAULT);

        hid_t sLen = H5Dget_space(dLen);
        hid_t sTs  = H5Dget_space(dTs);
        hid_t sOff = H5Dget_space(dOff);

        hsize_t start[2] = { offset, 0 }, cnt[2] = { count, MAX_HITS };
        H5Sselect_hyperslab(sLen, H5S_SELECT_SET, start, nullptr, cnt, nullptr);
        H5Sselect_hyperslab(sTs,  H5S_SELECT_SET, start, nullptr, cnt, nullptr);
        H5Sselect_hyperslab(sOff, H5S_SELECT_SET, start, nullptr, cnt, nullptr);

        hid_t memspace = H5Screate_simple(2, cnt, nullptr);

        // Taille courante de qt_data avant cet appel (point de départ des offsets)
        hid_t dData = H5Dopen2(m_file, pathData.c_str(), H5P_DEFAULT);
        hid_t sData = H5Dget_space(dData);
        hsize_t qtDataSizeBefore[1];
        H5Sget_simple_extent_dims(sData, qtDataSizeBefore, nullptr);
        H5Sclose(sData);
        H5Dclose(dData);

        std::vector<int>    bufLen(count * MAX_HITS, 0);
        std::vector<double> bufTs(count * MAX_HITS, 0.0);
        std::vector<int>    bufOff(count * MAX_HITS, -9999);
        std::vector<double> qtFlat;  // toutes les valeurs qt à ajouter, dans l'ordre

        hsize_t runningOffset = qtDataSizeBefore[0];
        for (hsize_t i = 0; i < count; ++i) {
            const Fragment& f = isL ? events[i].L : events[i].H;
            hsize_t nHits = std::min((hsize_t)f.proc_trace.size(), (hsize_t)MAX_HITS);
            for (hsize_t j = 0; j < nHits; ++j) {
                const PadHit& hit = f.proc_trace[j];
                bufLen[i * MAX_HITS + j] = (int)hit.qt.size();
                bufTs[i * MAX_HITS + j]  = hit.t_start;
                bufOff[i * MAX_HITS + j] = (int)runningOffset;
                qtFlat.insert(qtFlat.end(), hit.qt.begin(), hit.qt.end());
                runningOffset += hit.qt.size();
            }
        }

        H5Dwrite(dLen, H5T_NATIVE_INT,    memspace, sLen, H5P_DEFAULT, bufLen.data());
        H5Dwrite(dTs,  H5T_NATIVE_DOUBLE, memspace, sTs,  H5P_DEFAULT, bufTs.data());
        H5Dwrite(dOff, H5T_NATIVE_INT,    memspace, sOff, H5P_DEFAULT, bufOff.data());

        H5Sclose(memspace);
        H5Sclose(sLen); H5Sclose(sTs); H5Sclose(sOff);
        H5Dclose(dLen); H5Dclose(dTs); H5Dclose(dOff);

        // Append séquentiel dans qt_data (extensible)
        appendQtData(grp, qtFlat);
    }
}

// ---------------------------------------------------------------------------
void WriterH5::writeChunk(const std::vector<Event>& events, hsize_t offset)
{
    writeFragmentChunk("L", events, true,  offset);
    writeFragmentChunk("H", events, false, offset);
}

// ---------------------------------------------------------------------------
void WriterH5::copyMetadata(const std::string& inputPath)
{
    hid_t inFile = H5Fopen(inputPath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (inFile < 0) {
        std::cerr << "Warning: cannot open input file to copy metadata\n";
        return;
    }

    // H5Ocopy copie récursivement groupe + sous-objets + attributs
    herr_t status = H5Ocopy(inFile, "/metadata", m_file, "/metadata",
                             H5P_DEFAULT, H5P_DEFAULT);
    if (status < 0)
        std::cerr << "Warning: failed to copy /metadata\n";

    H5Fclose(inFile);
}
