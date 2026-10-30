#pragma once

#include <hdf5.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>

// ---------------------------------------------------------------------------
// H5Reader — lecture générique de fichiers HDF5 (scalaires 1D + VLEN)
//
// Thread-safety : instancier un H5Reader PAR THREAD (lecture simultanée
// sur le même fichier est supportée par HDF5 en lecture seule).
// ---------------------------------------------------------------------------

class H5Reader {
public:
    explicit H5Reader(const std::string& filepath);
    ~H5Reader();

    // Non-copiable, non-assignable (handle HDF5 non partageable)
    H5Reader(const H5Reader&)            = delete;
    H5Reader& operator=(const H5Reader&) = delete;

    // Déplaceable
    H5Reader(H5Reader&&) noexcept;
    H5Reader& operator=(H5Reader&&) noexcept;

    // -----------------------------------------------------------------------
    // Introspection
    // -----------------------------------------------------------------------

    // Liste les membres (groupes et datasets) d'un groupe
    std::vector<std::string> ls(const std::string& group_path = "/") const;

    // Nombre d'éléments du dataset (première dimension)
    hsize_t size(const std::string& dataset_path) const;

    // true si le dataset est VLEN
    bool is_vlen(const std::string& dataset_path) const;

    // -----------------------------------------------------------------------
    // Lecture scalaire 1D  (int32, float32, float64 → cast vers T)
    // Lève std::runtime_error si le dataset est VLEN.
    // -----------------------------------------------------------------------

    // Un seul élément à l'index i
    template<typename T>
    T read(const std::string& dataset_path, hsize_t i) const;

    // Chunk de n éléments à partir de l'index start
    template<typename T>
    std::vector<T> read_chunk(const std::string& dataset_path,
                              hsize_t start, hsize_t n) const;

    // Tout le dataset
    template<typename T>
    std::vector<T> read_all(const std::string& dataset_path) const;

    // -----------------------------------------------------------------------
    // Lecture d'attributs HDF5 (scalaires sur groupes ou datasets)
    // Ex: reader.read_attribute<double>("/metadata", "step_size")
    //     reader.read_attribute_string("/metadata", "material")
    // -----------------------------------------------------------------------

    template<typename T>
    T read_attribute(const std::string& obj_path,
                     const std::string& attr_name) const;

    std::string read_attribute_string(const std::string& obj_path,
                                      const std::string& attr_name) const;

    // -----------------------------------------------------------------------
    // Lecture VLEN  (tableau de longueur variable par event)
    // Lève std::runtime_error si le dataset n'est pas VLEN.
    // -----------------------------------------------------------------------

    // Trace de l'event i
    std::vector<double> read_vlen(const std::string& dataset_path,
                                  hsize_t i) const;

    // n traces à partir de start
    std::vector<std::vector<double>> read_vlen_chunk(
        const std::string& dataset_path,
        hsize_t start, hsize_t n) const;

private:
    hid_t m_file_id;

    // Helpers internes
    hid_t open_dataset(const std::string& path) const;
    hid_t open_object(const std::string& path) const;  // groupe ou dataset

    template<typename T>
    static hid_t native_type();
};

// ---------------------------------------------------------------------------
// Implémentation des templates (doit rester dans le header)
// ---------------------------------------------------------------------------

template<typename T>
hid_t H5Reader::native_type() {
    if constexpr (std::is_same_v<T, double>)   return H5T_NATIVE_DOUBLE;
    if constexpr (std::is_same_v<T, float>)    return H5T_NATIVE_FLOAT;
    if constexpr (std::is_same_v<T, int32_t>)  return H5T_NATIVE_INT32;
    if constexpr (std::is_same_v<T, int64_t>)  return H5T_NATIVE_INT64;
    if constexpr (std::is_same_v<T, uint32_t>) return H5T_NATIVE_UINT32;
    if constexpr (std::is_same_v<T, uint64_t>) return H5T_NATIVE_UINT64;
    throw std::runtime_error("H5Reader: type T non supporté");
}

template<typename T>
T H5Reader::read(const std::string& path, hsize_t i) const {
    auto v = read_chunk<T>(path, i, 1);
    return v[0];
}

template<typename T>
std::vector<T> H5Reader::read_chunk(const std::string& path,
                                     hsize_t start, hsize_t n) const {
    hid_t dset = open_dataset(path);

    // Vérification VLEN
    hid_t dtype = H5Dget_type(dset);
    if (H5Tget_class(dtype) == H5T_VLEN) {
        H5Tclose(dtype);
        H5Dclose(dset);
        throw std::runtime_error("H5Reader::read_chunk : dataset VLEN, utiliser read_vlen()");
    }
    H5Tclose(dtype);

    hid_t fspace = H5Dget_space(dset);
    hsize_t total;
    H5Sget_simple_extent_dims(fspace, &total, nullptr);

    if (start + n > total)
        throw std::runtime_error("H5Reader::read_chunk : indices hors limites");

    hsize_t offset = start, count = n;
    H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &offset, nullptr, &count, nullptr);

    hid_t mspace = H5Screate_simple(1, &count, nullptr);

    std::vector<T> buf(n);
    herr_t status = H5Dread(dset, native_type<T>(), mspace, fspace, H5P_DEFAULT, buf.data());

    H5Sclose(mspace);
    H5Sclose(fspace);
    H5Dclose(dset);

    if (status < 0)
        throw std::runtime_error("H5Reader::read_chunk : H5Dread échoué sur " + path);

    return buf;
}

template<typename T>
std::vector<T> H5Reader::read_all(const std::string& path) const {
    hid_t dset   = open_dataset(path);
    hid_t fspace = H5Dget_space(dset);
    hsize_t total;
    H5Sget_simple_extent_dims(fspace, &total, nullptr);
    H5Sclose(fspace);
    H5Dclose(dset);
    return read_chunk<T>(path, 0, total);
}

template<typename T>
T H5Reader::read_attribute(const std::string& obj_path,
                            const std::string& attr_name) const {
    hid_t obj  = open_object(obj_path);
    hid_t attr = H5Aopen(obj, attr_name.c_str(), H5P_DEFAULT);
    if (attr < 0) {
        H5Oclose(obj);
        throw std::runtime_error("H5Reader::read_attribute : attribut introuvable : "
                                 + obj_path + "/" + attr_name);
    }

    T val;
    herr_t status = H5Aread(attr, native_type<T>(), &val);
    H5Aclose(attr);
    H5Oclose(obj);

    if (status < 0)
        throw std::runtime_error("H5Reader::read_attribute : H5Aread échoué sur "
                                 + obj_path + "/" + attr_name);
    return val;
}
