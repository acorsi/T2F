#pragma once

#include <hdf5.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>

// ---------------------------------------------------------------------------
// H5Writer — écriture générique de fichiers HDF5 (scalaires 1D + VLEN)
//
// Workflow :
//   1. Construire avec le chemin et le nombre d'events par défaut
//   2. Déclarer les datasets avec add_dataset / add_vlen_dataset
//      (taille par défaut = n_events, ou explicite pour un dataset
//       de taille différente, ex: géométrie de pads)
//   3. Écrire avec write_chunk / write_all / write_vlen_chunk
// ---------------------------------------------------------------------------

class H5Writer {
public:
    explicit H5Writer(const std::string& filepath, hsize_t n_events);
    ~H5Writer();

    H5Writer(const H5Writer&)            = delete;
    H5Writer& operator=(const H5Writer&) = delete;
    H5Writer(H5Writer&&) noexcept;
    H5Writer& operator=(H5Writer&&) noexcept;

    // -----------------------------------------------------------------------
    // Déclaration des datasets (avant toute écriture)
    // size = 0 → utilise n_events (comportement par défaut)
    // size > 0 → taille explicite (ex: nombre de pads, indépendant des events)
    // -----------------------------------------------------------------------

    template<typename T>
    void add_dataset(const std::string& dataset_path, hsize_t size = 0);

    void add_vlen_dataset(const std::string& dataset_path, hsize_t size = 0);

    // -----------------------------------------------------------------------
    // Écriture scalaire 1D
    // -----------------------------------------------------------------------

    template<typename T>
    void write_chunk(const std::string& dataset_path,
                     const std::vector<T>& data, hsize_t start);

    template<typename T>
    void write_all(const std::string& dataset_path,
                   const std::vector<T>& data);

    // -----------------------------------------------------------------------
    // Écriture VLEN
    // -----------------------------------------------------------------------

    void write_vlen_chunk(const std::string& dataset_path,
                          const std::vector<std::vector<double>>& data,
                          hsize_t start);

    void write_vlen_all(const std::string& dataset_path,
                        const std::vector<std::vector<double>>& data);

    // -----------------------------------------------------------------------
    // Écriture d'attributs HDF5 (scalaires sur groupes ou datasets)
    // Le groupe est créé automatiquement s'il n'existe pas.
    // Ex: writer.write_attribute<double>("/metadata", "step_size", 0.02);
    // -----------------------------------------------------------------------

    template<typename T>
    void write_attribute(const std::string& obj_path,
                         const std::string& attr_name, T value);

    void write_attribute_string(const std::string& obj_path,
                                const std::string& attr_name,
                                const std::string& value);

private:
    hid_t   m_file_id;
    hsize_t m_n_events;

    hid_t open_dataset(const std::string& path) const;
    void  ensure_groups(const std::string& dataset_path);
    hid_t open_or_create_group(const std::string& group_path);

    template<typename T>
    static hid_t native_type();
};

// ---------------------------------------------------------------------------
// Implémentation des templates
// ---------------------------------------------------------------------------

template<typename T>
hid_t H5Writer::native_type() {
    if constexpr (std::is_same_v<T, double>)   return H5T_NATIVE_DOUBLE;
    if constexpr (std::is_same_v<T, float>)    return H5T_NATIVE_FLOAT;
    if constexpr (std::is_same_v<T, int32_t>)  return H5T_NATIVE_INT32;
    if constexpr (std::is_same_v<T, int64_t>)  return H5T_NATIVE_INT64;
    if constexpr (std::is_same_v<T, uint32_t>) return H5T_NATIVE_UINT32;
    if constexpr (std::is_same_v<T, uint64_t>) return H5T_NATIVE_UINT64;
    throw std::runtime_error("H5Writer: type T non supporté");
}

template<typename T>
void H5Writer::add_dataset(const std::string& path, hsize_t size) {
    ensure_groups(path);

    hsize_t dims    = (size > 0) ? size : m_n_events;
    hsize_t maxdims = dims;
    hid_t space = H5Screate_simple(1, &dims, &maxdims);
    hid_t dset  = H5Dcreate2(m_file_id, path.c_str(),
                              native_type<T>(), space,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0) {
        H5Sclose(space);
        throw std::runtime_error("H5Writer::add_dataset : échec création " + path);
    }
    H5Dclose(dset);
    H5Sclose(space);
}

template<typename T>
void H5Writer::write_chunk(const std::string& path,
                            const std::vector<T>& data, hsize_t start) {
    hid_t dset   = open_dataset(path);
    hid_t fspace = H5Dget_space(dset);

    hsize_t count  = data.size();
    hsize_t offset = start;
    H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &offset, nullptr, &count, nullptr);

    hid_t mspace = H5Screate_simple(1, &count, nullptr);
    herr_t status = H5Dwrite(dset, native_type<T>(), mspace, fspace,
                              H5P_DEFAULT, data.data());
    H5Sclose(mspace);
    H5Sclose(fspace);
    H5Dclose(dset);

    if (status < 0)
        throw std::runtime_error("H5Writer::write_chunk : H5Dwrite échoué sur " + path);
}

template<typename T>
void H5Writer::write_all(const std::string& path, const std::vector<T>& data) {
    write_chunk<T>(path, data, 0);
}

template<typename T>
void H5Writer::write_attribute(const std::string& obj_path,
                               const std::string& attr_name, T value) {
    hid_t group = open_or_create_group(obj_path);

    hid_t aspace = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(group, attr_name.c_str(), native_type<T>(),
                            aspace, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) {
        H5Sclose(aspace);
        H5Gclose(group);
        throw std::runtime_error("H5Writer::write_attribute : échec création "
                                 + obj_path + "/" + attr_name);
    }

    herr_t status = H5Awrite(attr, native_type<T>(), &value);

    H5Aclose(attr);
    H5Sclose(aspace);
    H5Gclose(group);

    if (status < 0)
        throw std::runtime_error("H5Writer::write_attribute : H5Awrite échoué sur "
                                 + obj_path + "/" + attr_name);
}
