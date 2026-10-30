#include "h5reader.hpp"
#include <sstream>

// ---------------------------------------------------------------------------
// Constructeur / Destructeur
// ---------------------------------------------------------------------------

H5Reader::H5Reader(const std::string& filepath) {
    // Désactive les messages d'erreur HDF5 vers stderr (on lève des exceptions)
    H5Eset_auto(H5E_DEFAULT, nullptr, nullptr);

    m_file_id = H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (m_file_id < 0)
        throw std::runtime_error("H5Reader: impossible d'ouvrir " + filepath);
}

H5Reader::~H5Reader() {
    if (m_file_id >= 0)
        H5Fclose(m_file_id);
}

H5Reader::H5Reader(H5Reader&& other) noexcept
    : m_file_id(other.m_file_id)
{
    other.m_file_id = -1;
}

H5Reader& H5Reader::operator=(H5Reader&& other) noexcept {
    if (this != &other) {
        if (m_file_id >= 0) H5Fclose(m_file_id);
        m_file_id       = other.m_file_id;
        other.m_file_id = -1;
    }
    return *this;
}

// ---------------------------------------------------------------------------
// Helper : ouvre un dataset, lève une exception si introuvable
// ---------------------------------------------------------------------------

hid_t H5Reader::open_dataset(const std::string& path) const {
    hid_t dset = H5Dopen2(m_file_id, path.c_str(), H5P_DEFAULT);
    if (dset < 0)
        throw std::runtime_error("H5Reader: dataset introuvable : " + path);
    return dset;
}

// ---------------------------------------------------------------------------
// Introspection
// ---------------------------------------------------------------------------

// Callback pour H5Literate
namespace {
struct LsData {
    std::vector<std::string>* names;
};

herr_t ls_callback(hid_t /*loc*/, const char* name,
                   const H5L_info_t* /*info*/, void* op_data) {
    auto* d = static_cast<LsData*>(op_data);
    d->names->emplace_back(name);
    return 0;
}
} // namespace

std::vector<std::string> H5Reader::ls(const std::string& group_path) const {
    hid_t gid = H5Gopen2(m_file_id, group_path.c_str(), H5P_DEFAULT);
    if (gid < 0)
        throw std::runtime_error("H5Reader::ls : groupe introuvable : " + group_path);

    std::vector<std::string> names;
    LsData data{&names};
    H5Literate(gid, H5_INDEX_NAME, H5_ITER_NATIVE, nullptr, ls_callback, &data);
    H5Gclose(gid);
    return names;
}

hsize_t H5Reader::size(const std::string& path) const {
    hid_t dset   = open_dataset(path);
    hid_t fspace = H5Dget_space(dset);
    hsize_t dims;
    H5Sget_simple_extent_dims(fspace, &dims, nullptr);
    H5Sclose(fspace);
    H5Dclose(dset);
    return dims;
}

bool H5Reader::is_vlen(const std::string& path) const {
    hid_t dset  = open_dataset(path);
    hid_t dtype = H5Dget_type(dset);
    bool vlen   = (H5Tget_class(dtype) == H5T_VLEN);
    H5Tclose(dtype);
    H5Dclose(dset);
    return vlen;
}

// ---------------------------------------------------------------------------
// Lecture VLEN
// ---------------------------------------------------------------------------

std::vector<double> H5Reader::read_vlen(const std::string& path, hsize_t i) const {
    auto result = read_vlen_chunk(path, i, 1);
    return std::move(result[0]);
}

std::vector<std::vector<double>> H5Reader::read_vlen_chunk(
    const std::string& path, hsize_t start, hsize_t n) const
{
    hid_t dset  = open_dataset(path);
    hid_t dtype = H5Dget_type(dset);

    if (H5Tget_class(dtype) != H5T_VLEN) {
        H5Tclose(dtype);
        H5Dclose(dset);
        throw std::runtime_error("H5Reader::read_vlen : dataset non VLEN : " + path);
    }

    hid_t fspace = H5Dget_space(dset);
    hsize_t total;
    H5Sget_simple_extent_dims(fspace, &total, nullptr);

    if (start + n > total) {
        H5Sclose(fspace);
        H5Tclose(dtype);
        H5Dclose(dset);
        throw std::runtime_error("H5Reader::read_vlen_chunk : indices hors limites");
    }

    // Sélection hyperslab
    hsize_t offset = start, count = n;
    H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &offset, nullptr, &count, nullptr);
    hid_t mspace = H5Screate_simple(1, &count, nullptr);

    // Type mémoire : VLEN de double natif
    hid_t mem_dtype = H5Tvlen_create(H5T_NATIVE_DOUBLE);

    std::vector<hvl_t> buf(n);
    herr_t status = H5Dread(dset, mem_dtype, mspace, fspace, H5P_DEFAULT, buf.data());

    if (status < 0) {
        H5Tclose(mem_dtype);
        H5Sclose(mspace);
        H5Sclose(fspace);
        H5Tclose(dtype);
        H5Dclose(dset);
        throw std::runtime_error("H5Reader::read_vlen_chunk : H5Dread échoué sur " + path);
    }

    // Copie dans des vecteurs C++ propres
    std::vector<std::vector<double>> result(n);
    for (hsize_t k = 0; k < n; ++k) {
        auto* ptr = static_cast<double*>(buf[k].p);
        result[k].assign(ptr, ptr + buf[k].len);
    }

    // Libération mémoire HDF5 (obligatoire pour VLEN)
    // H5Treclaim depuis HDF5 >= 1.12 ; H5Dvlen_reclaim sur les versions antérieures
#if H5_VERSION_GE(1, 12, 0)
    H5Treclaim(mem_dtype, mspace, H5P_DEFAULT, buf.data());
#else
    H5Dvlen_reclaim(mem_dtype, mspace, H5P_DEFAULT, buf.data());
#endif

    H5Tclose(mem_dtype);
    H5Sclose(mspace);
    H5Sclose(fspace);
    H5Tclose(dtype);
    H5Dclose(dset);

    return result;
}

// ---------------------------------------------------------------------------
// Attributs
// ---------------------------------------------------------------------------

hid_t H5Reader::open_object(const std::string& path) const {
    hid_t obj = H5Oopen(m_file_id, path.c_str(), H5P_DEFAULT);
    if (obj < 0)
        throw std::runtime_error("H5Reader::open_object : objet introuvable : " + path);
    return obj;
}

std::string H5Reader::read_attribute_string(const std::string& obj_path,
                                             const std::string& attr_name) const {
    hid_t obj  = open_object(obj_path);
    hid_t attr = H5Aopen(obj, attr_name.c_str(), H5P_DEFAULT);
    if (attr < 0) {
        H5Oclose(obj);
        throw std::runtime_error("H5Reader::read_attribute_string : attribut introuvable : "
                                 + obj_path + "/" + attr_name);
    }

    hid_t dtype = H5Aget_type(attr);
    hid_t mem_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(mem_type, H5T_VARIABLE);
    H5Tset_strpad(mem_type, H5T_STR_NULLTERM);

    char* buf = nullptr;
    herr_t status = H5Aread(attr, mem_type, &buf);

    std::string result;
    if (status >= 0 && buf) {
        result = buf;
        H5free_memory(buf);
    }

    H5Tclose(mem_type);
    H5Tclose(dtype);
    H5Aclose(attr);
    H5Oclose(obj);

    if (status < 0)
        throw std::runtime_error("H5Reader::read_attribute_string : H5Aread échoué sur "
                                 + obj_path + "/" + attr_name);
    return result;
}
