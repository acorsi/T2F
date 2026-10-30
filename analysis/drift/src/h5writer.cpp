#include "h5writer.hpp"

// ---------------------------------------------------------------------------
// Constructeur / Destructeur
// ---------------------------------------------------------------------------

H5Writer::H5Writer(const std::string& filepath, hsize_t n_events)
    : m_n_events(n_events)
{
    H5Eset_auto(H5E_DEFAULT, nullptr, nullptr);

    m_file_id = H5Fcreate(filepath.c_str(), H5F_ACC_TRUNC,
                           H5P_DEFAULT, H5P_DEFAULT);
    if (m_file_id < 0)
        throw std::runtime_error("H5Writer: impossible de créer " + filepath);
}

H5Writer::~H5Writer() {
    if (m_file_id >= 0)
        H5Fclose(m_file_id);
}

H5Writer::H5Writer(H5Writer&& other) noexcept
    : m_file_id(other.m_file_id), m_n_events(other.m_n_events)
{
    other.m_file_id = -1;
}

H5Writer& H5Writer::operator=(H5Writer&& other) noexcept {
    if (this != &other) {
        if (m_file_id >= 0) H5Fclose(m_file_id);
        m_file_id       = other.m_file_id;
        m_n_events      = other.m_n_events;
        other.m_file_id = -1;
    }
    return *this;
}

// ---------------------------------------------------------------------------
// Helpers internes
// ---------------------------------------------------------------------------

void H5Writer::ensure_groups(const std::string& dataset_path) {
    // Crée récursivement les groupes intermédiaires
    // Ex: "/L/KE" → crée "/L" si absent
    std::string path = dataset_path;
    size_t pos = 1;  // ignore le '/' initial
    while ((pos = path.find('/', pos)) != std::string::npos) {
        std::string group = path.substr(0, pos);
        if (!H5Lexists(m_file_id, group.c_str(), H5P_DEFAULT)) {
            hid_t gid = H5Gcreate2(m_file_id, group.c_str(),
                                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if (gid < 0)
                throw std::runtime_error("H5Writer: impossible de créer le groupe " + group);
            H5Gclose(gid);
        }
        ++pos;
    }
}

hid_t H5Writer::open_dataset(const std::string& path) const {
    hid_t dset = H5Dopen2(m_file_id, path.c_str(), H5P_DEFAULT);
    if (dset < 0)
        throw std::runtime_error("H5Writer: dataset introuvable : " + path
                                 + " (add_dataset appelé ?)");
    return dset;
}

// ---------------------------------------------------------------------------
// Déclaration VLEN
// ---------------------------------------------------------------------------

void H5Writer::add_vlen_dataset(const std::string& path, hsize_t size) {
    ensure_groups(path);

    hid_t vlen_type = H5Tvlen_create(H5T_NATIVE_DOUBLE);

    hsize_t dims    = (size > 0) ? size : m_n_events;
    hsize_t maxdims = dims;
    hid_t space = H5Screate_simple(1, &dims, &maxdims);
    hid_t dset  = H5Dcreate2(m_file_id, path.c_str(),
                              vlen_type, space,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Tclose(vlen_type);

    if (dset < 0) {
        H5Sclose(space);
        throw std::runtime_error("H5Writer::add_vlen_dataset : échec création " + path);
    }
    H5Dclose(dset);
    H5Sclose(space);
}

// ---------------------------------------------------------------------------
// Écriture VLEN
// ---------------------------------------------------------------------------

void H5Writer::write_vlen_chunk(const std::string& path,
                                 const std::vector<std::vector<double>>& data,
                                 hsize_t start)
{
    hid_t dset   = open_dataset(path);
    hid_t fspace = H5Dget_space(dset);

    hsize_t count  = data.size();
    hsize_t offset = start;
    H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &offset, nullptr, &count, nullptr);
    hid_t mspace = H5Screate_simple(1, &count, nullptr);

    // Construction du buffer hvl_t (pas de copie, pointeurs vers les données)
    std::vector<hvl_t> buf(count);
    for (hsize_t i = 0; i < count; ++i) {
        buf[i].len = data[i].size();
        buf[i].p   = const_cast<double*>(data[i].data());
    }

    hid_t mem_type = H5Tvlen_create(H5T_NATIVE_DOUBLE);
    herr_t status  = H5Dwrite(dset, mem_type, mspace, fspace,
                               H5P_DEFAULT, buf.data());
    H5Tclose(mem_type);
    H5Sclose(mspace);
    H5Sclose(fspace);
    H5Dclose(dset);

    if (status < 0)
        throw std::runtime_error("H5Writer::write_vlen_chunk : H5Dwrite échoué sur " + path);
}

void H5Writer::write_vlen_all(const std::string& path,
                               const std::vector<std::vector<double>>& data)
{
    write_vlen_chunk(path, data, 0);
}

// ---------------------------------------------------------------------------
// Attributs
// ---------------------------------------------------------------------------

hid_t H5Writer::open_or_create_group(const std::string& path) {
    // Crée les groupes intermédiaires si nécessaire
    ensure_groups(path + "/_dummy");  // trick pour créer le groupe lui-même

    hid_t gid = H5Gopen2(m_file_id, path.c_str(), H5P_DEFAULT);
    if (gid < 0) {
        // Le groupe n'existe pas encore, on le crée
        gid = H5Gcreate2(m_file_id, path.c_str(),
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (gid < 0)
            throw std::runtime_error("H5Writer::open_or_create_group : échec pour " + path);
    }
    return gid;
}

void H5Writer::write_attribute_string(const std::string& obj_path,
                                       const std::string& attr_name,
                                       const std::string& value) {
    hid_t group = open_or_create_group(obj_path);

    hid_t str_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(str_type, H5T_VARIABLE);
    H5Tset_strpad(str_type, H5T_STR_NULLTERM);

    hid_t aspace = H5Screate(H5S_SCALAR);
    hid_t attr   = H5Acreate2(group, attr_name.c_str(), str_type,
                               aspace, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) {
        H5Sclose(aspace);
        H5Tclose(str_type);
        H5Gclose(group);
        throw std::runtime_error("H5Writer::write_attribute_string : échec création "
                                 + obj_path + "/" + attr_name);
    }

    const char* cstr = value.c_str();
    herr_t status = H5Awrite(attr, str_type, &cstr);

    H5Aclose(attr);
    H5Sclose(aspace);
    H5Tclose(str_type);
    H5Gclose(group);

    if (status < 0)
        throw std::runtime_error("H5Writer::write_attribute_string : H5Awrite échoué sur "
                                 + obj_path + "/" + attr_name);
}
