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
