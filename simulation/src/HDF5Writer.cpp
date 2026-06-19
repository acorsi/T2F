#include "HDF5Writer.h"
#include <iostream>
#include <ctime>
#include <algorithm>

// ===== CLASSE POUR SUPPRIMER LES ERREURS HDF5 ATTENDUES =====
class HDF5ErrorSuppressor {
    H5E_auto2_t old_func_;
    void* old_client_data_;
    
public:
    HDF5ErrorSuppressor() {
        H5::Exception::getAutoPrint(old_func_, &old_client_data_);
        H5::Exception::dontPrint();
    }
    
    ~HDF5ErrorSuppressor() {
        H5::Exception::setAutoPrint(old_func_, old_client_data_);
    }
};

HDF5Writer::HDF5Writer(const std::string& filepath, const HDF5WriteConfig& config)
    : filepath_(filepath), config_(config), file_(nullptr), 
      bufferSize_(0) {
    
    std::cout << "[HDF5Writer] Will create: " << filepath_ << std::endl;
    initialize() ;
}

HDF5Writer::~HDF5Writer() {
    finalize();
}

void HDF5Writer::initialize() {
    try {
        std::cout << "[HDF5Writer] Initializing file..." << std::endl;
        
        // Créer le fichier HDF5
        file_ = new H5::H5File(filepath_, H5F_ACC_TRUNC);
        
        // Créer la structure de groupes
        file_->createGroup("/metadata");
        
        // Métadonnées globales
        std::time_t now = std::time(nullptr);
        char timeStr[100];
        std::strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
        
        addMetadata("created", std::string(timeStr));
        addMetadata("format_version", "1.0");
        addMetadata("program", "T2F");
        
        std::cout << "[HDF5Writer] Initialized successfully" << std::endl;
        
    } catch (H5::Exception& e) {
        throw std::runtime_error("HDF5 initialization failed: " + std::string(e.getCDetailMsg()));
    }
}

// ===== HELPER FUNCTION (pas membre de classe) =====
template<typename T>
static void writeColumnHelper(H5::H5File* file,
                              const std::string& datasetPath, 
                              const std::vector<T>& data,
                              const H5::PredType& h5type) {
    if (data.empty()) return;

    HDF5ErrorSuppressor suppress;
    
    H5::DataSet* dataset = nullptr;
    
    try {
        // ===== EXTRAIRE ET CRÉER LE GROUPE PARENT =====
        size_t pos = datasetPath.find_last_of('/');
        std::string groupPath = "/";
        std::string datasetName = datasetPath;
        
        if (pos != std::string::npos && pos > 0) {
            groupPath = datasetPath.substr(0, pos);
            datasetName = datasetPath.substr(pos + 1);
        }
        
        // Ouvrir ou créer le groupe
        H5::Group group;
        try {
            group = file->openGroup(groupPath);
        } catch (H5::Exception&) {
            // Groupe n'existe pas, le créer
            group = file->createGroup(groupPath);
	    //            std::cout << "[HDF5] Created group: " << groupPath << std::endl;
        }
        // ==============================================
        
        // Vérifier si le dataset existe dans ce groupe
        bool datasetExists = false;
        try {
            dataset = new H5::DataSet(group.openDataSet(datasetName));
            datasetExists = true;
        } catch (H5::Exception&) {
            datasetExists = false;
        }

	hsize_t mysize ;
        if (!datasetExists) {
            // Créer nouveau dataset (extensible)
            hsize_t dims[1] = { data.size() };
            hsize_t maxdims[1] = { H5S_UNLIMITED };
            H5::DataSpace dataspace(1, dims, maxdims);
            
            // Propriétés : chunking requis pour extensible
            H5::DSetCreatPropList plist;
            hsize_t chunk_dims[1] = { std::min(size_t(1000), data.size()) };
            plist.setChunk(1, chunk_dims);
            
            dataset = new H5::DataSet(
                group.createDataSet(datasetName, h5type, dataspace, plist)
            );
            
            // Écrire les données
            dataset->write(data.data(), h5type);
            
            // std::cout << "[HDF5] Created dataset: " << datasetPath 
            //           << " (size=" << data.size() << ")" << std::endl;
            
        } else {
            // Dataset existe : étendre et ajouter
            // Taille actuelle
            H5::DataSpace fspace = dataset->getSpace();
            hsize_t current_size[1];
            fspace.getSimpleExtentDims(current_size);
            
            // Étendre
            hsize_t new_size[1] = { current_size[0] + data.size() };
            dataset->extend(new_size);
            
            // Sélectionner la région à écrire (hyperslab)
            fspace = dataset->getSpace();
            hsize_t offset[1] = { current_size[0] };
            hsize_t count[1] = { data.size() };
            fspace.selectHyperslab(H5S_SELECT_SET, count, offset);
            
            // Dataspace mémoire
            H5::DataSpace memspace(1, count);
            
            // Écrire
            dataset->write(data.data(), h5type, memspace, fspace);
            
            // std::cout << "[HDF5] Extended dataset: " << datasetPath 
            //           << " (+" << data.size() << " → " << new_size[0] << ")" << std::endl;
        }
	
        delete dataset;
        group.close();
        
    } catch (H5::Exception& e) {
        if (dataset) delete dataset;
        std::cerr << "[HDF5] Error writing " << datasetPath << ": " 
                  << e.getCDetailMsg() << std::endl;
        throw;
    }
}

// ===== NOUVEAU HELPER POUR Variable Length vector =====
static void writeVarLenColumnHelper(H5::H5File* file,
                                   const std::string& datasetPath,
                                   const std::vector<std::vector<double>>& data) {
    if (data.empty()) return;
    
    HDF5ErrorSuppressor suppress;
    
    H5::DataSet* dataset = nullptr;
    
    try {
        // Extraire groupe parent et nom dataset
        size_t pos = datasetPath.find_last_of('/');
        std::string groupPath = "/";
        std::string datasetName = datasetPath;
        
        if (pos != std::string::npos && pos > 0) {
            groupPath = datasetPath.substr(0, pos);
            datasetName = datasetPath.substr(pos + 1);
        }
        
        // Ouvrir ou créer le groupe
        H5::Group group;
        try {
            group = file->openGroup(groupPath);
        } catch (H5::Exception&) {
            group = file->createGroup(groupPath);
        }
        
        // Créer le type variable-length
        H5::VarLenType vlenType(&H5::PredType::NATIVE_DOUBLE);
        
        // Préparer les données au format HDF5 VLen
        std::vector<hvl_t> vlenData(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            vlenData[i].len = data[i].size();
            if (data[i].size() > 0) {
                vlenData[i].p = const_cast<double*>(data[i].data());
            } else {
                vlenData[i].p = nullptr;
            }
        }
        
        // Vérifier si dataset existe
        bool datasetExists = false;
        try {
            dataset = new H5::DataSet(group.openDataSet(datasetName));
            datasetExists = true;
        } catch (H5::Exception&) {
            datasetExists = false;
        }
        
        if (!datasetExists) {
            // Créer nouveau dataset VLen extensible
            hsize_t dims[1] = { data.size() };
            hsize_t maxdims[1] = { H5S_UNLIMITED };
            H5::DataSpace dataspace(1, dims, maxdims);
            
            H5::DSetCreatPropList plist;
            hsize_t chunk_dims[1] = { std::min(size_t(100), data.size()) };
            plist.setChunk(1, chunk_dims);
            
            dataset = new H5::DataSet(
                group.createDataSet(datasetName, vlenType, dataspace, plist)
            );
            
            dataset->write(vlenData.data(), vlenType);
            
            // std::cout << "[HDF5] Created VLen dataset: " << datasetPath 
            //           << " (size=" << data.size() << ")" << std::endl;
            
        } else {
            // Étendre dataset existant
            H5::DataSpace fspace = dataset->getSpace();
            hsize_t current_size[1];
            fspace.getSimpleExtentDims(current_size);
            
            hsize_t new_size[1] = { current_size[0] + data.size() };
            dataset->extend(new_size);
            
            fspace = dataset->getSpace();
            hsize_t offset[1] = { current_size[0] };
            hsize_t count[1] = { data.size() };
            fspace.selectHyperslab(H5S_SELECT_SET, count, offset);
            
            H5::DataSpace memspace(1, count);
            dataset->write(vlenData.data(), vlenType, memspace, fspace);
            
            // std::cout << "[HDF5] Extended VLen dataset: " << datasetPath 
            //           << " (+" << data.size() << " → " << new_size[0] << ")" << std::endl;
        }
        
        delete dataset;
        group.close();
        
    } catch (H5::Exception& e) {
        if (dataset) delete dataset;
        throw;
    }
}

void HDF5Writer::writeFragmentPair(const FragmentPair& pair, const int event_ID) {
    
    const Fragment& light = pair.getLightFragment();
    const Fragment& heavy = pair.getHeavyFragment();
    int eventID = pair.getEventID();

    //const FeatureSet& featLight = light.getFeatures();
    //const FeatureSet& featHeavy = heavy.getFeatures();
    
    // Buffer light fragment
    A_light_.push_back(light.getA());
    Z_light_.push_back(light.getZ());
    KE_light_.push_back(light.getKineticEnergy());
    n_light_.push_back(light.getNeutronMultiplicity());
/*
    ME0_light_.push_back(featLight.momentEOrdre0);
    ME1_light_.push_back(featLight.momentEOrdre1);
    ME2_light_.push_back(featLight.momentEOrdre2);
    ME3_light_.push_back(featLight.momentEOrdre3);
    ME4_light_.push_back(featLight.momentEOrdre4);
    
    I40_light_.push_back(featLight.integral40mm);
    Init_light_.push_back(featLight.initialdEdx);
    Rt_light_.push_back(featLight.trueR);
    R_light_.push_back(featLight.range);

    MZ1_light_.push_back(featLight.momentZOrdre1);
    MZ2_light_.push_back(featLight.momentZOrdre2);
    MZ3_light_.push_back(featLight.momentZOrdre3);
    MZ4_light_.push_back(featLight.momentZOrdre4);

    MZE_light_.push_back(featLight.momentZE);
    MZ2E_light_.push_back(featLight.momentZ2E);
    MZE2_light_.push_back(featLight.momentZE2);

    // MN2_light_.push_back(featLight.momentNOrdre2);
    // MN3_light_.push_back(featLight.momentNOrdre3);
    // MN4_light_.push_back(featLight.momentNOrdre4);

    pD1_light_.push_back(featLight.pD1);
    pD2_light_.push_back(featLight.pD2);
*/
    // betheTrace_light_.push_back(light.getBetheTrace());
    rawTrace_light_.push_back(light.getRawTrace());
    // denoisedTrace_light_.push_back(light.getDenoisedTrace());
    
    // Buffer heavy fragment
    A_heavy_.push_back(heavy.getA());
    Z_heavy_.push_back(heavy.getZ());
    KE_heavy_.push_back(heavy.getKineticEnergy());
    n_heavy_.push_back(heavy.getNeutronMultiplicity());
/*
    ME0_heavy_.push_back(featHeavy.momentEOrdre0);
    ME1_heavy_.push_back(featHeavy.momentEOrdre1);
    ME2_heavy_.push_back(featHeavy.momentEOrdre2);
    ME3_heavy_.push_back(featHeavy.momentEOrdre3);
    ME4_heavy_.push_back(featHeavy.momentEOrdre4);
    
    I40_heavy_.push_back(featHeavy.integral40mm);
    Init_heavy_.push_back(featHeavy.initialdEdx);
    Rt_heavy_.push_back(featHeavy.trueR);
    R_heavy_.push_back(featHeavy.range);

    MZ1_heavy_.push_back(featHeavy.momentZOrdre1);
    MZ2_heavy_.push_back(featHeavy.momentZOrdre2);
    MZ3_heavy_.push_back(featHeavy.momentZOrdre3);
    MZ4_heavy_.push_back(featHeavy.momentZOrdre4);

    MZE_heavy_.push_back(featHeavy.momentZE);
    MZ2E_heavy_.push_back(featHeavy.momentZ2E);
    MZE2_heavy_.push_back(featHeavy.momentZE2);

    // MN2_heavy_.push_back(featHeavy.momentNOrdre2);
    // MN3_heavy_.push_back(featHeavy.momentNOrdre3);
    // MN4_heavy_.push_back(featHeavy.momentNOrdre4);
    
    pD1_heavy_.push_back(featHeavy.pD1);
    pD2_heavy_.push_back(featHeavy.pD2);
*/
    // betheTrace_heavy_.push_back(heavy.getBetheTrace());
    rawTrace_heavy_.push_back(heavy.getRawTrace());
    // denoisedTrace_heavy_.push_back(heavy.getDenoisedTrace());
    
    bufferSize_ ++;
    
    // Flush si buffer plein
    if (bufferSize_ >= MAX_BUFFER_SIZE) {
        flushBuffers();
	std::cout << event_ID << " writen events.\r" ;
	std::cout.flush() ;
    }
    
}

void HDF5Writer::flushBuffers() {
  if (bufferSize_ == 0) return;
  
  try {
    std::cout << "[HDF5Writer] Flushing fragments... " ;
    
    // Écrire chaque colonne avec la fonction helper
    writeColumnHelper(file_, "/L/A", A_light_, H5::PredType::NATIVE_INT);
    writeColumnHelper(file_, "/L/Z", Z_light_, H5::PredType::NATIVE_INT);
    writeColumnHelper(file_, "/L/KE", KE_light_, H5::PredType::NATIVE_DOUBLE);
    writeColumnHelper(file_, "/L/n", n_light_, H5::PredType::NATIVE_INT);

    //writeColumnHelper(file_, "/Lf/I40", I40_light_, H5::PredType::NATIVE_DOUBLE);
    //writeColumnHelper(file_, "/Lf/Init", Init_light_, H5::PredType::NATIVE_DOUBLE);
    //writeColumnHelper(file_, "/Lf/Rt", Rt_light_, H5::PredType::NATIVE_DOUBLE);
    //writeColumnHelper(file_, "/Lf/R", R_light_, H5::PredType::NATIVE_DOUBLE);

  /*  writeColumnHelper(file_, "/Lf/ME0", ME0_light_, H5::PredType::NATIVE_DOUBLE);
    writeColumnHelper(file_, "/Lf/ME1", ME1_light_, H5::PredType::NATIVE_DOUBLE);
    writeColumnHelper(file_, "/Lf/ME2", ME2_light_, H5::PredType::NATIVE_DOUBLE);
    writeColumnHelper(file_, "/Lf/ME3", ME3_light_, H5::PredType::NATIVE_DOUBLE);
    writeColumnHelper(file_, "/Lf/ME4", ME4_light_, H5::PredType::NATIVE_DOUBLE);
    writeColumnHelper(file_, "/Lf/MZ1", MZ1_light_, H5::PredType::NATIVE_DOUBLE);
    writeColumnHelper(file_, "/Lf/MZ2", MZ2_light_, H5::PredType::NATIVE_DOUBLE);
    writeColumnHelper(file_, "/Lf/MZ3", MZ3_light_, H5::PredType::NATIVE_DOUBLE);
    writeColumnHelper(file_, "/Lf/MZ4", MZ4_light_, H5::PredType::NATIVE_DOUBLE);

    writeColumnHelper(file_, "/Lf/MZE", MZE_light_, H5::PredType::NATIVE_DOUBLE);
    writeColumnHelper(file_, "/Lf/MZ2E", MZ2E_light_, H5::PredType::NATIVE_DOUBLE);
    writeColumnHelper(file_, "/Lf/MZE2", MZE2_light_, H5::PredType::NATIVE_DOUBLE);
    // writeColumnHelper(file_, "/Lf/MN2", MN2_light_, H5::PredType::NATIVE_DOUBLE);
    // writeColumnHelper(file_, "/Lf/MN3", MN3_light_, H5::PredType::NATIVE_DOUBLE);
    // writeColumnHelper(file_, "/Lf/MN4", MN4_light_, H5::PredType::NATIVE_DOUBLE);

    writeColumnHelper(file_, "/Lf/pD1", pD1_light_, H5::PredType::NATIVE_DOUBLE);
    writeColumnHelper(file_, "/Lf/pD2", pD2_light_, H5::PredType::NATIVE_DOUBLE);
    */
    // writeVarLenColumnHelper(file_, "/L/bethe_trace", betheTrace_light_);
    writeVarLenColumnHelper(file_, "/L/raw_trace", rawTrace_light_);
    // writeVarLenColumnHelper(file_, "/L/denoised_trace", denoisedTrace_light_);
    
    writeColumnHelper(file_, "/H/A", A_heavy_, H5::PredType::NATIVE_INT);
    writeColumnHelper(file_, "/H/Z", Z_heavy_, H5::PredType::NATIVE_INT);
    writeColumnHelper(file_, "/H/KE", KE_heavy_, H5::PredType::NATIVE_DOUBLE);
    writeColumnHelper(file_, "/H/n", n_heavy_, H5::PredType::NATIVE_INT);
/*
    writeColumnHelper(file_, "/Hf/I40", I40_heavy_, H5::PredType::NATIVE_DOUBLE);
    writeColumnHelper(file_, "/Hf/Init", Init_heavy_, H5::PredType::NATIVE_DOUBLE);
    writeColumnHelper(file_, "/Hf/Rt", Rt_heavy_, H5::PredType::NATIVE_DOUBLE);
    writeColumnHelper(file_, "/Hf/R", R_heavy_, H5::PredType::NATIVE_DOUBLE);

    writeColumnHelper(file_, "/Hf/ME0", ME0_heavy_, H5::PredType::NATIVE_DOUBLE);
    writeColumnHelper(file_, "/Hf/ME1", ME1_heavy_, H5::PredType::NATIVE_DOUBLE);
    writeColumnHelper(file_, "/Hf/ME2", ME2_heavy_, H5::PredType::NATIVE_DOUBLE);
    writeColumnHelper(file_, "/Hf/ME3", ME3_heavy_, H5::PredType::NATIVE_DOUBLE);
    writeColumnHelper(file_, "/Hf/ME4", ME4_heavy_, H5::PredType::NATIVE_DOUBLE);
    writeColumnHelper(file_, "/Hf/MZ1", MZ1_heavy_, H5::PredType::NATIVE_DOUBLE);
    writeColumnHelper(file_, "/Hf/MZ2", MZ2_heavy_, H5::PredType::NATIVE_DOUBLE);
    writeColumnHelper(file_, "/Hf/MZ3", MZ3_heavy_, H5::PredType::NATIVE_DOUBLE);
    writeColumnHelper(file_, "/Hf/MZ4", MZ4_heavy_, H5::PredType::NATIVE_DOUBLE);

    writeColumnHelper(file_, "/Hf/MZE", MZE_heavy_, H5::PredType::NATIVE_DOUBLE);
    writeColumnHelper(file_, "/Hf/MZ2E", MZ2E_heavy_, H5::PredType::NATIVE_DOUBLE);
    writeColumnHelper(file_, "/Hf/MZE2", MZE2_heavy_, H5::PredType::NATIVE_DOUBLE);

    // writeColumnHelper(file_, "/Hf/MN2", MN2_heavy_, H5::PredType::NATIVE_DOUBLE);
    // writeColumnHelper(file_, "/Hf/MN3", MN3_heavy_, H5::PredType::NATIVE_DOUBLE);
    // writeColumnHelper(file_, "/Hf/MN4", MN4_heavy_, H5::PredType::NATIVE_DOUBLE);

    writeColumnHelper(file_, "/Hf/pD1", pD1_heavy_, H5::PredType::NATIVE_DOUBLE);
    writeColumnHelper(file_, "/Hf/pD2", pD2_heavy_, H5::PredType::NATIVE_DOUBLE);

  */  
    // writeVarLenColumnHelper(file_, "/H/bethe_trace", betheTrace_heavy_);
    writeVarLenColumnHelper(file_, "/H/raw_trace", rawTrace_heavy_);
    // writeVarLenColumnHelper(file_, "/H/denoised_trace", denoisedTrace_heavy_);
    
    // Vider les buffers
    A_light_.clear();
    Z_light_.clear();
    KE_light_.clear();
    n_light_.clear();

    /*I40_light_.clear();
    Init_light_.clear();
    Rt_light_.clear();
    R_light_.clear();

    ME0_light_.clear() ;
    ME1_light_.clear() ;
    ME2_light_.clear() ;
    ME3_light_.clear() ;
    ME4_light_.clear() ;

    MZ1_light_.clear() ;
    MZ2_light_.clear() ;
    MZ3_light_.clear() ;
    MZ4_light_.clear() ;

    MZE_light_.clear() ;
    MZ2E_light_.clear() ;
    MZE2_light_.clear() ;

    // MN2_light_.clear() ;
    // MN3_light_.clear() ;
    // MN4_light_.clear() ;

    pD1_light_.clear() ;
    pD2_light_.clear() ;
*/
    // betheTrace_light_.clear() ;
    rawTrace_light_.clear() ;
    // denoisedTrace_light_.clear() ;
    
    A_heavy_.clear();
    Z_heavy_.clear();
    KE_heavy_.clear();
    n_heavy_.clear();
/*
    I40_heavy_.clear();
    Init_heavy_.clear();
    Rt_heavy_.clear();
    R_heavy_.clear();

    ME0_heavy_.clear() ;
    ME1_heavy_.clear() ;
    ME2_heavy_.clear() ;
    ME3_heavy_.clear() ;
    ME4_heavy_.clear() ;

    MZ1_heavy_.clear() ;
    MZ2_heavy_.clear() ;
    MZ3_heavy_.clear() ;
    MZ4_heavy_.clear() ;

    MZE_heavy_.clear() ;
    MZ2E_heavy_.clear() ;
    MZE2_heavy_.clear() ;

    // MN2_heavy_.clear() ;
    // MN3_heavy_.clear() ;
    // MN4_heavy_.clear() ;

    pD1_heavy_.clear() ;
    pD2_heavy_.clear() ;

  */
    // betheTrace_heavy_.clear() ;
    rawTrace_heavy_.clear() ;
    // denoisedTrace_heavy_.clear() ;
    
    bufferSize_ = 0;
    
  } catch (H5::Exception& e) {
    throw std::runtime_error("HDF5 flush failed: " + std::string(e.getCDetailMsg()));
  }
}

void HDF5Writer::finalize() {
    std::cout << "[HDF5Writer] Finalizing..." << std::endl;

    // Désactiver les erreurs HDF5 sur stderr
    H5::Exception::dontPrint();

    flushBuffers();
    
    // Métadonnées finales
    size_t totalEvents = 0;
    
    try {
        // Essayer d'ouvrir un dataset (n'importe lequel)
        H5::DataSet dataset = file_->openDataSet("/L/A"); 
        H5::DataSpace dspace = dataset.getSpace();
        hsize_t dims[1];
        dspace.getSimpleExtentDims(dims);
        totalEvents = dims[0];
        dataset.close();
    } catch (H5::Exception&) {
        // Pas de données écrites
        totalEvents = 0;
    }
    
    addMetadata("total_events", static_cast<double>(totalEvents));

    if (denoising_trace_count_ > 0) {
        try {
            H5::Group denoising_group = file_->openGroup("/denoising");
            H5::DataSpace attr_space(H5S_SCALAR);
            
            {
                H5::Attribute attr = denoising_group.createAttribute(
                    "n_traces", H5::PredType::NATIVE_UINT64, attr_space
                );
                uint64_t n = denoising_trace_count_;
                attr.write(H5::PredType::NATIVE_UINT64, &n);
            }
            
            {
                H5::Attribute attr = denoising_group.createAttribute(
                    "max_length", H5::PredType::NATIVE_UINT64, attr_space
                );
                uint64_t m = max_trace_length_seen_;
                attr.write(H5::PredType::NATIVE_UINT64, &m);
            }
            
            denoising_group.close();
            
            std::cout << "✓ Denoising dataset: " << denoising_trace_count_ 
                      << " traces written" << std::endl;
        } catch (H5::Exception&) {
	  // Groupe n'existe pas (pas de denoising data)
        }
    }
    
    // Fermer fichier
    if (file_) {
        file_->close();
        delete file_;
        file_ = nullptr;
    }
    
    std::cout << "[HDF5Writer] Closed. Total: " << totalEvents << " events" << std::endl;
}

void HDF5Writer::addMetadata(const std::string& key, double value) {
    
    try {
        H5::Group metadataGroup = file_->openGroup("/metadata");
        H5::DataSpace attr_dataspace(H5S_SCALAR);
        H5::Attribute attribute = metadataGroup.createAttribute(
            key, H5::PredType::NATIVE_DOUBLE, attr_dataspace
        );
        attribute.write(H5::PredType::NATIVE_DOUBLE, &value);
    } catch (H5::Exception& e) {
        std::cerr << "[HDF5Writer] Warning: metadata '" << key << "' failed" << std::endl;
    }
}

void HDF5Writer::addMetadata(const std::string& key, const std::string& value) {
    
    try {
        H5::Group metadataGroup = file_->openGroup("/metadata");
        H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
        H5::DataSpace attr_dataspace(H5S_SCALAR);
        H5::Attribute attribute = metadataGroup.createAttribute(
            key, str_type, attr_dataspace
        );
        attribute.write(str_type, value);
    } catch (H5::Exception& e) {
        std::cerr << "[HDF5Writer] Warning: metadata '" << key << "' failed" << std::endl;
    }
}

void HDF5Writer::writeDenoisingPair(const FragmentPair& pair) {
  
  const auto& light = pair.getLightFragment();
  const auto& heavy = pair.getHeavyFragment();
    
    // Lambda pour écrire une trace
    auto write_single_trace = [&](const Fragment& frag) {
        const auto& bethe = frag.getBetheTrace();
        const auto& raw = frag.getRawTrace();
        
        size_t length = raw.size();
        
        if (length > max_trace_length_seen_) {
            max_trace_length_seen_ = length;
        }
        
        // Group name simple: trace_0, trace_1, trace_2...
        std::string group_name = "/denoising/trace_" + std::to_string(denoising_trace_count_);
        H5::Group trace_group = file_->createGroup(group_name);
        
        hsize_t dims[1] = {length};
        H5::DataSpace dataspace(1, dims);
        
        // Clean trace
        {
            std::vector<float> clean_data(length);
            for (size_t i = 0; i < length; ++i) {
                clean_data[i] = static_cast<float>(bethe[i]);
            }
            H5::DataSet dataset = trace_group.createDataSet(
                "clean", H5::PredType::NATIVE_FLOAT, dataspace
            );
            dataset.write(clean_data.data(), H5::PredType::NATIVE_FLOAT);
        }
        
        // Noisy trace
        {
            std::vector<float> noisy_data(length);
            for (size_t i = 0; i < length; ++i) {
                noisy_data[i] = static_cast<float>(raw[i]);
            }
            H5::DataSet dataset = trace_group.createDataSet(
                "noisy", H5::PredType::NATIVE_FLOAT, dataspace
            );
            dataset.write(noisy_data.data(), H5::PredType::NATIVE_FLOAT);
        }
        
        // Length as attribute
        H5::DataSpace attr_space(H5S_SCALAR);
        H5::Attribute attr = trace_group.createAttribute(
            "length", H5::PredType::NATIVE_UINT64, attr_space
        );
        uint64_t l = length;
        attr.write(H5::PredType::NATIVE_UINT64, &l);
        
        trace_group.close();
        denoising_trace_count_++;
    };
    
    // Créer /denoising si première fois
    if (denoising_trace_count_ == 0) {
        try {
            file_->createGroup("/denoising");
        } catch (H5::FileIException&) {
            // Existe déjà
        }
    }
    
    // Écrire les deux traces (light puis heavy)
    write_single_trace(light);
    write_single_trace(heavy);
}
