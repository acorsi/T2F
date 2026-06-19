#ifndef HDF5_WRITE_CONFIG_H
#define HDF5_WRITE_CONFIG_H

/**
 * Configuration pour l'écriture HDF5.
 * Permet de contrôler quelles données sont écrites (pour gérer la taille des fichiers).
 */
struct HDF5WriteConfig {
    // Traces
    bool writeRawTrace = true;
    bool writeDenoisedTrace = false;  // Optionnel (peut faire de gros fichiers)
    
    // Données traitées
    bool writeFeatures = true;
    bool writeLatent = true;
    
    // Métadonnées
    bool writeMetadata = true;
    
    // Configuration par défaut : tout sauf denoisedTrace
    HDF5WriteConfig() = default;
    
    // Configuration minimaliste (juste features, pas de traces)
    static HDF5WriteConfig minimal() {
        HDF5WriteConfig config;
        config.writeRawTrace = false;
        config.writeDenoisedTrace = false;
        return config;
    }
    
    // Configuration complète (debug/validation)
    static HDF5WriteConfig full() {
        HDF5WriteConfig config;
        config.writeDenoisedTrace = true;
        return config;
    }
};

#endif // HDF5_WRITE_CONFIG_H
