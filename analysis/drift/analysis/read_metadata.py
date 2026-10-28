"""
Utilitaire : lecture des métadonnées depuis un fichier H5 T2F.
Fournit des valeurs par défaut si un attribut est absent.
"""

def read_metadata(f):
    """
    Lit les métadonnées depuis le groupe /metadata du fichier h5py `f`.
    Retourne un dict avec toutes les valeurs utiles.
    """
    meta = f.get("/metadata", {})

    def attr(name, default):
        try:
            return meta.attrs[name]
        except Exception:
            return default

    return {
        "dkg":          float(attr("dkg",          20.0)),
        "dga":          float(attr("dga",           1.0)),
        "pad_size":     float(attr("pad_size",       0.3)),
        "inter_pad":    float(attr("inter_pad",     0.015)),
        "r_active":     float(attr("r_active",       10)),
        "n_pads":       int  (attr("n_pads",        0)),
        "step_size_in": float(attr("step_size_in",  0.004)),
        "step_size_eff":float(attr("step_size_eff", 0.2)),
        "n_merge":      int  (attr("n_merge",        1)),
        "W":            float(attr("W",             26e-6)),
        "D_T":          float(attr("D_T",           0.03)),
        "delta_t":      float(attr("delta_t",       10.0)),
        "D_L":          float(attr("D_L",           0.015)),
        "vertex_r_max": float(attr("vertex_r_max",  1.0)),
        "n_events":     int  (attr("n_events",       0)),
        "input_file":   str  (attr("input_file",    "unknown")),
        "created":      str  (attr("created",       "unknown")),
    }
