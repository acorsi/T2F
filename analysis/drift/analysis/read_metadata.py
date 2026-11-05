"""
Utilitaire : lecture des métadonnées depuis un fichier H5 T2F.
Fournit des valeurs par défaut si un attribut est absent.
Compatible avec les anciens fichiers (D_T, D_L, v_e) et les nouveaux (avec induction).
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
        "Fano":         float(attr("Fano",          0.2)),
        # diffusion : anciens fichiers D_T / D_L (un seul jeu), nouveaux fichiers kg / ga
        "DT_kg":        float(attr("DT_kg",         attr("D_T", 0.06))),
        "DT_ga":        float(attr("DT_ga",         attr("D_T", 0.06))),
        "DL_kg":        float(attr("DL_kg",         attr("D_L", 0.035))),
        "DL_ga":        float(attr("DL_ga",         attr("D_L", 0.02))),
        "ve_kg":        float(attr("ve_kg",         attr("v_e", 0.0055))),
        "ve_ga":        float(attr("ve_ga",         0.002)),
        "delta_t":      float(attr("delta_t",       10.0)),
        # seuils de lecture : anciens fichiers seuil_qt (>= 0), nouveaux seuil_qt_pos / seuil_qt_neg
        "seuil_qt_pos": float(attr("seuil_qt_pos",  attr("seuil_qt", 200.0))),
        "seuil_qt_neg": float(attr("seuil_qt_neg",  -1e300)),
        "n_bins":       int  (attr("n_bins",         0)),
        "reach_pitch":  float(attr("reach_pitch",    0.0)),
        "induction":    bool ("seuil_qt_pos" in getattr(meta, "attrs", {})),
        "vertex_r_max": float(attr("vertex_r_max",  1.0)),
        "n_events":     int  (attr("n_events",       0)),
        "input_file":   str  (attr("input_file",    "unknown")),
        "created":      str  (attr("created",       "unknown")),
    }
