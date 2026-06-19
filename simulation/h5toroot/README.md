# h5toroot - HDF5 to ROOT Converter

Convertit automatiquement des fichiers HDF5 (Time Projection Chamber simulation) vers format ROOT pour exploration interactive.

## Compilation
```bash
make
```

**Prérequis** :
- HDF5 C++ (Homebrew ou conda)
- ROOT 6.x
- g++ avec support C++17

Si HDF5 est dans conda, modifier `Makefile` :
```makefile
HDF5_CFLAGS = -I$(CONDA_PREFIX)/include
HDF5_LIBS = -L$(CONDA_PREFIX)/lib -lhdf5_cpp -lhdf5
```

## Usage

### Conversion basique
```bash
./h5toroot input.h5 output.root
```

### Avec longueur max des traces
```bash
./h5toroot input.h5 output.root --maxlen 200
```

### Avec validation
```bash
./h5toroot input.h5 output.root --check
```

### Installation globale
```bash
make install  # Copie vers ~/bin/
```

## Fonctionnalités

✅ **Scan automatique** : Détecte la structure HDF5 (`/heavy/`, `/light/`)  
✅ **Gestion VLen** : Convertit traces à longueur variable en arrays fixes  
✅ **Truncate intelligent** : Warning si trace > MAXLEN  
✅ **Progress bar** : Affichage temps réel  
✅ **Validation** : Mode `--check` compare HDF5 vs ROOT  
✅ **Métadonnées** : Copie `/metadata` comme TNamed  

## Structure ROOT générée
```cpp
TTree "T2F"
├── heavy_Z        (Int_t)
├── heavy_A        (Int_t)
├── heavy_KE       (Float_t)
├── heavy_n        (Int_t)
├── heavy_raw_trace[150]  (Float_t, padding=0.0)
├── light_Z        (Int_t)
├── light_A        (Int_t)
├── light_KE       (Float_t)
├── light_n        (Int_t)
└── light_raw_trace[150]  (Float_t, padding=0.0)
```

**Note** : Traces < MAXLEN sont paddées avec `0.0f`

## Exploration ROOT
```cpp
root output.root
TBrowser b;  // Interface graphique

// Ou en ligne de commande
root -l output.root
T2F->Draw("heavy_Z:light_Z")
T2F->Draw("heavy_KE", "heavy_Z==44")
```

## Performances

- **300 MB HDF5** → **~15 secondes** (MacBook M1)
- **200k events** → **~10 secondes**
- Mémoire : ~2× taille fichier HDF5

## Troubleshooting

### Erreur linkage HDF5
```bash
# Vérifier installation
brew list hdf5
conda list hdf5

# Ajuster Makefile paths
```

### Erreur ROOT
```bash
# Vérifier ROOT
root-config --version
root-config --libs
```

### Traces tronquées
```
WARNING: Trace truncated at event 1234 (heavy_raw_trace len=175 > 150)
```
→ Augmenter `--maxlen 200`

## Limitations

- **MAXLEN fixe** : Toutes traces stockées avec même longueur (padding)
- **Mémoire** : Charge tout le HDF5 en RAM (OK jusqu'à ~1 GB)
- **Métadonnées** : Copie seulement datasets STRING

## Développement

**Structure code** :
- `scanGroup()` : Détection automatique structure HDF5
- `readVLenDataset()` : Lecture traces variables (`hvl_t`)
- `convertH5ToRoot()` : Boucle principale
- `validateConversion()` : Tests cohérence

**Ajout nouveau type** :
```cpp
// Dans scanGroup(), ajouter détection
if (dtype.getClass() == H5T_COMPOUND) {
    // Gérer structures
}
```

