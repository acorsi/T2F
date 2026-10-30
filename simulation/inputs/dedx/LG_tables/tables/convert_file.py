#!/usr/bin/env python3
"""
Conversion des tables dE/dx format LG vers le format Geant4.
- Col 1 : KE/A  (déjà en MeV/A, recopié tel quel)
- Col 2 + Col 3 : dEdx_elec + dEdx_nucl -> sommés -> dEdx_total
- Fichiers source : ./dedx_ZZ.lg   (ZZ sans zéro de remplissage)
- Sortie : G4/dedx_ZXX.dat         (XX sur 2 chiffres)
"""

import os

INPUT_DIR = "./"
OUTPUT_DIR = "./LG"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for Z in range(20, 71):
    infile = os.path.join(INPUT_DIR, f"dedx_{Z}.lg")
    if not os.path.exists(infile):
        print(f"Z={Z:2d} : fichier introuvable ({infile})")
        continue

    data_rows = []

    with open(infile, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            cols = stripped.split()
            if len(cols) < 3:
                continue
            try:
                ke_per_a = float(cols[0])
                dedx_tot = float(cols[1]) + float(cols[2])
                data_rows.append((ke_per_a, dedx_tot))
            except ValueError:
                continue

    if not data_rows:
        print(f"Z={Z:2d} : aucune donnée lue dans {infile}")
        continue

    outname = f"dedx_Z{Z:02d}.dat"
    outfile = os.path.join(OUTPUT_DIR, outname)

    with open(outfile, "w") as f:
        f.write("# KE/A (MeV/A)   dedx (MeV/mg.cm**2)\n")
        for ke_per_a, dedx in data_rows:
            f.write(f"{ke_per_a:.6e}  {dedx:.6e}\n")

    print(f"Z={Z:2d} -> {outname}  ({len(data_rows)} points)")

print("\nTerminé.")
