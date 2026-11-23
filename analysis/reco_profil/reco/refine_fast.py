"""
refine_fast.py : lance refine_geom.py avec le blanchiment rapide (whiten_fast.py) et le jacobien par fragment (jac_fast.py),
sans modifier refine_geom.py ni fit_profile.py. Necessite le serveur profile_basis_fast (--exe ../build/profile_basis_fast).
Arguments : exactement ceux de refine_geom.py. JAC_FAST=0 dans l'environnement : sans le jacobien par fragment
(utilisable alors avec l'ancien profile_basis).
"""
import os
import sys

import fit_profile
import whiten_fast

whiten_fast.install(fit_profile)

import refine_geom

if os.environ.get("JAC_FAST", "1") != "0":
    import jac_fast
    jac_fast.install(refine_geom)

if __name__ == "__main__":
    sys.argv = ["refine_geom.py"] + sys.argv[1:]
    refine_geom.main()
