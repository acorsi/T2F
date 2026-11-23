cmake_minimum_required(VERSION 3.16)

# reco_profil : serveur de base de chapeaux (profile_basis_fast) pour la reconstruction des profils dE/dx.
# Les sources de la physique (induction, reponse, bruit) ne sont PAS copiees : elles sont prises dans le
# simulateur drift, pour que la reconstruction utilise exactement la meme physique que ./process.
# Dossier de drift : option DRIFT_DIR (defaut : ../drift, c'est-a-dire analysis/drift).
# Pas de HDF5.

if(APPLE)
    set(CMAKE_CXX_COMPILER /usr/bin/clang++ CACHE STRING "" FORCE)
    set(CMAKE_C_COMPILER   /usr/bin/clang   CACHE STRING "" FORCE)
    execute_process(COMMAND xcrun --show-sdk-path
                    OUTPUT_VARIABLE CMAKE_OSX_SYSROOT
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(CMAKE_OSX_SYSROOT ${CMAKE_OSX_SYSROOT} CACHE STRING "" FORCE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0")
endif()

project(reco_profil CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Type de build" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Sources de drift ---
set(DRIFT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../drift" CACHE PATH "Dossier du simulateur drift (contient src/ et inc/)")
if(NOT EXISTS "${DRIFT_DIR}/src/pad_geometry.cpp")
    message(FATAL_ERROR "DRIFT_DIR invalide : ${DRIFT_DIR} (src/pad_geometry.cpp introuvable). Relancer avec -DDRIFT_DIR=/chemin/vers/analysis/drift")
endif()
file(READ "${DRIFT_DIR}/inc/pad_geometry.hpp" PAD_GEOMETRY_HPP)
if(NOT PAD_GEOMETRY_HPP MATCHES "project_hats")
    message(FATAL_ERROR "${DRIFT_DIR}/inc/pad_geometry.hpp ne contient pas PadPlane::project_hats : version de drift trop ancienne.")
endif()

# --- OpenMP (meme logique que le simulateur) ---
if(APPLE)
    if(EXISTS "$ENV{CONDA_PREFIX}/lib/libomp.dylib")
        set(LIBOMP_PREFIX "$ENV{CONDA_PREFIX}")
    else()
        execute_process(COMMAND brew --prefix libomp
                        OUTPUT_VARIABLE LIBOMP_PREFIX
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
    if(NOT EXISTS "${LIBOMP_PREFIX}/lib/libomp.dylib")
        message(FATAL_ERROR "OpenMP introuvable : activer un environnement Conda contenant libomp.")
    endif()
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
    include_directories("${LIBOMP_PREFIX}/include")
endif()
find_package(OpenMP REQUIRED)

include_directories(${DRIFT_DIR}/inc)

add_library(reco_lib
    ${DRIFT_DIR}/src/pad_geometry.cpp
    ${DRIFT_DIR}/src/weighting_potential.cpp
    ${DRIFT_DIR}/src/psi_builder.cpp
    ${DRIFT_DIR}/src/psi_table.cpp
    ${DRIFT_DIR}/src/noise.cpp
)
target_link_libraries(reco_lib OpenMP::OpenMP_CXX)

add_executable(profile_basis_fast src/profile_basis_fast.cpp)
target_link_libraries(profile_basis_fast reco_lib)
