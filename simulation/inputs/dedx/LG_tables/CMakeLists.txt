cmake_minimum_required(VERSION 3.18)
project(LG_tables VERSION 1.0 LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0")


# =============================================================================
# CONFIGURATION DE BASE
# =============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# =============================================================================
# DÉPENDANCES
# =============================================================================

# Variables d'environnement
set(CONDA_ENV "$ENV{CONDA_PREFIX}")
set(LIBTORCH_PATH "$ENV{HOME}/dev/libtorch")

# HDF5 C++
find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS})

# Force l'utilisation de LibTorch depuis conda
if(DEFINED ENV{CONDA_PREFIX})
    message(STATUS "Using Conda environment: $ENV{CONDA_PREFIX}")
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}/lib/python3.11/site-packages/torch")
endif()

find_package(Torch REQUIRED)
message(STATUS "Torch found: ${TORCH_INCLUDE_DIRS}")

# OpenMP (depuis conda)
if(APPLE)
    if(CONDA_ENV)
        message(STATUS "Using conda OpenMP from: ${CONDA_ENV}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
        include_directories("${CONDA_ENV}/include")
#        link_directories("${CONDA_ENV}/lib")
        set(OPENMP_LIBRARIES "${CONDA_ENV}/lib/libomp.dylib")
    else()
        message(WARNING "CONDA_PREFIX not set. OpenMP may not work.")
        set(OPENMP_LIBRARIES "")
    endif()
else()
    find_package(OpenMP REQUIRED)
    set(OPENMP_LIBRARIES OpenMP::OpenMP_CXX)
endif()

# GSL (depuis conda)
if(CONDA_ENV)
    find_library(GSL_LIBRARY gsl PATHS "${CONDA_ENV}/lib" NO_DEFAULT_PATH)
    find_library(GSLCBLAS_LIBRARY gslcblas PATHS "${CONDA_ENV}/lib" NO_DEFAULT_PATH)
    set(GSL_LIBRARIES ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})
    include_directories("${CONDA_ENV}/include")
else()
    find_package(GSL REQUIRED)
    set(GSL_LIBRARIES GSL::gsl GSL::gslcblas)
endif()

# =============================================================================
# CHEMINS DES DONNÉES
# =============================================================================

set(DATA_DIR "${PROJECT_SOURCE_DIR}/data")
add_definitions(-DDATA_DIR="${DATA_DIR}")

# =============================================================================
# SOURCES
# =============================================================================

include_directories(${PROJECT_SOURCE_DIR}/inc)


set(PHYSICS_SOURCES
    src/Particle.cpp
    src/Material.cpp
    src/DeDx_LG.cpp
)

set(ALL_SOURCES
    ${PHYSICS_SOURCES}
 )

# =============================================================================
# EXÉCUTABLE
# =============================================================================

add_executable(LG_tables
    src/main.cpp
    ${ALL_SOURCES}
)

# Linking
target_link_libraries(LG_tables
    ${TORCH_LIBRARIES}
    ${HDF5_LIBRARIES}
    ${OPENMP_LIBRARIES}
    ${GSL_LIBRARIES}
)

# Fix RPATH spécifique pour LG_tables
if(APPLE AND CONDA_ENV)
    set_target_properties(LG_tables PROPERTIES
        INSTALL_RPATH "${CONDA_ENV}/lib;${LIBTORCH_PATH}/lib"
        INSTALL_RPATH_USE_LINK_PATH FALSE  # ← Clé ici
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# =============================================================================
# INSTALLATION
# =============================================================================

install(TARGETS LG_tables
    RUNTIME DESTINATION bin
)

# =============================================================================
# MESSAGES DE CONFIGURATION
# =============================================================================

message(STATUS "===== Configuration Summary =====")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Data directory: ${DATA_DIR}")
message(STATUS "HDF5 Include: ${HDF5_INCLUDE_DIRS}")
message(STATUS "LibTorch Path: ${LIBTORCH_PATH}")
if(APPLE AND CONDA_ENV)
    message(STATUS "OpenMP (conda): ${OPENMP_LIBRARIES}")
endif()
message(STATUS "=================================")