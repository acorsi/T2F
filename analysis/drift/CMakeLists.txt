cmake_minimum_required(VERSION 3.20)

# --- Compilateur système Apple (doit être avant project()) ---
# Le compilateur conda (arm64-apple-darwin20) est incompatible avec macOS 25+
set(CMAKE_C_COMPILER   "/usr/bin/cc"  CACHE PATH "" FORCE)
set(CMAKE_CXX_COMPILER "/usr/bin/c++" CACHE PATH "" FORCE)

project(T2F_Processor CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- HDF5 via conda ---
set(CONDA_PREFIX "$ENV{CONDA_PREFIX}")
set(HDF5_ROOT "${CONDA_PREFIX}")
find_package(HDF5 REQUIRED COMPONENTS C)

# --- OpenMP : clang Apple nécessite libomp externe ---
set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
set(OpenMP_CXX_LIB_NAMES "omp")
set(OpenMP_omp_LIBRARY "${CONDA_PREFIX}/lib/libomp.dylib")
find_package(OpenMP REQUIRED)

# --- Includes ---
include_directories(include ${HDF5_INCLUDE_DIRS} ${CONDA_PREFIX}/include)

# --- Sources ---
set(SOURCES
    src/main.cpp
    src/read_h5.cpp
    src/process.cpp
    src/pad_geometry.cpp
    src/write_h5.cpp
)

add_executable(t2f_processor ${SOURCES})

target_link_libraries(t2f_processor
    PRIVATE
    ${HDF5_LIBRARIES}
    OpenMP::OpenMP_CXX
)

target_compile_options(t2f_processor PRIVATE -O2 -Wall)

target_link_options(t2f_processor PRIVATE
    -L${CONDA_PREFIX}/lib
    -Wl,-rpath,${CONDA_PREFIX}/lib
)

# --- Exécutable de test ---
add_executable(test_process
    test/test_process.cpp
    src/process.cpp
    src/pad_geometry.cpp
)

target_link_libraries(test_process
    PRIVATE
    OpenMP::OpenMP_CXX
)

target_compile_options(test_process PRIVATE -O2 -Wall)

target_link_options(test_process PRIVATE
    -L${CONDA_PREFIX}/lib
    -Wl,-rpath,${CONDA_PREFIX}/lib
)

# --- Étude de convergence du pas d'échantillonnage ---
add_executable(test_sampling
    test/test_sampling.cpp
    src/process.cpp
    src/pad_geometry.cpp
)
target_link_libraries(test_sampling PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(test_sampling PRIVATE -O2 -Wall)
target_link_options(test_sampling PRIVATE
    -L${CONDA_PREFIX}/lib
    -Wl,-rpath,${CONDA_PREFIX}/lib
)