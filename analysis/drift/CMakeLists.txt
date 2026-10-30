cmake_minimum_required(VERSION 3.16)

# --- Forcer le compilateur système macOS avant project() ---
# (doit être avant project() sinon ignoré)
if(APPLE)
    set(CMAKE_CXX_COMPILER /usr/bin/clang++ CACHE STRING "" FORCE)
    set(CMAKE_C_COMPILER   /usr/bin/clang   CACHE STRING "" FORCE)
    execute_process(COMMAND xcrun --show-sdk-path
                    OUTPUT_VARIABLE CMAKE_OSX_SYSROOT
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(CMAKE_OSX_SYSROOT ${CMAKE_OSX_SYSROOT} CACHE STRING "" FORCE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0")
endif()

project(T2F_drift CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- HDF5 ---
find_package(HDF5 REQUIRED COMPONENTS C)
include_directories(${HDF5_INCLUDE_DIRS})

# --- OpenMP (transparent sur macOS/Linux) ---
if(APPLE)
    execute_process(COMMAND brew --prefix libomp
                    OUTPUT_VARIABLE LIBOMP_PREFIX
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
    include_directories("${LIBOMP_PREFIX}/include")
endif()
find_package(OpenMP REQUIRED)

# --- Includes ---
include_directories(inc)

# --- Bibliothèque commune ---
add_library(t2f_lib
    src/h5reader.cpp
    src/h5writer.cpp
    src/processor.cpp
    src/pad_geometry.cpp
)
target_link_libraries(t2f_lib
    ${HDF5_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# --- Exécutables ---
add_executable(process src/main.cpp)
target_link_libraries(process t2f_lib)