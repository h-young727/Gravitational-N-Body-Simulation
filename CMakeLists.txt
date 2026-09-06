cmake_minimum_required(VERSION 3.16)

project(NBodySimulation LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP REQUIRED)

add_library(NBodyCore STATIC
    src/InitialConditions.cpp
    src/Simulation.cpp
)

target_include_directories(NBodyCore
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(NBodyCore
    PUBLIC
        OpenMP::OpenMP_CXX
)

add_executable(nbody_main
    Main.cpp
)

target_link_libraries(nbody_main
    PRIVATE
        NBodyCore
)

add_executable(nbody_bench_cpu
    bench/Benchmark.cpp
)

target_link_libraries(nbody_bench_cpu
    PRIVATE
        NBodyCore
)

include(CheckLanguage)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    if (NOT CMAKE_CUDA_ARCHITECTURES)
        if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
            set(CMAKE_CUDA_ARCHITECTURES native)  # auto-detects the building machine's GPU
        else()
            set(CMAKE_CUDA_ARCHITECTURES 61)  # CMake < 3.24 has no "native"; override with -DCMAKE_CUDA_ARCHITECTURES=<yours> if not Pascal
            message(STATUS "CMake < 3.24: defaulting CMAKE_CUDA_ARCHITECTURES to 61 (Pascal); override with -DCMAKE_CUDA_ARCHITECTURES=<yours> if needed")
        endif()
    endif()

    add_executable(nbody_bench_gpu
        cuda/Benchmark.cu
        cuda/Simulation.cu
    )

    target_include_directories(nbody_bench_gpu
        PRIVATE
            cuda
    )

    target_link_libraries(nbody_bench_gpu
        PRIVATE
            NBodyCore
    )
else()
    message(STATUS "No CUDA compiler found - skipping nbody_bench_gpu target")
endif()
