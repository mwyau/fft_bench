cmake_minimum_required(VERSION 3.25)
project(fft_bench LANGUAGES CXX)
find_package(Threads REQUIRED)

set(BENCH_BACKEND "ducc" CACHE STRING "ducc, fftw or admiral")
set(BENCH_ARCH "x86-64" CACHE STRING "Compiler architecture")
set(DUCC_SOURCE "" CACHE PATH "DUCC source checkout")
set(DUCC_CXX_STANDARD "17" CACHE STRING "DUCC C++ standard (17 or 23)")
set_property(CACHE DUCC_CXX_STANDARD PROPERTY STRINGS 17 23)
set(ADMIRAL_SOURCE "" CACHE PATH "Admiral source checkout")

add_executable(fft_native bench.cpp)
target_compile_features(fft_native PRIVATE cxx_std_17)
target_compile_options(fft_native PRIVATE -O3 -march=${BENCH_ARCH})
target_link_libraries(fft_native PRIVATE Threads::Threads)

if(BENCH_BACKEND STREQUAL "ducc")
    add_library(ducc_fft STATIC
        "${DUCC_SOURCE}/src/ducc0/fft/fft_inst1.cc"
        "${DUCC_SOURCE}/src/ducc0/fft/fft_inst2.cc"
        "${DUCC_SOURCE}/src/ducc0/infra/threading.cc"
        "${DUCC_SOURCE}/src/ducc0/infra/mav.cc"
        "${DUCC_SOURCE}/src/ducc0/infra/string_utils.cc")
    if(NOT DUCC_CXX_STANDARD MATCHES "^(17|23)$")
        message(FATAL_ERROR "DUCC_CXX_STANDARD must be 17 or 23")
    endif()
    set_target_properties(ducc_fft fft_native PROPERTIES
        CXX_STANDARD "${DUCC_CXX_STANDARD}"
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS YES)
    target_compile_definitions(fft_native PRIVATE BENCH_DUCC_CXX_STANDARD=${DUCC_CXX_STANDARD})
    target_include_directories(ducc_fft PUBLIC "${DUCC_SOURCE}/src")
    target_compile_options(ducc_fft PRIVATE -O3 -ffast-math -march=${BENCH_ARCH})
    target_link_libraries(ducc_fft PUBLIC Threads::Threads)
    target_compile_definitions(fft_native PRIVATE BENCH_DUCC)
    target_link_libraries(fft_native PRIVATE ducc_fft)
elseif(BENCH_BACKEND STREQUAL "fftw")
    target_compile_definitions(fft_native PRIVATE BENCH_FFTW)
    foreach(name fftw3_threads fftw3 fftw3f_threads fftw3f fftw3l_threads fftw3l)
        find_library(lib_${name} NAMES ${name} REQUIRED)
        target_link_libraries(fft_native PRIVATE ${lib_${name}})
    endforeach()
elseif(BENCH_BACKEND STREQUAL "admiral")
    set(ADM_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(ADM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(ADM_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(ADM_TARGET_ARCH "${BENCH_ARCH}" CACHE STRING "" FORCE)
    add_subdirectory("${ADMIRAL_SOURCE}" admiral)
    target_compile_features(fft_native PRIVATE cxx_std_20)
    target_compile_definitions(fft_native PRIVATE BENCH_ADMIRAL)
    target_link_libraries(fft_native PRIVATE admiral::admiral_static)
else()
    message(FATAL_ERROR "Unknown backend: ${BENCH_BACKEND}")
endif()
