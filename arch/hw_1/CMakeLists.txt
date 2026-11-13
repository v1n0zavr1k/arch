cmake_minimum_required(VERSION 3.20)
project(simd_computing VERSION 0.1.0 LANGUAGES CXX)

option(SIMD_ENABLE_OPENMP "Enable OpenMP when available" ON)
option(SIMD_BUILD_TESTS "Build unit tests" ON)
option(SIMD_BUILD_BENCHMARKS "Build benchmark executable" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(xsimd REQUIRED)

if(SIMD_ENABLE_OPENMP)
    find_package(OpenMP)
endif()

add_library(simd_computing INTERFACE)
target_include_directories(simd_computing INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(simd_computing INTERFACE xsimd)

if(OpenMP_CXX_FOUND)
    target_link_libraries(simd_computing INTERFACE OpenMP::OpenMP_CXX)
    target_compile_definitions(simd_computing INTERFACE SIMD_HAS_OPENMP=1)
else()
    target_compile_definitions(simd_computing INTERFACE SIMD_HAS_OPENMP=0)
endif()

add_executable(simd_demo src/main.cpp)
target_link_libraries(simd_demo PRIVATE simd_computing)

if(SIMD_BUILD_BENCHMARKS)
    add_executable(simd_bench src/benchmark.cpp)
    target_link_libraries(simd_bench PRIVATE simd_computing)
endif()

if(SIMD_BUILD_TESTS)
    enable_testing()
    find_package(GTest CONFIG REQUIRED)
    add_executable(simd_tests tests/test_scalar.cpp)
    target_link_libraries(simd_tests PRIVATE simd_computing GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(simd_tests)
endif()
