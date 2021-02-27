cmake_minimum_required(VERSION 3.15)

# Setting project name 
project(BenchmarkingOperations CXX)
# Including cmake settings

if(DEFINED ENV{OpenBLAS_HOME})
    set(OpenBLAS_HOME $ENV{OpenBLAS_HOME})
    message(STATUS "OpenBLAS_HOME = ${OpenBLAS_HOME}")
    add_compile_definitions(OpenBLAS_HOME=${OpenBLAS_HOME})
else()
    message(FATAL_ERROR "Please set OpenBLAS_HOME with openblas home directory")
endif(DEFINED ENV{OpenBLAS_HOME})

if(DEFINED ENV{BLIS_HOME})
    set(BLIS_HOME $ENV{BLIS_HOME})
    message(STATUS "BLIS_HOME = ${BLIS_HOME}")
else()
    message(FATAL_ERROR "Please set BLIS_HOME with blis home directory")
endif(DEFINED ENV{BLIS_HOME})

if(DEFINED ENV{MKL_HOME})
    set(MKL_HOME $ENV{MKL_HOME})
    message(STATUS "MKL_HOME = ${MKL_HOME}")
else()
    message(FATAL_ERROR "Please set MKL_HOME with mkl home directory")
endif(DEFINED ENV{MKL_HOME})


include(cmake/StandardProjectSettings.cmake)
include(cmake/OpenMP.cmake)
include(cmake/Matplot.cmake)
include(cmake/Boost.cmake)
include(cmake/OpenBlas.cmake)
include(cmake/Blis.cmake)
include(cmake/Eigen.cmake)
include(cmake/MKL.cmake)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories("${PROJECT_SOURCE_DIR}/include")

# Linking Library
add_library(project_options INTERFACE)

add_library(project_warnings INTERFACE)

include(cmake/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

include(cmake/CompilerArchSettings.cmake)
set_project_arch_settings(project_options)

# enable sanitizer options supported by compiler
include(cmake/Sanitizers.cmake)
enable_sanitizers(project_options)

include(cmake/Conan.cmake)
run_conan()

include(cmake/StaticAnalyzers.cmake)

option(BUILD_SHARED_LIBS "Enable Compilation of shared libraries" OFF)

option(ENABLE_PCH "Enable Precompiled Header" OFF)
if (ENABLE_PCH)
    target_precompile_headers(project_options INTERFACE <iostream> <vector> <map> <utility>)
endif()

option(ENABLE_TESTING "Enable Test Builds" ON)
if (ENABLE_TESTING)
    enable_testing()
    message(
        "Building Tests."
    )
    add_subdirectory(test)
endif()

add_subdirectory(src)
