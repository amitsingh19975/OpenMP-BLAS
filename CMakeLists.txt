cmake_minimum_required(VERSION 3.15)

# Setting project name 
project(BenchmarkingOperations CXX)
# Including cmake settings

if(NOT DEFINED OpenBLAS_HOME)
    set(OpenBLAS_HOME "/usr/local/opt/openblas")
endif(NOT DEFINED OpenBLAS_HOME)


include(cmake/StandardProjectSettings.cmake)
include(cmake/OpenMP.cmake)
include(cmake/Matplot.cmake)
include(cmake/Boost.cmake)
include(cmake/OpenBlas.cmake)

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

include(cmake/StaticAnalyzers.cmake)

option(BUILD_SHARED_LIBS "Enable Compilation of shared libraries" OFF)

option(ENABLE_PCH "Enable Precompiled Header" OFF)
if (ENABLE_PCH)
    target_precompile_headers(project_options INTERFACE <iostream> <vector> <map> <utility>)
endif()

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer ")

add_subdirectory(src)
