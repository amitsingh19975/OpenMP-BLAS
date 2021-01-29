# https://github.com/alandefreitas/matplotplusplus
find_package(Matplot++ REQUIRED)
if(Matplot++_FOUND)
    message(STATUS "Found Matplot++")
    set(MATPLOT_LIB Matplot++::matplot)
else()
    message(FATAL_ERROR "Please install Matplot++ from https://github.com/alandefreitas/matplotplusplus")
endif()
