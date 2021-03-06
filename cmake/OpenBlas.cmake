SET(Open_BLAS_INCLUDE_SEARCH_PATHS
    ${OpenBLAS_HOME}
    ${OpenBLAS_HOME}/include
)

SET(Open_BLAS_LIB_SEARCH_PATHS
    ${OpenBLAS_HOME}
    ${OpenBLAS_HOME}/lib
)

# FIND_PATH(OpenBLAS_INCLUDE_DIR NAMES 
#     cblas.h PATHS ${Open_BLAS_INCLUDE_SEARCH_PATHS}
# )

FIND_LIBRARY(OpenBLAS_LIB NAMES 
    openblas PATHS ${Open_BLAS_LIB_SEARCH_PATHS}
)

# include_directories(${OpenBLAS_INCLUDE_DIR})
add_compile_definitions(OpenBLAS_LIB="${OpenBLAS_LIB}")

message("OpenBLAS_INCLUDE_DIR: ${OpenBLAS_INCLUDE_DIR}, OpenBLAS_LIB: ${OpenBLAS_LIB}")