SET(MKL_INCLUDE_SEARCH_PATHS
    ${MKL_HOME}
    ${MKL_HOME}/include
)

link_directories(${MKL_HOME}/lib)
link_directories(/opt/intel/lib)


FIND_PATH(MKL_INCLUDE_DIR NAMES 
    mkl_cblas.h PATHS ${MKL_INCLUDE_SEARCH_PATHS}
)

SET(MKL_LIB 
    mkl_intel_ilp64
    mkl_intel_thread 
    mkl_core
    iomp5
    m
    dl
)

add_compile_definitions(MKL_ILP64)

include_directories(${MKL_INCLUDE_DIR})

message("MKL_INCLUDE_DIR: ${MKL_INCLUDE_DIR}, MKL_LIB: ${MKL_LIB}")