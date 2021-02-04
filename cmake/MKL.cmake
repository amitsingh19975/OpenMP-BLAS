set(BLA_VENDOR Intel10_64lp)

find_package(BLAS REQUIRED)

if(BLAS_FOUND)
    message(STATUS "MKL FOUND")
    set(MKL_LIB BLAS::BLAS)
    include_directories()
else()
    message(FATAL_ERROR "MKL NOT FOUND")
endif(BLAS_FOUND)

