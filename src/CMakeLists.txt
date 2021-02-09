set(SRC_LIST_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src_list.cmake")

if ( EXISTS "${SRC_LIST_PATH}" )
  include("${SRC_LIST_PATH}")

  foreach(SRC_FULL_NAME ${SRC_LIST})
    
    string( REPLACE ".cpp" "" src_name ${SRC_FULL_NAME} )
    add_executable( "${src_name}" "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_FULL_NAME}")
    target_link_libraries( "${src_name}" PRIVATE 
                            project_options project_warnings 
                            ${OPENMP_LIB} ${MATPLOT_LIB} Eigen3::Eigen
                            ${BLIS_LIB} 
                            ${OpenBLAS_LIB}
                            ${MKL_LIB} 
    )
    
  endforeach(SRC_FULL_NAME ${SRC_LIST})

endif()