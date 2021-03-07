# automatically enable catch2 to generate ctest targets
if(CONAN_CATCH2_ROOT_DEBUG)
  include(${CONAN_CATCH2_ROOT_DEBUG}/lib/cmake/Catch2/Catch.cmake)
else()
  include(${CONAN_CATCH2_ROOT}/lib/cmake/Catch2/Catch.cmake)
endif()


add_library(catch_main STATIC catch_main.cpp)
target_link_libraries(catch_main PUBLIC CONAN_PKG::catch2)

if ( EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_list.cmake" )
  include("${CMAKE_CURRENT_SOURCE_DIR}/test_list.cmake")
  
  foreach(TEST_FILE ${TEST_LIST})
    
    string( REPLACE ".cpp" "" test_name ${TEST_FILE} )
    add_executable( "${test_name}" "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_FILE}")
    target_link_libraries( "${test_name}" PRIVATE project_warnings project_options
                                    catch_main
                                    ${OPENMP_LIB}
                                    ${BLIS_LIB}
                                    Boost::filesystem
                        )

    # automatically discover tests that are defined in catch based test files you
    # can modify the unittests. TEST_PREFIX to whatever you want, or use different
    # for different binaries
    catch_discover_tests(
      "${test_name}"
      TEST_PREFIX
      "unittests."
      EXTRA_ARGS
      -s
      --reporter=xml
      --out=tests.xml)

  endforeach(TEST_FILE ${TEST_LIST})

endif()
