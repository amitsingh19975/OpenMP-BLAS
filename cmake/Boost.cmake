find_package(Boost REQUIRED)

if(Boost_FOUND)
    message(STATUS "Found Boost")
    include_directories(${Boost_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Boost not found, please check")
endif(Boost_FOUND)
