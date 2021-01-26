set(project_name "benchmark")


add_executable(${project_name} main.cpp)

target_link_libraries(
  ${project_name} PRIVATE project_options project_warnings
    ${OPENMP_LIB} ${MATPLOT_LIB} ${OpenBLAS_LIB}
)