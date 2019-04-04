cmake_minimum_required(VERSION 3.0.2)

enable_language(Fortran)

project(lbfgs_playground)

set(TARGET lbfgs_pg)

add_subdirectory(lib)

add_executable(${TARGET} main.f90)
target_link_libraries(${TARGET} LBFGS)
