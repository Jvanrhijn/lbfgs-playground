cmake_minimum_required(VERSION 3.0.2)

enable_language(Fortran)

project(lbfgs_playground)

set(TARGET lbfgs_pg)
include_directories(lib/numeric_kinds)

add_subdirectory(lib)
add_subdirectory(src)
