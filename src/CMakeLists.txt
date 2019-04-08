cmake_minimum_required(VERSION 3.0.2)

set(TARGET lbfgs_test)

add_executable(${TARGET} main.f90 numeric_kinds.f90)
target_link_libraries(${TARGET} LBFGS numeric_kinds)
