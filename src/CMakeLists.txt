cmake_minimum_required(VERSION 3.0.2)

include_directories(modules)
set(CMAKE_Fortran_MODULE_DIRECTORY modules)
file(GLOB MODULE_FILES "modules/*.f90")

set(TARGET lbfgs_test)

add_executable(${TARGET} main.f90 ${MODULE_FILES})