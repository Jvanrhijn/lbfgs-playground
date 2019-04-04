cmake_minimum_required(VERSION 3.0.2)

set(lbfgs_files lbfgs.f)

add_library(LBFGS STATIC ${lbfgs_files})