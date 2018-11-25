cmake_minimum_required(VERSION 2.8)

project(polynom_multiplication)
add_executable(${PROJECT_NAME} main.c)
add_library(polynom_mul_lib STATIC polynom_mul.c polynom_mul.h)
target_compile_options(polynom_mul_lib PRIVATE -m64 -mpclmul)
target_link_libraries(${PROJECT_NAME} polynom_mul_lib)
