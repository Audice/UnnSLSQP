cmake_minimum_required(VERSION 3.14)

project(slsqp_lib)

file (GLOB SRC src/*.cpp)

add_library(slsqp ${SRC})

target_include_directories(slsqp PUBLIC include)