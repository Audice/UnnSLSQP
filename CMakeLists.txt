cmake_minimum_required(VERSION 3.14)

set (CMAKE_CXX_STANDARD 17)

project(test_my_lib)

add_subdirectory(slsqp_lib)

add_executable(main main.cpp)
target_link_libraries(main slsqp)