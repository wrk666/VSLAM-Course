cmake_minimum_required(VERSION 2.8)
project(ch3_T4)

set(CMAKE_CXX_STANDARD 11)

include_directories( "/usr/local/include/eigen3" )  #添加头文件
add_executable(ch3_T4 main.cpp)
