cmake_minimum_required(VERSION 2.8)
project(ch3_T7)

set(CMAKE_CXX_STANDARD 11)

find_package( Sophus REQUIRED )
include_directories( "/usr/local/include/eigen3" )  #添加头文件
include_directories( "/usr/local/include/sophus" )

#add_executable(ch3_T7 main.cpp)
add_executable(DrawTrajectory draw_trajectory.cpp)