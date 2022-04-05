cmake_minimum_required(VERSION 3.21)
project(T4)

set(CMAKE_CXX_STANDARD 11)

include_directories( "/usr/local/include/eigen3" )  #添加头文件

find_package(Pangolin REQUIRED)
find_package(OpenCV 3 REQUIRED)

add_executable(disparity disparity.cpp)
target_link_libraries(disparity ${OpenCV_LIBS} ${Pangolin_LIBRARIES})