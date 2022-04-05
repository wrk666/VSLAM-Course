cmake_minimum_required(VERSION 3.21)
project(T2)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#头文件
include_directories("/usr/local/include/eigen3")

add_executable(fisheye fisheye.cpp)
target_link_libraries(fisheye ${OpenCV_LIBS})