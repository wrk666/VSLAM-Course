cmake_minimum_required(VERSION 3.21)
project(T2_2)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE "Debug")
#add_definitions("-DENABLE_SSE")
#set(CMAKE_CXX_FLAGS "-std=c++11 ${SSE_FLAGS} -g -O3 -march=native")

find_package(OpenCV 3 REQUIRED)

include_directories(
        "/usr/local/include/eigen3"
        ${OpenCV_INCLUDE_DIRS}
        )

add_executable(optical_flow optical_flow.cpp )
#add_executable(optical_flow optical_flow_1.cpp )
target_link_libraries(optical_flow ${OpenCV_LIBS} tbb)


