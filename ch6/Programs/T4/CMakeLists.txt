cmake_minimum_required(VERSION 3.21)
project(T4)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE "Release")
#add_definitions("-DENABLE_SSE")
#set(CMAKE_CXX_FLAGS "-std=c++11 ${SSE_FLAGS} -g -O3 -march=native")

find_package(OpenCV 3 REQUIRED)

include_directories(
        "/usr/local/include/eigen3"
        ${OpenCV_INCLUDE_DIRS}
)

add_executable(DisCalByLK DisCalByLK.cpp DisCalByLK.cpp)
#add_executable(optical_flow optical_flow_1.cpp )
target_link_libraries(DisCalByLK ${OpenCV_LIBS} tbb)
