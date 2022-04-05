cmake_minimum_required(VERSION 3.21)
project(T3)

set(CMAKE_CXX_STANDARD 11)

include_directories("/usr/local/include/eigen3/")
include_directories("/usr/local/include/sophus")

find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )

add_executable(T3 E2Rt.cpp )
target_link_libraries( T3 ${Sophus_LIBRARIES} )