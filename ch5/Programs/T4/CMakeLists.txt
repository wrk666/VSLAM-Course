cmake_minimum_required(VERSION 3.21)
project(T4)

set(CMAKE_CXX_STANDARD 11)

find_package(Sophus REQUIRED)

include_directories(
        "/usr/local/include/eigen3/"
        ${Sophus_INCLUDE_DIRS}
)


add_executable(T4 GN-BA.cpp)
