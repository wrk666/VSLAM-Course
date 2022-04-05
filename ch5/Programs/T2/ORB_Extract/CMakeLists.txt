cmake_minimum_required(VERSION 3.21)
project(ORB_Extract)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_BUILD_TYPE "Debug")
MESSAGE(STATUS "CMAKE_BUILD_TYPE IS ${CMAKE_BUILD_TYPE}")

find_package(OpenCV 3 REQUIRED)

#添加头文件
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        "/usr/local/include/eigen3/"
)

# 手写ORB特征
add_executable(computeORB computeORB.cpp)
#链接OpenCV库和tbb库
target_link_libraries(computeORB ${OpenCV_LIBS} tbb)

#add_executable(test  test.cpp)
#target_link_libraries(test tbb)


