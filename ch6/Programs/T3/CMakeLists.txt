cmake_minimum_required(VERSION 3.21)
project(T3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")

find_package(OpenCV 3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Boost)

include_directories(
        "/usr/local/include/eigen3"
        ${Sophus_INCLUDE_DIRS}
        ${Pangolin_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
)

#add_executable(direct_method direct_method.cpp )
#target_link_libraries(direct_method  ${OpenCV_LIBS} ${Pangolin_LIBRARIES})

add_executable(direct_method_1 direct_method_1.cpp )
target_link_libraries(direct_method_1  ${OpenCV_LIBS} ${Pangolin_LIBRARIES} tbb)

add_executable(test test.cpp)
target_link_libraries(test ${Boost_LIBRARIES})