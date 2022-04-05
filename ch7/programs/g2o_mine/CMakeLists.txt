cmake_minimum_required(VERSION 3.21)
project(g2o_mine)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3 -std=c++11")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(G2O REQUIRED)
find_package(OpenCV 3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)
find_package(CSparse REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        "/usr/include/eigen3/"
        ${CSPARSE_INCLUDE_DIR}
)

add_library(bal_common common.cpp)

set(G2O_LIBS g2o_core g2o_stuff g2o_csparse_extension cxsparse)

add_executable(g2oCurveFitting main.cpp)
target_link_libraries(g2oCurveFitting
                    g2o_core g2o_stuff
                    ${OpenCV_LIBS})

add_executable(handmade_curvefitting_g2o handmade_curvefitting_g2o.cpp)
target_link_libraries(handmade_curvefitting_g2o
        g2o_core g2o_stuff
        ${OpenCV_LIBS})

add_executable(BAL_g2o BAL_g2o.cpp)
target_link_libraries(BAL_g2o
        ${G2O_LIBS}
        ${OpenCV_LIBS}
        bal_common
        )