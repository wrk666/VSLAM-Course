cmake_minimum_required(VERSION 3.21)
project(g2o_mine)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-O3 -std=c++11")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(G2O REQUIRED)
find_package(OpenCV 3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)
find_package(CSparse REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        "/usr/local/include/eigen3/"
        ${CSPARSE_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
)


set(G2O_LIBS g2o_core g2o_stuff g2o_csparse_extension cxsparse)

add_executable(Direct_BA_Mine Direct_BA_Mine.cpp)
target_link_libraries(Direct_BA_Mine
        g2o_core g2o_stuff
        ${OpenCV_LIBS}
        )

add_executable(directBA directBA.cpp)
target_link_libraries(directBA
        g2o_core g2o_stuff g2o_types_sba
        ${OpenCV_LIBS}
        ${Pangolin_LIBRARIES}
        ${Sophus_LIBS}
        )

add_executable(directBA_withJacobian directBA_withJacobian.cpp)
target_link_libraries(directBA_withJacobian
        g2o_core g2o_stuff g2o_types_sba
        ${OpenCV_LIBS}
        ${Pangolin_LIBRARIES}
        ${Sophus_LIBS}
        )