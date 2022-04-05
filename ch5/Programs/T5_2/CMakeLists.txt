cmake_minimum_required(VERSION 3.21)
project(T5_2)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Debug")

find_package(Pangolin REQUIRED)
find_package(Sophus REQUIRED)
find_package(OpenCV 3 REQUIRED)

include_directories(${Pangolin_INCLUDE_DIRECTORY}
                    "/usr/local/include/eigen3"
                    ${OpenCV_INCLUDE_DIRS}
                    ${Sophus_INCLUDE_DIRS}
)

add_executable(T5_2 main.cpp)
target_link_libraries(T5_2
                      ${Pangolin_LIBRARIES}
                      ${OpenCV_LIBS})

add_executable(T5_2_RMSE main_RMSE.cpp)
target_link_libraries(T5_2_RMSE
        ${Pangolin_LIBRARIES}
        ${OpenCV_LIBS})