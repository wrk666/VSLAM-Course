cmake_minimum_required(VERSION 2.8)
project(ch3_T2)

set(CMAKE_CXX_STANDARD 11)

include_directories( "/usr/local/include/eigen3" )  #添加头文件

add_executable(ch3_T2 main.cpp)
#因为Eigen只有头文件，所以不需要再target_link_libraries,如果安装到了其他地方，要相应的改库的路径