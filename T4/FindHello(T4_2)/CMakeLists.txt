PROJECT(Find)
cmake_minimum_required(VERSION 3.16)
SET(HELLO_DIR /usr/local/lib)  #库文件位置
SET(CMAKE_MODULE_PATH /usr/local/lib/cmake/hello)  #FindHELLO.cmake位置
ADD_SUBDIRECTORY(src)
