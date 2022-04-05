ADD_EXECUTABLE(sayhello useHello.cpp)
TARGET_LINK_LIBRARIES(sayhello hello)
INCLUDE_DIRECTORIES(../include)  #添加一个头文件搜索路径
