INCLUDE_DIRECTORIES(../include)  #添加一个头文件搜索路径
FIND_PACKAGE(gflags REQUIRED)  #找到gflags包
FIND_PACKAGE(glog REQUIRED)    #找到glog包
ADD_EXECUTABLE(sayhello useHello.cpp)
TARGET_LINK_LIBRARIES(sayhello hello gflags glog)  #最重要的事情：千万不要忘记把所有的库都link到sayhello上
