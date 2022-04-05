INCLUDE_DIRECTORIES(../include)  #hello.h搜索路径
FIND_PACKAGE(gflags REQUIRED)  #找到gflags包
FIND_PACKAGE(glog REQUIRED)    #找到glog包


set(CMAKE_CXX_STANDARD 11)
include(FetchContent)  #载入模块
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

#后面又添加的
enable_testing()
add_executable(  main_test  main_test.cpp)
target_link_libraries(main_test  gtest_main hello gflags glog)

#使CMake runner能够在二进制中找到测试，使用GoogleTest模块
include(GoogleTest)  #用来载入 CMakeLists.txt 文件，也用于载入预定义的 cmake 模块. INCLUDE(file1 [OPTIONAL])  INCLUDE(module [OPTIONAL])
gtest_discover_tests(main_test) 
