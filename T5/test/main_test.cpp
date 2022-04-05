#include <gtest/gtest.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <hello.h>



TEST(FactorialTest, ZeroInput){
        EXPECT_EQ(my_fac(0), 1);
}

TEST(FactorialTest, PositiveInput){
        EXPECT_EQ(my_fac(1), 1);
        EXPECT_EQ(my_fac(2), 2);
        EXPECT_EQ(my_fac(3), 6);
}
