#include "hello.h"
#include <iostream>

#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_int32(tmp, 100, "This is temp test value!");


void sayHello() 
{
	FLAGS_logtostderr = 1;  //输出到控制台
	LOG(INFO) <<"Hello SLAM";
}

long my_fac(long n)
{
	if(n<1) return 1;
	else	return n*my_fac(n-1);
}
