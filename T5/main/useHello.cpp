#include <iostream>
#include "hello.h"
#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_int32(print_times, 1, "The print times");

int main( int argc, char** argv ) 
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);  //用于接受命令行的flag参数并更新默认参数
        google::InitGoogleLogging("daqing");    //初始化一个log,这个参数我还不知道怎么设置
        FLAGS_logtostderr = 1;  //输出到控制台
	for(int i=FLAGS_print_times; i>0;--i)
	  sayHello();
	//LOG(INFO)<<"FLAGS_tmp的值为:"<<FLAGS_tmp;

	//以下是glog的使用
	//LOG(INFO) << "info test";  //输出一个Info日志
	//LOG(WARNING) << "hello i am warning test";  //输出一个Warning日志
	//LOG(ERROR) << "hello i am error test";  //输出一个Error日志
	google::ShutdownGoogleLogging();    //不用log的时候应该释放掉，不然会内存溢出
	return 0;	
}
