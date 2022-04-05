//
// Created by wrk on 2022/2/11.
//

#include <iostream>
#include <fstream>
#include <unistd.h>
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>

using namespace Sophus;
using namespace std;

string groundtruth_file = "../groundtruth.txt";
string estimated_file = "../estimated.txt";

typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;  //是轨迹的类型

//函数声明
void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti);

TrajectoryType ReadTrajectory(const string &path);

int main(int argc, char **argv)
{
    //读取估计轨迹和真实轨迹
    TrajectoryType  groundtruth = ReadTrajectory(groundtruth_file);
    TrajectoryType estimated = ReadTrajectory(estimated_file);
    assert(!groundtruth.empty() && !estimated.empty());
    assert(groundtruth.size() == estimated.size());

    //计算误差RMSE
    double rmse = 0;
    for(size_t i=0;i<estimated.size();i++){
        Sophus::SE3d p1 = estimated[i], p2=groundtruth[i];
        double error = (p2.inverse()*p1).log().norm();
        rmse += error * error;
    }
    rmse = rmse/double(estimated.size());  //均值
    rmse = sqrt(rmse);   //开方
    cout<<"RMSE = "<<rmse<<endl;

    DrawTrajectory(groundtruth, estimated);
    return 0;
}

//从文件中读取轨迹
TrajectoryType ReadTrajectory(const string &path){
    ifstream fin(path);
    TrajectoryType trajectory;
    if(!fin){
        cerr<<"trajectory "<<path<<"not found."<<endl;
        return trajectory;
    }
    while(!fin.eof()){
        double time, tx, ty, tz, qx, qy, qz, qw;
        fin>>time>>tx>>ty>>tz>>qx>>qy>>qz>>qw;
        //根据四元数和平移向量构造出SE(3)
        Sophus::SE3d p1(Eigen::Quaterniond(qw,qx,qy,qz), Eigen::Vector3d(tx,ty,tz));
        trajectory.push_back(p1);
    }
    return trajectory;
}

void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti){
    //创建Pangolin窗口并画出轨迹
    pangolin::CreateWindowAndBind("My Trajectory Viewer Name", 1024, 768);
    //下面3条看不懂
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    //这个s_cam和d_cam分别有啥用？
    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500,512,389,0.1,1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
            );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    //当还没有画完的时候继续画
    while(pangolin::ShouldQuit()==false){
        glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);   //RGB Flpha

        //绘制GroundTruth
        glLineWidth(2);  //线的粗细
        for(size_t i=0; i<gt.size()-1; i++){
            glColor3f(0.0f, 0.0f, 1.0f);   //GroundTruth用蓝线
            glBegin(GL_LINES);
            auto p1=gt[i],p2=gt[i+1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }

        //绘制esti
        glLineWidth(2);  //线的粗细
        for(size_t i=0;i<esti.size()-1;i++){
            glColor3f(1.0f, 0.0f, 0.0f);   //估计的轨迹用红线
            glBegin(GL_LINES);
            auto p1=esti[i],p2=esti[i+1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000);  //sleep 5s
    }
}
