//
// Created by wrk on 2022/3/28.
//

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel_impl.h>
#include <iostream>

#include "common.h"
#include "sophus/se3.hpp"

using namespace Sophus;
using namespace Eigen;
using namespace std;

/*
 * 问题梳理：
 * 1.首先，我们需要同时优化相机的位姿和路标点，相机位姿和路标点分别是顶点，
 * 2.然后，误差=观测-预测，采用重投影误差来当做误差
 * 3.最后，将所有的边和顶点插入到图中，构建g2o优化问题求解即可
 * 问题是第二步怎么求重投影？没有深度信息得不到3d点，还是说直接就给了3d的？
 * */

//先定义如何存放姿态，内参，畸变系数
struct PoseAndIntrinsics
{
    PoseAndIntrinsics() {}  //构造函数

    explicit PoseAndIntrinsics(double *data_addr)  //explicit之后只能用()进行初始化，不能用等号进行赋值
    {
        rotation = SO3d::exp(Vector3d(data_addr[0], data_addr[1], data_addr[2]));  //指数映射转换成SO3，罗德里格斯公式
        translation = Vector3d(data_addr[3], data_addr[4], data_addr[5]);
        focal = data_addr[6];
        k1 = data_addr[7];
        k2 = data_addr[8];
    }

    //将估计值放入内存
    void set_to(double *data_addr)
    {
        auto r = rotation.log();   //对数变换得se3
        for(int i=0;i<3;++i) data_addr[i] = r[i];
        for(int i=0;i<3;++i) data_addr[i+3] = translation[i];
        data_addr[6] = focal;
        data_addr[7] = k1;
        data_addr[8] = k2;
    }

    SO3d rotation;  //旋转SO3
    Vector3d translation;  //平移向量
    double focal = 0;  // 焦距
    double k1=0,k2=0;   //畸变系数
};

class VertexPoseAndIntrinsics: public g2o::BaseVertex<9, PoseAndIntrinsics>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  //重写new，使内存对齐

    VertexPoseAndIntrinsics() {}

    //
    virtual void setToOriginImpl() override
    {
        _estimate = PoseAndIntrinsics();  //这就是要估计的对象：9维数据都需要被估计
    }

    // 估计更新
    virtual void oplusImpl(const double *update) override{
        _estimate.rotation = SO3d::exp(Vector3d(update[0], update[1], update[2])) * _estimate.rotation;
        _estimate.translation += Vector3d(update[3],update[4],update[5]);
        _estimate.focal += update[6];
        _estimate.k1 += update[7];
        _estimate.k2 += update[8];
    }

    //根据估计值投影一个点(估计的是相机系下的3d点)
    Vector2d project(const Vector3d &point){
        Vector3d pc = _estimate.rotation * point + _estimate.translation;
        pc = -pc / pc[2];
        /*这个畸变在数据官网有解释*/
        double r2 = pc.squaredNorm();  //2范数平方
        double distortion = 1.0 + r2 * (_estimate.k1 + _estimate.k2 * r2);
        return Vector2d(_estimate.focal * distortion * pc[0],
                        _estimate.focal * distortion * pc[1]);
    }

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}
};

class VertexLandMark: public g2o::BaseVertex<3, Eigen::Vector3d>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  //重写new，使内存对齐

    VertexLandMark() {}

    virtual void setToOriginImpl() override
    {
        _estimate = Eigen::Vector3d(0,0,0);  //这就是要估计的对象：9维数据都需要被估计
    }

    virtual void oplusImpl(const double *update) override{
        _estimate += Eigen::Vector3d(update[0], update[1], update[2]);
    }

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}
};

//传入参数参数2 ：观测值（这里是3D点在像素坐标系下的投影坐标）的维度
//参数Vector ：观测值类型，piexl.x，piexl.y
//参数VertexSBAPointXYZ：第一个顶点类型
//参数VertexSE3Expmap ：第二个顶点类型
class EdgeProjection: public g2o::BaseBinaryEdge<2, Vector2d, VertexPoseAndIntrinsics, VertexLandMark>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

//    EdgeProjection(double x): BaseBinaryEdge(),
    virtual void computeError() override  //使用override显式地声明虚函数覆盖
    {
        VertexPoseAndIntrinsics * v0 = dynamic_cast<VertexPoseAndIntrinsics *>(_vertices[0]);  //读取位姿9维顶点
        VertexLandMark * v1 = dynamic_cast<VertexLandMark *>(_vertices[1]);  //读取位姿9维顶点
        // 利用估计的相机位姿和路标点的3d坐标将3d坐标重投影称像素坐标，与观测数据_measurement(实际上就是传进来的Vector2d)计算error，
        // 再由g2o自己计算雅可比，或者我么你自己定义那个2*6的雅可比
        auto proj = v0->project(v1->estimate());
        _error = proj - _measurement;    //为什么不是观测-预测？？
    }

    // use numeric derivatives
    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}


};


/*构建g2o问题并求解*/
void SolveBA(BALProblem &bal_problem) {
    const int point_block_size = bal_problem.point_block_size();
    const int camera_block_size = bal_problem.camera_block_size();  //位姿，内参，畸变系数
    double *points = bal_problem.mutable_points();      //  观测点的起始地址
    double *cameras = bal_problem.mutable_cameras();    //  camera参数的起始地址

    // pose dimension 9， landmark is 3
    // 1，2定义blocksolver和linearsolver类型
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<9, 3>> BlockSolverType;
    // 用到对应的LinearSolver就得include对应的.h文件，比如这里的linear_solver_csparse,h
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
    // 3.选择优化算法，创建总求解器
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    // 4.创建稀疏优化器
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);
    // 5.定义图的顶点和边，添加到稀疏优化器中
    const double *observations = bal_problem.observations();  //这个观测值就是前面的4维数据<camera_index_1> <point_index_1> <x_1> <y_1>
    vector<VertexPoseAndIntrinsics *> vertex_pose_intrinsics;  //9维顶点临时变量
    vector<VertexLandMark *> vertex_points;  //3位landmark顶点临时变量
    // 插入相机位姿顶点：3维罗德里格斯旋转向量R，3维平移t，1维焦距f，2维径向畸变系数
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        VertexPoseAndIntrinsics *v = new VertexPoseAndIntrinsics();
        double *camera = cameras + camera_block_size * i;
        v->setId(i);
        v->setEstimate(PoseAndIntrinsics(camera));  //待估计的变量是一个9维的结构体对象
        optimizer.addVertex(v);
        vertex_pose_intrinsics.push_back(v);
    }
    // 插入路标点(这个)
    for (int i = 0; i < bal_problem.num_points(); ++i)
    {
        VertexLandMark *v = new VertexLandMark();
        double *point = points + point_block_size * i;
        v->setId(i + bal_problem.num_cameras());  //从camera后面开始继续编号
        v->setEstimate(Vector3d(point[0], point[1], point[2]));  //对每个顶点设置估计的初值
        // g2o在BA中需要手动设置待Marg的顶点,在这里设置就是要把路标点给marg掉
        v->setMarginalized(true);
        optimizer.addVertex(v);
        vertex_points.push_back(v);
    }

    // 插入边
    for(int i=0; i < bal_problem.num_observations(); ++i)  //观测的数量,2个值算一组观测，所以取值的时候2*i
    {
        EdgeProjection *edge = new EdgeProjection;
        edge->setVertex(0, vertex_pose_intrinsics[bal_problem.camera_index()[i]]);
        edge->setVertex(1, vertex_points[bal_problem.point_index()[i]]);
        edge->setMeasurement(Vector2d(observations[2*i+0], observations[2*i+1]));
        edge->setInformation(Matrix2d::Identity());  //使用单位阵作为协方差矩阵
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(40);

    //优化完成，保存
    //更新相机位姿那9维数据
    for(int i=0; i<bal_problem.num_cameras(); ++i)
    {
        double *camera = cameras + camera_block_size * i;  //找camera地址
        auto vertex = vertex_pose_intrinsics[i];  //取出优化后的9维顶点
        auto estimate = vertex->estimate();  //读取估计值
        estimate.set_to(camera);    //保存至9维顶点
    }
    //更新三维点坐标
    for(int i=0; i<bal_problem.num_points(); ++i)
    {
        double *point = points + point_block_size * i;  //找三维点地址
        auto vertex = vertex_points[i];
        for(int k=0; k<3; ++k)
            point[k] = vertex->estimate()[k];
    }
}


int main(int argc, char** argv)
{
    if (argc != 2) {
        cout << "usage: BAL_g2o bal_data.txt" << endl;
        return 1;
    }
    BALProblem bal_problem(argv[1]);
    bal_problem.Normalize();
    bal_problem.Perturb(0.1, 0.5, 0.5);  //R,t,P标准差，加入噪声
    bal_problem.WriteToPLYFile("initial_g2o.ply");
    SolveBA(bal_problem);
    bal_problem.WriteToPLYFile("final_g2o.ply");
    return 0;
}
