//
// Created by wrk on 2022/3/27.
//

#include <iostream>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>

using namespace std;
//using namespace g2o;  //using指示，所有的名字都是可见的

//这里需要优化的变量是曲线的abc参数，所以参数1：3维，参数2：用3维向量来表示。
class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   //内存对齐宏，重载了new函数，使得分配的对象都是16Bytes对齐的

    // 重置，把估计值置0即可
    virtual void setToOriginImpl() override {
        _estimate <<0,0,0;
    }

    // 更新   处理xk+1 = xk + ∆x，每个优化的加法运算不一致，可能还包括向量，所以需要重写。
    virtual void oplusImpl(const double *update) override
    {
        _estimate +=Eigen::Vector3d(update);
    }

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}
};


//当男人看向女人的时候，女人就看向了别处，当男人看向远方的时候，女人就看向了你。

//一元边(相当于仅优化相机位姿)，参数1：观测值的维度（这里是y，一维）；参数2：观测值类型；参数3：顶点类型，这里是自定义的顶点
class CurveFittingEdge : public g2o::BaseUnaryEdge<1,double,CurveFittingVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CurveFittingEdge(double x): BaseUnaryEdge(), _x(x){}  //构造函数，先执行基类构造函数，再执行剩余的初始化部分

    //
    //虚函数：基类的指针和引用，指向派生类，调用派生类的虚函数；指向基类，调用基类的虚函数
    virtual void computeError() override  //使用override显式地声明虚函数覆盖
    {
        const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);  //_vertices是pyper_graph.h中的
        const Eigen::Vector3d abc = v->estimate();
        _error(0, 0) = _measurement - std::exp(abc(0, 0) * _x * _x + abc(1,0)*_x + abc(2,0));   //定义误差=观测-预测
    }

    virtual void linearizeOplus() override
    {
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex *>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        double y = std::exp(abc[0] * _x * _x + abc[1]*_x + abc[2]);
        //J=观测-预测    (\partial J)/(\partial a)  (\partial J)/(\partial b)  (\partial J)/(\partial c)
        _jacobianOplusXi[0] = -_x * _x * y;
        _jacobianOplusXi[1] = -_x * y;
        _jacobianOplusXi[2] = -y;
    }

    virtual bool read(istream &in) {}
    virtual bool write(ostream &out) const {}

public:
    double _x;   // x是数据， y为 _measurement观测

};

//书上的曲线拟合的例子,曲线方程：y=exp(ax^w+bx+c)，添加噪声w.给定曲线的数据，用数据来估计a，b，c
int main(int argc, char** argv)
{
    double ar = 1.0, br = 2.0, cr = 1.0;    // 真实参数值
    double ae = 2.0, be = -1.0, ce = 5.0;   //估计值
    int N=100;                              //数据点
    double w_sigma = 1.0;                   //噪声的标准差
    double inv_sigma = 1.0/w_sigma;         //标准差倒数
    cv::RNG rng;                            // OpenCV随机数产生器

    //定义数据
    vector<double> x_data, y_data;
    //生成数据
    for(int i=0; i<N;++i)
    {
        double x=i/100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar*x*x + br*x + cr) + rng.gaussian(w_sigma * w_sigma));  //原始的曲线的数据加上高斯噪声
    }

    //下面来构建图优化模型
    // g2o步骤
    //    1.创建BlockSolver，并用下面面定义的线性求解器初始化。
    //    2.创建一个线性求解器LinearSolver。
    //    3.创建总求解器solver，并从GN/LM/DogLeg 中选一个作为迭代策略，再用上述块求解器BlockSolver初始化。
    //    4.创建图优化的核心：稀疏优化器（SparseOptimizer）。
    //    5.定义图的顶点和边，并添加到SparseOptimizer中。
    //    6.设置优化参数，开始执行优化。

//    1,2:创建blocksolver并使用linearsolver对其进行初始化
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3,1>> BlockSolverType;
    //定义linearsolver为LinearSolverDense，继承自类模板LinearSolver，传入参数MatrixType后实例化为模板类
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;  //使用dense cholesky分解法

    //3.创建总求解器solver，选择一个优化算法，BlockSolver<-LinearSolver
    // 并使用BlockSolver的unique_ptr指针初始化，而BlockSolver又是使用LinearSolver的unique_ptr来初始化的
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

    //4.创建图优化的核心：稀疏优化器（SparseOptimizer）
    g2o::SparseOptimizer optimizer;     //图模型
    optimizer.setAlgorithm(solver);     //
    optimizer.setVerbose(true);         //打开调试输出

    //5.定义边和顶点
    //往图中增加顶点，是abc三维向量
    CurveFittingVertex *v = new CurveFittingVertex();  //new一个顶点对象并指向它
    v->setEstimate(Eigen::Vector3d(ae, be, ce));  //设置估计初始值
    v->setId(0);    //设置顶点的编号
    optimizer.addVertex(v);

    // 往图中增加边，设置边的属性，然后加入到优化器中
    for(int i=0; i<N; ++i)
    {
        CurveFittingEdge* edge = new CurveFittingEdge(x_data[i]);
        edge->setId(i);
        edge->setVertex(0,v);               // 设置连接的顶点，这里只有一个顶点，所以是第0个顶点。v是定义的Vertex对象的指针，如果使用
        edge->setMeasurement(y_data[i]);    // 观测数值
        // 信息矩阵：协方差矩阵之逆（对角阵的逆是对角阵的各个元素的倒数组成的矩阵）
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1/ (w_sigma * w_sigma));

        optimizer.addEdge(edge);
    }

//    6.设置优化参数，开始执行优化。
    cout<<"开始优化"<<endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();  //初始化
    optimizer.optimize(10);  //迭代次数
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout<< "优化耗时=" <<time_used.count()<<" seconds. "<<endl;

    //输出优化值,只有一个顶点，就直接访问即可
    Eigen::Vector3d abc_estimate = v->estimate();
    cout<<"估计的模型："<<abc_estimate.transpose()<<endl;

    return 0;
}


