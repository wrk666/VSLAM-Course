//
// Created by xiang on 12/21/17.
//

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>

#include <sophus/se3.hpp>

using namespace std;

typedef vector<Vector3d, Eigen::aligned_allocator<Vector3d>> VecVector3d;  //allocator是一种类似于new的分配内存的东西，叫空间配置器，通过链表来实现内存的动态分配和释放，内存效率高
typedef vector<Vector2d, Eigen::aligned_allocator<Vector3d>> VecVector2d;
typedef Matrix<double, 6, 1> Vector6d;

string p3d_file = "./p3d.txt";
string p2d_file = "./p2d.txt";

int main(int argc, char **argv)
{
    VecVector2d p2d;
    VecVector3d p3d;
    Matrix3d K;
    double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;

    // load points in to p3d and p2d
    // START YOUR CODE HERE
    double data2d[2] = {0}, data3d[3] = {0};
    ifstream fin2d(p2d_file), fin3d(p3d_file);
    for(int i=0;i<76;++i)
    {
        fin2d>>data2d[0];
        fin2d>>data2d[1];
        p2d.push_back(Eigen::Vector2d(data2d[0], data2d[1]));
        fin3d>>data3d[0];
        fin3d>>data3d[1];
        fin3d>>data3d[2];
        p3d.push_back(Eigen::Vector3d(data3d[0], data3d[1], data3d[2]));
    }

    // END YOUR CODE HERE
    assert(p3d.size() == p2d.size());

    int iterations = 100;
    double cost = 0, lastCost = 0;
    int nPoints = p3d.size();
    cout << "points: " << nPoints << endl;

    Sophus::SE3d T_esti; // estimated pose,李群，不是李代数，李代数是se3，是Vector3d

    for (int iter = 0; iter < iterations; iter++) {

        Matrix<double, 6, 6> H = Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();

        cost = 0;
        // compute cost  计算误差,是 观测-预测
        for (int i = 0; i < nPoints; i++)
        {
            // compute cost for p3d[I] and p2d[I]
            // START YOUR CODE HERE
        Eigen::Vector3d pc = T_esti * p3d[i];  //3D点转换到相机坐标系下(取了前3维)
        double inv_z = 1.0 / pc[2];
        double inv_z2 = inv_z * inv_z;
        Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);  //重投影，预测
        Eigen::Vector2d e = p2d[i] - proj;
        cost += e.transpose() * e;  // cost += e.squaredNorm();2范数平方
	    // END YOUR CODE HERE

	    // compute jacobian
            Matrix<double, 2, 6> J;
            // START YOUR CODE HERE
        J<<fx * inv_z,
           0,
           -fx * pc[0] * inv_z2,
           -fx * pc[0] * pc[1] * inv_z2,
           fx + fx * pc[0] * pc[0] * inv_z2,
           -fx * pc[1] * inv_z,
           0,
           fy * inv_z,
           -fy * pc[1] * inv_z2,
           -fy - fy * pc[1] * pc[1] * inv_z2,
           fy * pc[0] * pc[1] * inv_z2,
           fy * pc[0] * inv_z;
        J = -J;
	    // END YOUR CODE HERE
            //高斯牛顿的系数矩阵和非齐次项
            H += J.transpose() * J;
            b += -J.transpose() * e;
        }

	// solve dx
        Vector6d dx;  //解出来的△x是李代数

        // START YOUR CODE HERE
        dx = H.ldlt().solve(b);  //解方程
        // END YOUR CODE HERE

        if (isnan(dx[0]))
        {
            cout << "result is nan!" << endl;
            break;
        }

        if (iter > 0 && cost >= lastCost) {
            // cost increase, update is not good
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }

        // update your estimation
        // START YOUR CODE HERE
        T_esti = Sophus::SE3d::exp(dx) * T_esti;

        // END YOUR CODE HERE

        lastCost = cost;

        cout << "iteration " << iter << " cost=" << cout.precision(12) << cost << endl;
    }

    cout << "estimated pose: \n" << T_esti.matrix() << endl;
    return 0;
}
