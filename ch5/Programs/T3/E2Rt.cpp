//
// Created by 高翔 on 2017/12/19.
// 本程序演示如何从Essential矩阵计算R,t
//

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace Eigen;

#include <sophus/so3.hpp>

#include <iostream>

using namespace std;

int main(int argc, char **argv) {

    // 给定Essential矩阵
    Matrix3d E;
    E << -0.0203618550523477, -0.4007110038118445, -0.03324074249824097,
            0.3939270778216369, -0.03506401846698079, 0.5857110303721015,
            -0.006788487241438284, -0.5815434272915686, -0.01438258684486258;

    // 待计算的R,t
    Matrix3d R;
    Vector3d t;

    // SVD and fix sigular values
    // START YOUR CODE HERE

    // SVD on Sigma
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);  //Eigen的svd函数，计算满秩的U和V
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Vector3d  sv = svd.singularValues();
    cout << "U=" << U << endl;
    cout << "V=" << V << endl;
    cout << "sv=" << sv << endl;

    Eigen::Matrix3d Sigma = Eigen::Matrix3d::Zero();
    Sigma(0,0) = sv(0);
    Sigma(1,1) = sv(1);

    cout << "Sigma:\n" << Sigma << endl;
    // END YOUR CODE HERE

    // set t1, t2, R1, R2
    // START YOUR CODE HERE

    //use AngleAxis
    Eigen::AngleAxisd rotation_vector_neg ( -M_PI/2, Eigen::Vector3d ( 0,0,1 ) );     //沿 Z 轴旋转 -90 度
    Eigen::AngleAxisd rotation_vector_pos ( M_PI/2, Eigen::Vector3d ( 0,0,1 ) );     //沿 Z 轴旋转 90 度
    Eigen::Matrix3d  RzNegHalfPi = rotation_vector_neg.toRotationMatrix();
    Eigen::Matrix3d  RzPosHalfPi = rotation_vector_pos.toRotationMatrix();

    //use Sophus
//    Sophus::SO3d SO3_v( 0, 0, M_PI/2 );  // 亦可从旋转向量构造


    Matrix3d t_wedge1 = U * RzPosHalfPi * Sigma * U.transpose();
    Matrix3d t_wedge2 = U * RzNegHalfPi * Sigma * U.transpose();

    Matrix3d R1 = U * RzPosHalfPi.transpose() * V.transpose();
    Matrix3d R2 = U * RzNegHalfPi.transpose() * V.transpose();
    // END YOUR CODE HERE

    cout << "R1 = \n" << R1 << endl;
    cout << "R2 = \n" << R2 << endl;
    cout << "t1 = \n" << Sophus::SO3d::vee(t_wedge1) << endl;  //求李代数？？
    cout << "t2 = \n" << Sophus::SO3d::vee(t_wedge2) << endl;

    // check t^R=E up to scale
    Matrix3d tR = t_wedge1 * R1;
    cout << "t^R = " << tR << endl;

    return 0;
}