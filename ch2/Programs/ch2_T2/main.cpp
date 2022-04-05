#include <iostream>
using namespace std;
#include <ctime>
// Eigen 部分
#include <Eigen/Core>
// 稠密矩阵的代数运算（逆，特征值等）
#include <Eigen/Dense>

#define MATRIX_SIZE 100

int main() {

    //解方程
    // 我们求解 matrix_NN * x = v_Nd 这个方程
    // N的大小在前边的宏里定义，它由随机数生成

    // 如果不确定矩阵大小，可以使用动态大小的矩阵，大于100,所以用动态的
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > matrix_NN;

    matrix_NN = Eigen::MatrixXd::Random( MATRIX_SIZE, MATRIX_SIZE );
    Eigen::Matrix< double, MATRIX_SIZE,  1> v_Nd= Eigen::MatrixXd::Random( MATRIX_SIZE,1 ); //非齐次项随机初始化
    Eigen::Matrix<double,MATRIX_SIZE,1> x = Eigen::MatrixXd::Random( MATRIX_SIZE, 1 );  //结果随机初始化

    // 通常用矩阵分解来求，例如QR分解，速度会快很多
    clock_t time_stt = clock(); // 计时
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout <<"time use in Qr decomposition is " <<1000*  (clock() - time_stt)/(double)CLOCKS_PER_SEC <<"ms" << endl;
    cout <<"Qr decomposition x size is:"<<x.size()<<", result is"<<x.transpose()<<endl;

    time_stt = clock(); // 计时
    x = matrix_NN.ldlt().solve(v_Nd);
    cout <<"time use in Qr decomposition is " <<1000*  (clock() - time_stt)/(double)CLOCKS_PER_SEC <<"ms" << endl;
    cout <<"Cholesky decomposition x size is:"<<x.size()<<", result is"<<x.transpose()<<endl;

    return 0;
}
