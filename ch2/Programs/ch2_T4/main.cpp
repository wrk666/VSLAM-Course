#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
using namespace std;
using namespace Eigen;

int main(int argc, char ** argv)
{
    //创建视觉传感器和激光传感器的四元数（构造函数顺序是wxyz，而存储的系数默认是xyzw）
    Quaterniond q_BL(0.3, 0.5,0, 20.1 ), q_BC(0.8, 0.2, 0.1,0.1);
    //四元数归一化
    q_BC.normalize();  //如果将四元数表示为旋转的话，如果四元数没有归一化的话，那么旋转是未定义的行为
    q_BL.normalize();

    //平移向量t_BL和t_BC
    Vector3d t_BL(0.4, 0, 0.5), t_BC(0.5, 0.1, 0.5);
    //p_C坐标
    Vector3d p_C(0.3, 0.2, 1.2);

    //构造变换矩阵Tc1w和Tc2w
    Isometry3d T_BL(q_BL), T_BC(q_BC);
    T_BL.pretranslate(t_BL);
    T_BC.pretranslate(t_BC);

    //计算p_L
    Vector3d p_L = T_BL.inverse() * T_BC * p_C;
    cout << "这个点在激光系下的坐标：" << p_L.transpose() << endl;

    //四元数求解p_L  这个不等价，平移不能直接取反，放到变换矩阵中一起求逆
    Vector3d p_LQ = q_BL.inverse() * (q_BC*p_C + t_BC)-q_BL.inverse()*t_BL;
    cout << "四元数解得这个点在激光系下的坐标：" << p_LQ.transpose() << endl;


    //计算在世界坐标系下的坐标
    //创建世界系和机器人本体的四元数
    Quaterniond q_WR(0.55, 0.3, 0.2,0.2), q_RB(0.99, 0, 0,0.01);
    //四元数归一化
    q_WR.normalize();
    q_RB.normalize();

    //平移向量t_WR，t_RB
    Vector3d t_WR(0.1, 0.2, 0.3), t_RB(0.05, 0, 0.5);

    //构造变换矩阵T_WR和T_RB
    Isometry3d T_WR(q_WR), T_RB(q_RB);
    T_WR.pretranslate(t_WR);
    T_RB.pretranslate(t_RB);

    //计算p_W
    Vector3d p_W = T_WR * T_RB * T_BC * p_C;
    cout << "这个点在世界系下的坐标:" << p_W.transpose() << endl;

//    //四元数计算世界坐标   这个还不对，再看看
//    Vector3d p_WQ = q_WR.inverse()*(q_RB.inverse()*(q_BC*p_C+t_BC)-q_RB.inverse()*t_RB)-q_WR.inverse()*t_WR;
//    cout << "四元数解得这个点在世界系下的坐标：" << p_WQ.transpose() << endl;
    return 0;
}
