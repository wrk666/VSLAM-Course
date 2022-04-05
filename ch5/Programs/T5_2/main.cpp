#include <iostream>
#include <fstream>
#include <unistd.h>
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>
#include <opencv2/core/core.hpp>

using namespace Sophus;
using namespace std;
using namespace cv;

string compare_file = "./compare.txt";

typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;
typedef vector<TrajectoryType> LongTrajectoryType;
typedef Eigen::Matrix<double,6,1> Vector6d;

void DrawTrajectory(const vector<Point3d> &gt, const vector<Point3d> &esti, const string& title);
vector<TrajectoryType> ReadTrajectory(const string &path);
vector<Point3d> GetPoint(TrajectoryType TT);
void pose_estimation_3d3d(const vector<Point3d> &pts1, const vector<Point3d> &pts2, Mat &R, Mat &t);
vector<Point3d> TrajectoryTransform(Mat T, Mat t, vector<Point3d> esti );

int main(int argc, char **argv) {
    LongTrajectoryType CompareData = ReadTrajectory(compare_file);
    assert(!CompareData.empty());
    cout<<"size: "<<CompareData.size()<<endl;

    vector<Point3d> EstiPt = GetPoint(CompareData[0]);
    vector<Point3d> GtPt = GetPoint(CompareData[1]);

    Mat R, t;  //待求位姿
    pose_estimation_3d3d( GtPt, EstiPt, R, t);
    cout << "ICP via SVD results: \n" << endl;
    cout << "R = \n" << R << endl;
    cout << "t = \n" << t << endl;
    cout << "R_inv = \n" << R.t() << endl;
    cout << "t_inv = \n" << -R.t() * t << endl;

    DrawTrajectory(GtPt, EstiPt, "Before Calibrate");
    vector<Point3d> EstiCali = TrajectoryTransform(R, t, EstiPt);
    DrawTrajectory(GtPt, EstiCali, "Atfer Calibrate");
    return 0;
}

LongTrajectoryType ReadTrajectory(const string &path)
{
    ifstream fin(path);
    TrajectoryType trajectory1, trajectory2;
    if (!fin) {
        cerr << "trajectory " << path << " not found." << endl;
        return {};
    }

    while (!fin.eof()) {
        double time1, tx1, ty1, tz1, qx1, qy1, qz1, qw1;
        fin >> time1 >> tx1 >> ty1 >> tz1 >> qx1 >> qy1 >> qz1 >> qw1;
        double time2, tx2, ty2, tz2, qx2, qy2, qz2, qw2;
        fin >> time2 >> tx2 >> ty2 >> tz2 >> qx2 >> qy2 >> qz2 >> qw2;
        Sophus::SE3d p1(Eigen::Quaterniond(qw1, qx1, qy1, qz1), Eigen::Vector3d(tx1, ty1, tz1));
        trajectory1.push_back(p1);
        Sophus::SE3d p2(Eigen::Quaterniond(qw2, qx2, qy2, qz2), Eigen::Vector3d(tx2, ty2, tz2));
        trajectory2.push_back(p2);
    }
    LongTrajectoryType ret{trajectory1, trajectory2};
    return ret;
}


void DrawTrajectory(const vector<Point3d> &gt, const vector<Point3d> &esti, const string& title)
{
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind(title, 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));


    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glLineWidth(2);
        for (size_t i = 0; i < gt.size() - 1; i++) {
            glColor3f(0.0f, 0.0f, 1.0f);  // blue for ground truth
            glBegin(GL_LINES);
            auto p1 = gt[i], p2 = gt[i + 1];
            glVertex3d(p1.x, p1.y, p1.z);
            glVertex3d(p2.x, p2.y, p2.z);
            glEnd();
        }

        for (size_t i = 0; i < esti.size() - 1; i++) {
            glColor3f(1.0f, 0.0f, 0.0f);  // red for estimated
            glBegin(GL_LINES);
            auto p1 = esti[i], p2 = esti[i + 1];
            glVertex3d(p1.x, p1.y, p1.z);
            glVertex3d(p2.x, p2.y, p2.z);
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }

}

void pose_estimation_3d3d(const vector<Point3d> &pts1,
                          const vector<Point3d> &pts2,
                          Mat &R, Mat &t) {
    Point3d p1, p2;     // center of mass 质心,这里p1表示第1幅图，p2表示第2幅图，和书上的R是反着的，所以要计算R21=这里的R12^(-1)=R12^(T)，最后也输出了
    int N = pts1.size();
    for (int i = 0; i < N; i++) {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = Point3d(Vec3d(p1) / N);
    p2 = Point3d(Vec3d(p2) / N);
    vector<Point3d> q1(N), q2(N); // remove the center  去质心
    for (int i = 0; i < N; i++) {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; i++)
    {
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();  //这里是2->1  R12,求R21要转置
    }
    cout << "W= \n" << W << endl;

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);  //Eigen的svd函数，计算满秩的U和V
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    cout << "U= \n" << U << endl;
    cout << "V= \n" << V << endl;

    Eigen::Matrix3d R_ = U * (V.transpose());  //这里能保证满足det(R)=1且正交吗？
    cout<<"我的输出： det(R_): "<<R_.determinant()<<"\nR_: \n"<<R_<<endl;  //Eigen的Mat
    if (R_.determinant() < 0)  //若行列式为负，取-R
    {
        R_ = -R_;
    }
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);   //最优的t=p-Rp'

    // convert to cv::Mat
    R = (Mat_<double>(3, 3) <<
            R_(0, 0), R_(0, 1), R_(0, 2),
            R_(1, 0), R_(1, 1), R_(1, 2),
            R_(2, 0), R_(2, 1), R_(2, 2)
    );
    t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}

vector<Point3d> GetPoint(TrajectoryType TT)
{
    vector<Point3d> pts;
    for(auto each:TT)
        //不用做相机模型的处理,也不/5000
        pts.push_back(Point3d(each.translation()[0], each.translation()[1], each.translation()[2]));
    return pts;
}

//转换
vector<Point3d> TrajectoryTransform(Mat T, Mat t, vector<Point3d> esti )
{
    vector<Point3d> calibrated={};
    Mat Mat__31;
    Sophus::SE3d SE3D;
    for(auto each:esti)
    {
        Mat__31 = (Mat_<double>(3, 1)<<each.x, each.y, each.z);
        Mat__31 = T * Mat__31 + t;
        calibrated.push_back( Point3d(Mat__31));
    }
    return calibrated;
}


//这里相当于仅仅是两帧图像间的一次位姿变换，只能得到1个李代数se(3)，而计算RMSE是针对所有轨迹的，
// 如果将这个文件看做正常的轨迹，直接给的就是轨迹的估计和真实，就能计算RMSE，之前做过