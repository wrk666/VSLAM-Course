//
// Created by xiang on 1/4/18.
// this program shows how to perform direct bundle adjustment
//
#include <iostream>



#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>

#include <pangolin/pangolin.h>
#include <boost/format.hpp>

using namespace std;
using namespace Eigen;

typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> VecSE3;  //装相机位姿
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVec3d;   //装3d路标点

// global variables
string pose_file = "./poses.txt";
string points_file = "./points.txt";

// intrinsics
float fx = 277.34;
float fy = 291.402;
float cx = 312.234;
float cy = 239.777;

// bilinear interpolation  双线性插值读取图像的灰度值
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[img.step] +
            xx * yy * data[img.step + 1]
    );
}

// g2o vertex that use sophus::SE3 as pose  自定义位姿顶点，数据类型是SE3d
class VertexSophus : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VertexSophus() {}

    ~VertexSophus() {}


    virtual void setToOriginImpl() {
        _estimate = Sophus::SE3d();
    }

    //根据估计值投影一个点(估计的是相机系下的3d点)
    Vector2f project(const Vector3d &point)
    {
        //KTP 读取SE(3)，转换，投影为像素坐标，访问像素灰度值计算光度误差
        Sophus::SE3d Tcw(estimate());
        Vector3d point_cam3d = Tcw * point;
        float u = fx * point_cam3d[0]/point_cam3d[2] + cx;
        float v = fy * point_cam3d[1]/point_cam3d[2] + cy;
        return Vector2f(u,v);
    }

    //更新
    virtual void oplusImpl(const double *update_) {
        //计算se(3)，再由se(3)为SE(3)
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> update(update_);
        //保存估计值，相当于_estimate = Sophus::SE3d::exp(update) * _estimate；
        setEstimate(Sophus::SE3d::exp(update) * estimate());
    }

    virtual bool read(std::istream &is) {}

    virtual bool write(std::ostream &os) const {}
};

// TODO edge of projection error, implement it
// 16x1 error, which is the errors in patch   16个像素点的光度差之和
typedef Eigen::Matrix<double,16,1> Vector16d;
class EdgeDirectProjection : public g2o::BaseBinaryEdge<16, Vector16d, g2o::VertexSBAPointXYZ, VertexSophus>  //一个是SBA的XYZ自带边，一个是自定义的边
        {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    //边构造函数
    EdgeDirectProjection(float *color, cv::Mat &target) {
        this->origColor = color;
        this->targetImg = target;
    }

    ~EdgeDirectProjection() {}

    virtual void computeError() override {
        // TODO START YOUR CODE HERE
        // compute projection error ...
        //projected = KTP  需要使用SE(3)
        g2o::VertexSBAPointXYZ* vertexPw = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[0]);
        VertexSophus* vertexTcw = static_cast<VertexSophus *>(_vertices[1]);
        Vector2f proj = vertexTcw->project(vertexPw->estimate());
        //判断是否越界，若越界，则将error该位置1，并setLevel(1)不知道啥意思，是记录好坏的吗？
        if(proj[0]<-2 || proj[0]+2>targetImg.cols || proj[1]<-2 || proj[1]+2>targetImg.rows)
        {
            this->setLevel(1);  //设置level为1，标记为outlier，下次不再对该边进行优化
            for(int i=0; i<16; ++i)  _error[i] = 0;
        }
        else{
            for(int i=-2; i<2;++i){
                for(int j=-2; j<2; ++j){
                    /*******************************************************/
                    int num = 4 * i + j + 10;   //为什么要加10？？？？？？？？？？？？？？？？？？？？？？？
                    /*******************************************************/
                    //_measurement是一个16*1的向量，所以_error也是16*1
                    _error[num] = origColor[num] - GetPixelValue(targetImg, proj[0]+i, proj[1]+j);
                }
            }
        }
        // END YOUR CODE HERE
    }
    // Let g2o compute jacobian for you  自己不算了,后面再算


    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}

private:
    cv::Mat targetImg;  // the target image
    float *origColor = nullptr;   // 16 floats, the color of this point
};

// plot the poses and points for you, need pangolin
void Draw(const VecSE3 &poses, const VecVec3d &points, string title);

int main(int argc, char **argv) {

    // read poses and points
    VecSE3 poses;
    VecVec3d points;
    ifstream fin(pose_file);  //读取相机位姿（7张图，7个位姿）

    //读取位姿
    while (!fin.eof())
    {
        double timestamp = 0;
        fin >> timestamp;
        if (timestamp == 0) break;
        double data[7];
        for (auto &d: data) fin >> d;
        //四元数和平移向量来构建SE3
        poses.push_back(Sophus::SE3d
        (
                Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                Eigen::Vector3d(data[0], data[1], data[2])
        ));
        if (!fin.good()) break;
    }
    fin.close();

    //读取3d路标点XYZ
    vector<float *> color;
    fin.open(points_file);
    while (!fin.eof())
    {
        double xyz[3] = {0};
        for (int i = 0; i < 3; i++) fin >> xyz[i];
        if (xyz[0] == 0) break;
        points.push_back(Eigen::Vector3d(xyz[0], xyz[1], xyz[2]));
        float *c = new float[16];
        for (int i = 0; i < 16; i++) fin >> c[i];
        color.push_back(c);

        if (fin.good() == false) break;
    }
    fin.close();

    cout << "poses: " << poses.size() << ", points: " << points.size() << endl;


    // read images 读取所有图片
    vector<cv::Mat> images;
    boost::format fmt("./%d.png");
    for (int i = 0; i < 7; i++)
    {
        images.push_back(cv::imread((fmt % i).str(), 0));
    }

    // build optimization problem
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> DirectBlock;  // 求解的向量是6＊1的
    typedef g2o::LinearSolverDense<DirectBlock::PoseMatrixType> LinearSolverType;
//    DirectBlock::LinearSolverType *linearSolver = new g2o::LinearSolverDense<DirectBlock::PoseMatrixType>();
//    DirectBlock *solver_ptr = new DirectBlock(linearSolver);
    auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<DirectBlock>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // TODO add vertices, edges into the graph optimizer
    vector<g2o::VertexSBAPointXYZ *> vertex_points;  //3位landmark顶点临时变量
    vector<VertexSophus *> vertex_pose;  //pose顶点临时变量

    // START YOUR CODE HERE
    //插入路标顶点
    for(int i=0; i<points.size(); ++i)
    {
        g2o::VertexSBAPointXYZ *v = new g2o::VertexSBAPointXYZ;
        v->setId(i);
        v->setEstimate(points[i]);
        v->setMarginalized(true);   //设置边缘化路标点
        optimizer.addVertex(v);
        vertex_points.push_back(v);
    }
    //插入位姿顶点
    for(int i=0; i<poses.size(); ++i)
    {
        VertexSophus *v = new VertexSophus();
        v->setId(i + points.size());
        v->setEstimate(poses[i]);
        optimizer.addVertex(v);
        vertex_pose.push_back(v);
    }



    //插入边
    for(int c=0; c<poses.size(); ++c)
        for(int p=0; p<points.size(); ++p)
        {
            EdgeDirectProjection *edge = new EdgeDirectProjection(color[p], images[c]);  //每个图中的每个点都插入到优化图中，都有一条边
            //先point后pose
            edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(p)));
            edge->setVertex(1, dynamic_cast<VertexSophus *>(optimizer.vertex(points.size()+c)));
//            edge->setMeasurement(Vector16d );
            // 信息矩阵可直接设置为 error_dim*error_dim 的单位阵
            edge->setInformation(Eigen::Matrix<double, 16, 16>::Identity());
            // 设置Huber核函数，减小错误点影响，加强鲁棒性
            g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
            rk->setDelta(1.0);  //A squared error above delta^2 is considered as outlier in the data
            edge->setRobustKernel(rk);
            optimizer.addEdge(edge);
        }

    // END YOUR CODE HERE

    Draw(poses, points, string("before"));

    // perform optimization
    optimizer.initializeOptimization(0);
    optimizer.optimize(1000);


    // TODO fetch data from the optimizer
    // START YOUR CODE HERE
    for(int c=0; c<poses.size(); ++c)
        for(int p=0; p<points.size(); ++p)
        {
            points[p] = dynamic_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(p))->estimate();
            poses[c] = dynamic_cast<VertexSophus *>(optimizer.vertex(points.size()+c))->estimate();
        }

    // END YOUR CODE HERE

    // plot the optimized points and poses
    Draw(poses, points, "after");

    // 看看这数据有没有什么不一样的？怎么看优化的对不对？


    // delete color data
    for (auto &c: color) delete[] c;
    return 0;
}

void Draw(const VecSE3 &poses, const VecVec3d &points, string title) {
    if (poses.empty() || points.empty()) {
        cerr << "parameter is empty!" << endl;
        return;
    }

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
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

        // draw poses
        float sz = 0.1;
        int width = 640, height = 480;
        for (auto &Tcw: poses) {
            glPushMatrix();
            Sophus::Matrix4f m = Tcw.inverse().matrix().cast<float>();
            glMultMatrixf((GLfloat *) m.data());
            glColor3f(1, 0, 0);
            glLineWidth(2);
            glBegin(GL_LINES);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glEnd();
            glPopMatrix();
        }

        // points
        glPointSize(2);
        glBegin(GL_POINTS);
        for (size_t i = 0; i < points.size(); i++) {
            glColor3f(0.0, points[i][2]/4, 1.0-points[i][2]/4);
            glVertex3d(points[i][0], points[i][1], points[i][2]);
        }
        glEnd();

        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
}

