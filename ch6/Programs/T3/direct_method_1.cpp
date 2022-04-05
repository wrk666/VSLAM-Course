#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <vector>
#include <string>
#include <boost/format.hpp>
#include <pangolin/pangolin.h>

#include <execution>  //多线程
#include <mutex>      //加锁


using namespace std;
using namespace cv;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

// Camera intrinsics
// 内参
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// 基线
double baseline = 0.573;
// paths
string left_file = "./left.png";
string disparity_file = "./disparity.png";
boost::format fmt_others("./%06d.png");    // other files
boost::format fmt_order("%2d");    // other files

// useful typedefs
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef vector<cv::Point2f> PtVec2f;

// TODO implement this function
/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationMultiLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3d &T21,
        string order
);

// TODO implement this function
/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationSingleLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3d &T21,
        string order
);

void DirectPoseEstimationSingleLayerMT(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,  //随机选择的点
        const vector<double> depth_ref,  //深度参考
        Sophus::SE3d &T21,
        string order
);

// bilinear interpolation
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

// 打印函数执行时间
template <typename FuncT>
void evaluate_and_call(FuncT func, const std::string &func_name = "",
                       int times = 10) {
    double total_time = 0;
    for (int i = 0; i < times; ++i) {
        auto t1 = std::chrono::steady_clock::now();
        func();
        auto t2 = std::chrono::steady_clock::now();
        total_time +=
                std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() * 1000;
    }

    std::cout << "方法 " << func_name
              << " 平均调用时间/次数: " << total_time / times << "/" << times
              << " 毫秒." << std::endl;
}

int main(int argc, char **argv)
{
    cv::Mat left_img = cv::imread(left_file, 0);
    cv::Mat disparity_img = cv::imread(disparity_file, 0);  //左右图像的视差图

    // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng;
    int nPoints = 1000;  //随机选择1000个点
    int boarder = 40;
    VecVector2d pixels_ref;
    vector<double> depth_ref;

    // generate pixels in ref and load depth data
    for (int i = 0; i < nPoints; i++) {
        int x = rng.uniform(boarder, left_img.cols - boarder);  // don't pick pixels close to boarder
        int y = rng.uniform(boarder, left_img.rows - boarder);  // don't pick pixels close to boarder
        int disparity = disparity_img.at<uchar>(y, x);
        double depth = fx * baseline / disparity; // you know this is disparity to depth
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));
    }

    // estimates 01~05.png's pose using this information
    Sophus::SE3d T_cur_ref, T_cur_refMT;

    for (int i = 1; i < 6; i++) {  // 1~10
        cv::Mat img = cv::imread((fmt_others % i).str(), 0);  //图像名字
//        DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref, (fmt_order%i).str());      // 单层测试
//        DirectPoseEstimationSingleLayerMT(left_img, img, pixels_ref, depth_ref, T_cur_ref, (fmt_order%i).str());      // 单层测试
//        DirectPoseEstimationMultiLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref, (fmt_order%i).str());      // 多层测试
        evaluate_and_call([&]() { DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref, (fmt_order%i).str()); },
                          "直接法", 1);
//        evaluate_and_call([&]() { DirectPoseEstimationSingleLayerMT(left_img, img, pixels_ref, depth_ref, T_cur_refMT, (fmt_order%i).str()); },
//                          "直接法MT", 1);
    }
}

void DirectPoseEstimationSingleLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,  //随机选择的点
        const vector<double> depth_ref,  //深度参考
        Sophus::SE3d &T21,
        string order
) {
    // parameters
    int half_patch_size = 4;
    int iterations = 100;

    double cost = 0, lastCost = 0;
    int nGood = 0;  // good projections
    VecVector2d goodProjection;
    VecVector2d GoodRefIndex ;
    VecVector2d projection = VecVector2d(px_ref.size(), Eigen::Vector2d(0, 0)); // projected points
    for (int iter = 0; iter < iterations; iter++) {
        nGood = 0;
        goodProjection.clear();  //每次清零，迭代结束之后就得到了最后的对应点
        GoodRefIndex.clear();

        // Define Hessian and bias
        Matrix6d H = Matrix6d::Zero();  // 6x6 Hessian
        Vector6d b = Vector6d::Zero();  // 6x1 bias

        for (size_t i = 0; i < px_ref.size(); i++)
        {
            // compute the projection in the second image
            // TODO START YOUR CODE HERE
            Eigen::Vector3d point_ref = depth_ref[i] * Eigen::Vector3d((px_ref[i][0]-cx)/fx, (px_ref[i][1]-cy)/fy, 1);  //ref中的3D点坐标
            Eigen::Vector3d point_cur = T21 * point_ref;  //ref中的3D点转换到cur中的3D点
            if (point_cur[2] < 0)   // depth invalid
                continue;

            float u = fx * point_cur[0]/point_cur[2] + cx, v = fy * point_cur[1]/point_cur[2] + cy;
            if(u<half_patch_size || u+half_patch_size>img2.cols || v<half_patch_size || v+half_patch_size>img2.rows)  //变换到cur中若越界则不优化
                continue;

            double X = point_cur[0], Y = point_cur[1], Z = point_cur[2], inv_z = 1.0 / Z, inv_z2 = inv_z * inv_z;  //cur中的3D坐标X'Y'Z'
            nGood++;

            //记录投影前后的uv坐标
            goodProjection.push_back(Eigen::Vector2d(u, v));
            GoodRefIndex.push_back(Eigen::Vector2d(px_ref[i][0],px_ref[i][1]));
//            projection[i] = Eigen::Vector2d(u, v);

            // and compute error and jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++)
                {
                    // 假设同一窗口内的深度信息不变；灰度不变（同一空间点在各个视角下成像的灰度不变）
                    double error =  GetPixelValue(img1, px_ref[i][0]+x, px_ref[i][1]+y) - GetPixelValue(img2, u+x, v+y);
                    Eigen::Vector2d J_img_pixel;    // image gradients(2*1)  像素梯度,使用cur中的像素坐标和窗口偏移量x，y计算*
                    J_img_pixel<<(1.0 / 2) * (GetPixelValue(img2, u+1+x, v+y)-GetPixelValue(img2, u-1+x, v+y)),
                            (1.0 / 2) * (GetPixelValue(img2, u+x, v+1+y)-GetPixelValue(img2, u+x, v-1+y));

                    Matrix26d J_pixel_xi;   // pixel to \xi in Lie algebra  2*6
                    J_pixel_xi<<fx * inv_z,
                                0,
                                -fx * X * inv_z2,
                                -fx * X * Y * inv_z2,
                                fx + fx * X * X * inv_z2,
                                -fx * Y * inv_z,
                                0,
                                fy * inv_z,
                                -fy * Y * inv_z2,
                                -fy - fy * Y * Y * inv_z2,
                                fy * X * Y * inv_z2,
                                fy * X * inv_z;

                    // total jacobian   应该是1*6的
                    Vector6d J=-1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose();

                    H += J * J.transpose();
                    b += -error * J;
                    cost += error * error;
                }
            // END YOUR CODE HERE
        }

        // solve update and put it into estimation
        // TODO START YOUR CODE HERE
        Vector6d update = H.ldlt().solve(b);
        T21 = Sophus::SE3d::exp(update) * T21;  //李群更新
        // END YOUR CODE HERE

        cost /= nGood;

        if (isnan(update[0]))
        {
            // sometimes occurred when we have a black or white patch and H is irreversible
            cout << "update is nan" << endl;
            break;
        }
        if (iter > 0 && cost > lastCost) {
            cout << "cost increased: " << cost << ", " << lastCost << endl;
            break;
        }
        lastCost = cost;
        cout << "cost = " << cost << ", good = " << nGood << endl;
    }
    cout << "good projection: " << nGood << endl;
    cout << "T21 = \n" << T21.matrix() << endl;

    // in order to help you debug, we plot the projected pixels here
    cv::Mat img1_show, img2_show;
    cv::cvtColor(img1, img1_show, CV_GRAY2BGR);
    cv::cvtColor(img2, img2_show, CV_GRAY2BGR);

    //原图画特征点
    for (auto &px: px_ref) {
        cv::rectangle(img1_show, cv::Point2f(px[0] - 2, px[1] - 2), cv::Point2f(px[0] + 2, px[1] + 2),  //左上和右下
                      cv::Scalar(0, 250, 0));
    }


    for(int i=0;i<GoodRefIndex.size();++i)
    {
        auto p_ref = GoodRefIndex[i];
        auto p_cur = goodProjection[i];
        cv::rectangle(img2_show, cv::Point2f(p_cur[0] - 2, p_cur[1] - 2), cv::Point2f(p_cur[0] + 2, p_cur[1] + 2), cv::Scalar(0, 250, 0));
        cv::line(img2_show, cv::Point2f(p_cur[0], p_cur[1]), cv::Point2f(p_ref[0], p_ref[1]), cv::Scalar(0, 250, 0));
    }
    cv::imshow("reference"+order, img1_show);
    cv::imshow("current"+order, img2_show);
    cv::waitKey();
}

void DirectPoseEstimationSingleLayerMT(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,  //随机选择的点
        const vector<double> depth_ref,  //深度参考
        Sophus::SE3d &T21,
        string order
) {
    // parameters
    int half_patch_size = 4;
    int iterations = 100;

    double cost = 0, lastCost = 0;
    int nGood = 0;  // good projections
    VecVector2d goodProjection;
    VecVector2d GoodRefIndex ;
    VecVector2d projection = VecVector2d(px_ref.size(), Eigen::Vector2d(0, 0)); // projected points
    vector<int> ref_index;
    for(int i=0;i<px_ref.size();++i)
        ref_index.push_back(i);
    for (int iter = 0; iter < iterations; iter++) {
        nGood = 0;
        goodProjection.clear();  //每次清零，迭代结束之后就得到了最后的对应点
        GoodRefIndex.clear();

        // Define Hessian and bias
        Matrix6d H = Matrix6d::Zero();  // 6x6 Hessian
        Vector6d b = Vector6d::Zero();  // 6x1 bias

        std::mutex m;
        for_each(execution::par_unseq, ref_index.begin(), ref_index.end(),
                 [&](auto& i)
                 {
                     // compute the projection in the second image
                     // TODO START YOUR CODE HERE
                     Eigen::Vector3d point_ref = depth_ref[i] * Eigen::Vector3d((px_ref[i][0]-cx)/fx, (px_ref[i][1]-cy)/fy, 1);  //ref中的3D点坐标
                     Eigen::Vector3d point_cur = T21 * point_ref;  //ref中的3D点转换到cur中的3D点
                     if (point_cur[2] >= 0)   // depth invalid
                     {
                         float u = fx * point_cur[0]/point_cur[2] + cx, v = fy * point_cur[1]/point_cur[2] + cy;
                         if(u>=half_patch_size && u+half_patch_size<=img2.cols && v>=half_patch_size && v+half_patch_size<=img2.rows)  //变换到cur中若越界则不优化
                         {
                             double X = point_cur[0], Y = point_cur[1], Z = point_cur[2], inv_z = 1.0 / Z, inv_z2 = inv_z * inv_z;  //cur中的3D坐标X'Y'Z'
                             nGood++;
                             std::lock_guard<std::mutex> guard(m);//代替m.lock; m.unlock();
                             //记录投影前后的uv坐标
                             goodProjection.push_back(Eigen::Vector2d(u, v));
                             GoodRefIndex.push_back(Eigen::Vector2d(px_ref[i][0],px_ref[i][1]));

                             // and compute error and jacobian
                             for (int x = -half_patch_size; x < half_patch_size; x++)
                                 for (int y = -half_patch_size; y < half_patch_size; y++)
                                 {
                                     double error =  GetPixelValue(img1, px_ref[i][0]+x, px_ref[i][1]+y) - GetPixelValue(img2, u+x, v+y);

                                     Eigen::Vector2d J_img_pixel;    // image gradients(2*1)  像素梯度,使用cur中的像素坐标和窗口偏移量x，y计算*
                                     J_img_pixel<<(1.0 / 2) * (GetPixelValue(img2, u+1+x, v+y)-GetPixelValue(img2, u-1+x, v+y)),
                                             (1.0 / 2) * (GetPixelValue(img2, u+x, v+1+y)-GetPixelValue(img2, u+x, v-1+y));

                                     Matrix26d J_pixel_xi;   // pixel to \xi in Lie algebra  2*6
                                     J_pixel_xi<<fx * inv_z,
                                             0,
                                             -fx * X * inv_z2,
                                             -fx * X * Y * inv_z2,
                                             fx + fx * X * X * inv_z2,
                                             -fx * Y * inv_z,
                                             0,
                                             fy * inv_z,
                                             -fy * Y * inv_z2,
                                             -fy - fy * Y * Y * inv_z2,
                                             fy * X * Y * inv_z2,
                                             fy * X * inv_z;

                                     // total jacobian   应该是1*6的
                                     Vector6d J=-1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose();

                                     H += J * J.transpose();
                                     b += -error * J;
                                     cost += error * error;
                                 }
                         }
                     }
                     // END YOUR CODE HERE
                 });

        // solve update and put it into estimation
        // TODO START YOUR CODE HERE
        Vector6d update = H.ldlt().solve(b);
        T21 = Sophus::SE3d::exp(update) * T21;  //李群更新
        // END YOUR CODE HERE

        cost /= nGood;

        if (isnan(update[0]))
        {
            // sometimes occurred when we have a black or white patch and H is irreversible
            cout << "update is nan" << endl;
            break;
        }
        if (iter > 0 && cost > lastCost) {
            cout << "cost increased: " << cost << ", " << lastCost << endl;
            break;
        }
        lastCost = cost;
        cout << "cost = " << cost << ", good = " << nGood << endl;
    }
    cout << "good projection: " << nGood << endl;
    cout << "T21 = \n" << T21.matrix() << endl;

    // in order to help you debug, we plot the projected pixels here
    cv::Mat img1_show, img2_show;
    cv::cvtColor(img1, img1_show, CV_GRAY2BGR);
    cv::cvtColor(img2, img2_show, CV_GRAY2BGR);

    //原图画特征点
    for (auto &px: px_ref) {
        cv::rectangle(img1_show, cv::Point2f(px[0] - 2, px[1] - 2), cv::Point2f(px[0] + 2, px[1] + 2),  //左上和右下
                      cv::Scalar(0, 250, 0));
    }

    for(int i=0;i<GoodRefIndex.size();++i)
    {
        auto p_ref = GoodRefIndex[i];
        auto p_cur = goodProjection[i];
        cv::rectangle(img2_show, cv::Point2f(p_cur[0] - 2, p_cur[1] - 2), cv::Point2f(p_cur[0] + 2, p_cur[1] + 2), cv::Scalar(0, 250, 0));
        cv::line(img2_show, cv::Point2f(p_cur[0], p_cur[1]), cv::Point2f(p_ref[0], p_ref[1]), cv::Scalar(0, 250, 0));
    }
    cv::imshow("reference"+order, img1_show);
    cv::imshow("current"+order, img2_show);
//    cv::waitKey();
}

void DirectPoseEstimationMultiLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3d &T21,
        string order
) {

    // parameters  4层2倍金字塔
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    vector<cv::Mat> pyr1, pyr2; // image pyramids
    // TODO START YOUR CODE HERE  构建图像金字塔
    for(int i=0; i<pyramids; i++)
    {
        if(i==0)
        {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        }
        else
        {
            Mat img1_pyr, img2_pyr;
            //自底向上缩放
            cv::resize(pyr1[i-1], img1_pyr, cv::Size(pyr1[i-1].cols * pyramid_scale, pyr1[i-1].rows * pyramid_scale));   //Size(width, height)
            cv::resize(pyr2[i-1], img2_pyr, cv::Size(pyr2[i-1].cols * pyramid_scale, pyr2[i-1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }

    // END YOUR CODE HERE
    //构建特征点金字塔
    double fxG = fx, fyG = fy, cxG = cx, cyG = cy;  // backup the old values
    for (int level = pyramids - 1; level >= 0; level--) {
        VecVector2d px_ref_pyr; // set the keypoints in this pyramid level
        for (auto &px: px_ref) {
            px_ref_pyr.push_back(scales[level] * px);
        }

        // TODO START YOUR CODE HERE
        // scale fx, fy, cx, cy in different pyramid levels
        fx = fxG * scales[level];
        fy = fyG * scales[level];
        cx = cxG * scales[level];
        cy = cyG * scales[level];
        // END YOUR CODE HERE
        DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21, order);
    }
}
