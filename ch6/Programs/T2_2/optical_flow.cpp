#include <opencv2/opencv.hpp>
#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>

#include <execution>  //多线程
#include <mutex>      //加锁

using namespace std;
using namespace cv;

// this program shows how to use optical flow

string file_1 = "./1.png";  // first image
string file_2 = "./2.png";  // second image

// TODO implement this funciton  单层光流
/**
 * single level optical flow
 * @param [in] img1 the first image
 * @param [in] img2 the second image
 * @param [in] kp1 keypoints in img1
 * @param [in|out] kp2 keypoints in img2, if empty, use initial guess in kp1
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse use inverse formulation?
 */
void OpticalFlowSingleLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse = false
);

// TODO implement this funciton  多层光流
/**
 * multi level optical flow, scale of pyramid is set to 2 by default
 * the image pyramid will be create inside the function
 * @param [in] img1 the first pyramid
 * @param [in] img2 the second pyramid
 * @param [in] kp1 keypoints in img1
 * @param [out] kp2 keypoints in img2
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse set true to enable inverse formulation
 */
void OpticalFlowMultiLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse = false
);

//多线程金字塔
void OpticalFlowMultiLevelMT(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse);

//TODO 多线程光流
void OpticalFlowSingleLevelMT(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse
);

/**
 * get a gray scale value from reference image (bi-linear interpolated)  双线性差值方法来获得图像的像素值
 * @param img
 * @param x
 * @param y
 * @return
 */
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


int main(int argc, char **argv) {

    // images, note they are CV_8UC1, not CV_8UC3
    Mat img1 = imread(file_1, 0);
    Mat img2 = imread(file_2, 0);

    // key points, using GFTT here.  提取GFTT角点
    vector<KeyPoint> kp1;
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints
    detector->detect(img1, kp1);

    // now lets track these key points in the second image
    // first use single level LK in the validation picture
    vector<KeyPoint> kp2_single;  //存的是特征点的坐标
    vector<bool> success_single;
//    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single, false);
    evaluate_and_call([&]() { OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single, false); }, "optical flow by SingleLevel_Forward", 1);
//    OpticalFlowSingleLevelMT(img1, img2, kp1, kp2_single, success_single, false);
//    evaluate_and_call([&]() { OpticalFlowSingleLevelMT(img1, img2, kp1, kp2_single, success_single, false); }, "optical flow by SingleLevelMT_Forward", 1);


    // then test multi-level LK
    vector<KeyPoint> kp2_multi,kp2_multi_test, kp2_multi_test_MT;
    vector<bool> success_multi,success_multi_test, success_multi_MT;
//    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi_test, success_multi_test, false);
    evaluate_and_call([&]() { OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi, false); }, "optical flow by MultiLevel_Forward", 1);
//    evaluate_and_call([&]() { OpticalFlowMultiLevelMT(img1, img2, kp1, kp2_multi_test_MT, success_multi_MT, false); }, "optical flow by MultiLevelMT_Forward", 1);


    // use opencv's flow for validation
    /*vector<Point2f> pt1, pt2;
    for (auto &kp: kp1) pt1.push_back(kp.pt);
    vector<uchar> status;
    vector<float> error;
    cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error, cv::Size(8, 8), 3);
    evaluate_and_call([&]() {cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error, cv::Size(8, 8), 10);}, "optical flow by OpenCV", 1);
     */

    /*vector<Point2f> pt1_16, pt2_16;
    for (auto &kp: kp1) pt1_16.push_back(kp.pt);
    vector<uchar> status_16;
    vector<float> error_16;
    evaluate_and_call([&]() {cv::calcOpticalFlowPyrLK(img1, img2, pt1_16, pt2_16, status_16, error_16, cv::Size(16,16));}, "optical flow by OpenCV", 1);*/

    // plot the differences of those functions   在原图上画出关键点和线
    Mat img2_single;
    cv::cvtColor(img2, img2_single, CV_GRAY2BGR);
    for (int i = 0; i < kp2_single.size(); i++)
    {
        if (success_single[i])  //如果追踪成功就画点
        {
            cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single, kp1[i].pt, kp2_single[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    Mat img2_multi;
    cv::cvtColor(img2, img2_multi, CV_GRAY2BGR);
    for (int i = 0; i < kp2_multi.size(); i++) {
        if (success_multi[i]) {
            cv::circle(img2_multi, kp2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_multi, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    //8
    /*Mat img2_CV;
    cv::cvtColor(img2, img2_CV, CV_GRAY2BGR);
    for (int i = 0; i < pt2.size(); i++) {
        if (status[i]) {
            cv::circle(img2_CV, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_CV, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
        }
    }*/

//    cv::imshow("forward tracked single level", img2_single);

    cv::imshow("forward tracked multi level", img2_multi);
    cv::imwrite("tracked_by_multi_0.15scale.png",img2_multi);

//    cv::imshow("tracked_by_opencv_8x8_layer5", img2_CV);
//    cv::imwrite("tracked_by_opencv_8x8_layer5.png",img2_CV);

    cv::waitKey(0);
    return 0;
}

void OpticalFlowSingleLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse
)
{
    // parameters
    int half_patch_size = 4;  //选取8*8的窗口
    int iterations = 10;
    bool have_initial = !kp2.empty();

    for (size_t i = 0; i < kp1.size(); i++) {
        auto kp = kp1[i];
        double dx = 0, dy = 0; // dx,dy need to be estimated  dx和dy是需要被估计的
        if (have_initial)   //如果已经初始化，那么减去kp1中的值（为什么？）
        {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastCost = 0;
        bool succ = true; // indicate if this point succeeded

        // Gauss-Newton iterations
        for (int iter = 0; iter < iterations; iter++)
        {
            Eigen::Matrix2d H = Eigen::Matrix2d::Zero();  //Hessian
            Eigen::Vector2d b = Eigen::Vector2d::Zero();  //bias
            cost = 0;

            if (kp.pt.x + dx <= half_patch_size || kp.pt.x + dx >= img1.cols - half_patch_size ||
                kp.pt.y + dy <= half_patch_size || kp.pt.y + dy >= img1.rows - half_patch_size) {   // go outside
                succ = false;   // 如果估计出的移动量超过了图像边界，放弃此次迭代
                break;
            }
            // compute cost and jacobian  在(2*half_patch_size,2*half_patch_size)窗口内计算cost和雅可比
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++)
                {
                    // TODO START YOUR CODE HERE (~8 lines)
                    //计算误差
                    double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y)-GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);
                    Eigen::Vector2d J;  // Jacobian
                    if (inverse == false)
                    {
                        // Forward Jacobian  前向雅可比（因为是离散的，不能用微分，使用中心差分方式来进行求导）
                        J = -1.0 * Eigen::Vector2d(
                                0.5 * (GetPixelValue(img2,kp.pt.x + dx + x + 1, kp.pt.y + dy + y)-GetPixelValue(img2, kp.pt.x + dx + x -1, kp.pt.y + dy + y)),
                                0.5 * (GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y+ dy + y + 1)- GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y -1))
                                );
                    }
                    else
                    {
                        if(iter == 0 )
                        // Inverse Jacobian
                        // NOTE this J does not change when dx, dy is updated, so we can store it and only compute error
                        //反向模式，使用I1处的梯度替换I2处的梯度，I1没有平移，无dx,dy
                        {
                            J = -1.0 * Eigen::Vector2d(
                                    0.5 * (GetPixelValue(img2,kp.pt.x + x + 1, kp.pt.y + y)-
                                           GetPixelValue(img2, kp.pt.x +  x -1, kp.pt.y + y)),
                                    0.5 * (GetPixelValue(img2, kp.pt.x +  x, kp.pt.y + y + 1)-
                                           GetPixelValue(img2, kp.pt.x +  x, kp.pt.y + y -1))
                                    );
                        }
                    }
                    // compute H, b and set cost;
                    b += -error * J;
                    cost += error * error;
                    if(inverse==false || iter==0)  //如果是正向或者是第一次迭代，就需要更新系数矩阵H
                    {
                        H +=  J * J.transpose() ;  //这里的雅可比定义出来是J^T，直接就是向量，求H要得是矩阵，所以得J*J^T
                    }
                    // TODO END YOUR CODE HERE
                }
            // compute update  更新
            // TODO START YOUR CODE HERE (~1 lines)
            Eigen::Vector2d update = H.ldlt().solve(b);  //求解方程H[dx,dy]^T=b
            // TODO END YOUR CODE HERE

            if (isnan(update[0])) {
                // sometimes occurred when we have a black or white patch and H is irreversible
                cout << "update is nan" << endl;
                succ = false;
                break;
            }
            if (iter > 0 && cost > lastCost) {
//                cout << "cost increased: " << cost << ", " << lastCost << endl;
                break;
            }

            // update dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;

            if(update.norm()<1e-2)  //coverage
                break;

        }  //迭代结束
        success.push_back(succ);

        // set kp2
        if (have_initial) {
            kp2[i].pt = kp.pt + Point2f(dx, dy);
        } else {
            KeyPoint tracked = kp;
            tracked.pt += cv::Point2f(dx, dy);
            kp2.push_back(tracked);
        }
    }
}

void OpticalFlowMultiLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse) {

    // parameters
    int pyramids = 4;  //4层金字塔
    double pyramid_scale = 0.15;  //缩放率为0.5
    double scales[] = {pow(pyramid_scale, 0),
                       pow(pyramid_scale, 1),
                       pow(pyramid_scale, 2),
                       pow(pyramid_scale, 3),
                       pow(pyramid_scale, 4),
                       pow(pyramid_scale, 5),
                       pow(pyramid_scale, 6),
                       pow(pyramid_scale, 7),
                       pow(pyramid_scale, 8)};  //每一层相当于底层的缩放系数

    // create pyramids
    vector<Mat> pyr1, pyr2; // image pyramids
    // TODO START YOUR CODE HERE (~8 lines)
    for (int i = 0; i < pyramids; i++) {
        if(i==0)
        {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        }
        else
        {
            Mat img1_pyr, img2_pyr;
            //自底向上缩放
            cv::resize(pyr1[i-1], img1_pyr, cv::Size(pyr1[i-1].cols * pyramid_scale, pyr1[i-1].rows * pyramid_scale));
            cv::resize(pyr2[i-1], img2_pyr, cv::Size(pyr2[i-1].cols * pyramid_scale, pyr2[i-1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }
    // TODO END YOUR CODE HERE

    // coarse-to-fine LK tracking in pyramids
    // TODO START YOUR CODE HERE
    vector<KeyPoint> kp1_pyr, kp2_pyr;  //特征点金字塔

    //顶层
    for(auto &kp:kp1)
    {
        auto kp_top = kp;
        kp_top.pt *= scales[pyramids - 1];  //底层到顶层总共缩放到了0.125倍，求得顶层的特征点，初始化为下一层的特征点
        kp1_pyr.push_back(kp_top);
        kp2_pyr.push_back(kp_top);
    }

    //coarse to fine计算光流，从顶层图像开始计算，把上一层的追踪结果作为下一层光流的初始值
    for(int level = pyramids-1; level>=0; level--)
    {
        success.clear();
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse);
        if(level>0)
        {
            for(auto &kp: kp1_pyr)  //引用，改变源数据，自顶向下扩大
                kp.pt /= pyramid_scale;
            for(auto &kp: kp2_pyr)
                kp.pt /= pyramid_scale;
        }
    }

    // don't forget to set the results into kp2
    // don't forget to set the results into kp2
    for(auto &kp: kp2_pyr)
        kp2.push_back(kp);
    // TODO END YOUR CODE HERE
}


void OpticalFlowMultiLevelMT(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse) {

    // parameters
    int pyramids = 4;  //4层金字塔
    double pyramid_scale = 0.5;  //缩放率为0.5
    double scales[] = {1.0, 0.5, 0.25, 0.125};  //每一层相当于底层的缩放系数

    // create pyramids
    vector<Mat> pyr1, pyr2; // image pyramids
    // TODO START YOUR CODE HERE (~8 lines)
    for (int i = 0; i < pyramids; i++) {
        if(i==0)
        {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        }
        else
        {
            Mat img1_pyr, img2_pyr;
            //自底向上缩放
            cv::resize(pyr1[i-1], img1_pyr, cv::Size(pyr1[i-1].cols * pyramid_scale, pyr1[i-1].rows * pyramid_scale));
            cv::resize(pyr2[i-1], img2_pyr, cv::Size(pyr2[i-1].cols * pyramid_scale, pyr2[i-1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }
    // TODO END YOUR CODE HERE

    // coarse-to-fine LK tracking in pyramids
    // TODO START YOUR CODE HERE
    vector<KeyPoint> kp1_pyr, kp2_pyr;  //特征点金字塔

    //顶层
    for(auto &kp:kp1)
    {
        auto kp_top = kp;
        kp_top.pt *= scales[pyramids - 1];  //底层到顶层总共缩放到了0.125倍，求得顶层的特征点，初始化为下一层的特征点
        kp1_pyr.push_back(kp_top);
        kp2_pyr.push_back(kp_top);
    }

    //coarse to fine计算光流，从顶层图像开始计算，把上一层的追踪结果作为下一层光流的初始值
    for(int level = pyramids-1; level>=0; level--)
    {
        success.clear();
        OpticalFlowSingleLevelMT(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse);
        if(level>0)
        {
            for(auto &kp: kp1_pyr)  //引用，改变源数据，自顶向下扩大
                kp.pt /= pyramid_scale;
            for(auto &kp: kp2_pyr)
                kp.pt /= pyramid_scale;
        }
    }

    // don't forget to set the results into kp2
    for(auto &kp: kp2_pyr)
        kp2.push_back(kp);
    // TODO END YOUR CODE HERE
}


void OpticalFlowSingleLevelMT(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse
)
{
    // parameters
    int half_patch_size = 4;  //选取8*8的窗口
    int iterations = 10;
    bool have_initial = !kp2.empty();

    vector<int> indexes;
    for (int (i) = 0; (i) < kp1.size(); ++(i))
        indexes.push_back(i);

    std::mutex m;
    std::lock_guard<std::mutex> guard(m);//代替m.lock; m.unlock();

    for_each(execution::par_unseq, indexes.begin(), indexes.end(),
                  [&](auto& i)
                  {
                      auto kp = kp1[i];
                      double dx = 0, dy = 0; // dx,dy need to be estimated  dx和dy是需要被估计的
                      if (have_initial)   //如果已经初始化，那么减去kp1中的值（为什么？）
                      {
                          dx = kp2[i].pt.x - kp.pt.x;
                          dy = kp2[i].pt.y - kp.pt.y;
                      }

                      double cost = 0, lastCost = 0;
                      bool succ = true; // indicate if this point succeeded

                      // Gauss-Newton iterations
                      for (int iter = 0; iter < iterations; iter++)
                      {
                          Eigen::Matrix2d H = Eigen::Matrix2d::Zero();  //Hessian
                          Eigen::Vector2d b = Eigen::Vector2d::Zero();  //bias
                          cost = 0;

                          if (kp.pt.x + dx <= half_patch_size || kp.pt.x + dx >= img1.cols - half_patch_size ||
                              kp.pt.y + dy <= half_patch_size || kp.pt.y + dy >= img1.rows - half_patch_size) {   // go outside
                              succ = false;   // 如果估计出的移动量超过了图像边界，放弃此次迭代
                              break;
                          }
                          // compute cost and jacobian  在(2*half_patch_size,2*half_patch_size)窗口内计算cost和雅可比
                          for (int x = -half_patch_size; x < half_patch_size; x++)
                              for (int y = -half_patch_size; y < half_patch_size; y++)
                              {
                                  // TODO START YOUR CODE HERE (~8 lines)
                                  //计算误差
                                  double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y)-GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);
                                  Eigen::Vector2d J;  // Jacobian
                                  if (inverse == false)
                                  {
                                      // Forward Jacobian  前向雅可比（因为是离散的，不能用微分，使用中心差分方式来进行求导）
                                      J = -1.0 * Eigen::Vector2d(
                                              0.5 * (GetPixelValue(img2,kp.pt.x + dx + x + 1, kp.pt.y + dy + y)-GetPixelValue(img2, kp.pt.x + dx + x -1, kp.pt.y + dy + y)),
                                              0.5 * (GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y+ dy + y + 1)- GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y -1))
                                      );
                                  }
                                  else
                                  {
                                      if(iter == 0 )
                                          // Inverse Jacobian
                                          // NOTE this J does not change when dx, dy is updated, so we can store it and only compute error
                                          //反向模式，使用I1处的梯度替换I2处的梯度，I1没有平移，无dx,dy
                                      {
                                          J = -1.0 * Eigen::Vector2d(
                                                  0.5 * (GetPixelValue(img2,kp.pt.x + x + 1, kp.pt.y + y)-
                                                         GetPixelValue(img2, kp.pt.x +  x -1, kp.pt.y + y)),
                                                  0.5 * (GetPixelValue(img2, kp.pt.x +  x, kp.pt.y + y + 1)-
                                                         GetPixelValue(img2, kp.pt.x +  x, kp.pt.y + y -1))
                                          );
                                      }
                                  }
                                  // compute H, b and set cost;
                                  b += -error * J;
                                  cost += error * error;
                                  if(inverse==false || iter==0)  //如果是正向或者是第一次迭代，就需要更新系数矩阵H
                                  {
                                      H +=  J * J.transpose() ;  //这里的雅可比定义出来是J^T，直接就是向量，求H要得是矩阵，所以得J*J^T
                                  }
                                  // TODO END YOUR CODE HERE
                              }
                          // compute update  更新
                          // TODO START YOUR CODE HERE (~1 lines)
                          Eigen::Vector2d update = H.ldlt().solve(b);  //求解方程H[dx,dy]^T=b
                          // TODO END YOUR CODE HERE

                          if (isnan(update[0])) {
                              // sometimes occurred when we have a black or white patch and H is irreversible
                              cout << "update is nan" << endl;
                              succ = false;
                              break;
                          }
                          if (iter > 0 && cost > lastCost) {
//                cout << "cost increased: " << cost << ", " << lastCost << endl;
                              break;
                          }

                          // update dx, dy
                          dx += update[0];
                          dy += update[1];
                          lastCost = cost;
                          succ = true;

                          if(update.norm()<1e-2)  //coverage
                              break;

                      }  //迭代结束
                      success.push_back(succ);

                      // set kp2
                      if (have_initial) {
                          kp2[i].pt = kp.pt + Point2f(dx, dy);
                      } else {
                          KeyPoint tracked = kp;
                          tracked.pt += cv::Point2f(dx, dy);
                          kp2.push_back(tracked);
                      }
                  }
    );
}
