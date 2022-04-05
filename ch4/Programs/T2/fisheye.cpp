//
// Created by xiang on 2021/9/9.
//

#include <opencv2/opencv.hpp>

// 文件路径，如果不对，请调整
std::string input_file = "../fisheye.jpg";

int main(int argc, char **argv) {
    // 本程序实现鱼眼的等距投影去畸变模型
    // 畸变参数（本例设为零）
    double k1 = 0, k2 = 0, k3 = 0, k4 = 0;

    // 内参
    double fx = 689.21, fy = 690.48, cx = 1295.56, cy = 942.17;

    cv::Mat image = cv::imread(input_file);
    int rows = image.rows, cols = image.cols;
    cv::Mat image_undistort = cv::Mat(rows, cols, CV_8UC3); // 去畸变以后的图

    cv::Mat before_image;
    cv::resize(image,before_image,cv::Size(1280,720)); //Size(1280,720)缩放后的图片尺寸
    cv::imshow("image distorted", before_image);

    // 计算去畸变后图像的内容
    for (int v = 0; v < rows; v++)
        for (int u = 0; u < cols; u++) {

            // TODO 按照公式，计算点(u,v)对应到畸变图像中的坐标(u_distorted, v_distorted) (~6 lines)

            // start your code here
            //变到相机坐标系
            double a = (u - cx) / fx, b = (v - cy) / fy;
            double r = sqrt(a * a + b * b);
            double theta = atan(r);
            double theta_d = theta * (1 + k1 * pow(theta, 2) + k2 * pow(theta, 4) + k3 * pow(theta, 6) + k4 * pow(theta, 8));
            double scale = theta_d / r;
            double x_distorted = scale * a;
            double y_distorted = scale * b;
            //去畸变完之后再变换到像素
            double u_distorted = fx * x_distorted + cx;
            double v_distorted = fy * y_distorted + cy;
            // end your code here

            // 赋值 (最近邻插值)
            if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < cols &&
                v_distorted < rows) {
                image_undistort.at<cv::Vec3b>(v, u) =
                        image.at<cv::Vec3b>((int)v_distorted, (int)u_distorted);
            } else {
                image_undistort.at<cv::Vec3b>(v, u) = 0;
            }
        }

    // 画图去畸变后图像
    cv::Mat after_image;
    cv::resize(image_undistort,after_image,cv::Size(1280,720)); //Size(1280,720)缩放后的图片尺寸
    cv::imshow("image undistorted", after_image);
    cv::imwrite("fisheye_undist.jpg", image_undistort);
    cv::waitKey();

    return 0;
}