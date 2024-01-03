#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    Mat src = imread("location photo");
    if (src.empty())
        return -1;

    imshow("src", src);

    // Tạo ảnh nhị phân từ ảnh nguồn
    Mat bw;
    cvtColor(src, bw, COLOR_BGR2GRAY);
    threshold(bw, bw, 40, 255, THRESH_BINARY);
    imshow("bw", bw);

    // Thực hiện thuật toán biến đổi khoảng cách
    Mat dist;
    distanceTransform(bw, dist, DIST_L2, 3);

    // Chuẩn hóa ảnh khoảng cách để phạm vi = {0.0, 1.0}
    // để chúng ta có thể hình dung và ngưỡng nó
    normalize(dist, dist, 0, 1., NORM_MINMAX);
    imshow("dist", dist);

    // Ngưỡng để thu được điểm cao điểm
    // Điều này sẽ là đánh dấu cho các đối tượng nền
    threshold(dist, dist, .5, 1., THRESH_BINARY);
    imshow("dist2", dist);

    // Chuyển đổi ảnh khoảng cách thành phiên bản CV_8U
    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);

    // Tìm các đường viền để xác định đánh dấu
    vector<vector<Point>> contours;
    findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Tạo ảnh đánh dấu cho thuật toán watershed
    Mat markers = Mat::zeros(dist.size(), CV_32SC1);

    // Vẽ đánh dấu trên ảnh
    for (size_t i = 0; i < contours.size(); i++) {
        drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i) + 1), -1);
    }

    // Áp dụng thuật toán watershed
    watershed(src, markers);

    // Tạo ảnh kết quả
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);

    // Áp dụng màu sắc vào các đối tượng đã phân đoạn
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            int index = markers.at<int>(i, j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
                dst.at<Vec3b>(i, j) = src.at<Vec3b>(i, j);
            else
                dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
        }
    }

    imshow("dst", dst);
    waitKey(0);
    return 0;
}
