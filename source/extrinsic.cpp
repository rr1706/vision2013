#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching/stitcher.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

//useful functions
Mat setCameraMatrix();
vector<Point3f> getObjectData();

int main()
{
    char c;
    char str[50];

    float max_area= 500000;
    float min_area = 500;

    Point2f center;
    Point2f current_center;

    Mat img, draw, final;

    vector<Point2f> corners(4);

    Mat kern = getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3), cv::Point(-1, -1));

    //  Vision Target Coordinates
    vector<Point3f> objectPoints = getObjectData();

    // Create camera matrix for Kinect with focal length found online
    Mat cameraMatrix = setCameraMatrix();

    //3D Rotation and Translation vectors
    Mat rvec1(3,1, cv::DataType<double>::type);
    Mat tvec1(3,1, cv::DataType<double>::type);

    Mat rvec2(3,1, cv::DataType<double>::type);
    Mat tvec2(3,1, cv::DataType<double>::type);

    cv::Mat distCoeffs(4,1,cv::DataType<double>::type);

    //Initialize to zero
    distCoeffs = Scalar::all(0);
    rvec1 = Scalar::all(0);
    tvec1 = Scalar::all(0);
    rvec2 = Scalar::all(0);
    tvec2 = Scalar::all(0);

    while (1)
    {
        img = imread("./ir_img_2.png", 1);

        draw = img;

        cvtColor(img, img, CV_BGR2GRAY);

        inRange(img, 250, 255, img);
        erode(img, img, kern, Point(-1,-1), 1);
        dilate(img, img, kern, Point(-1,-1), 1);

        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours(img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

        center.y = 999;

        for (size_t i = 0; i < contours.size(); i++)
        {

            if (contourArea(contours[i]) > max_area || contourArea(contours[i]) < min_area)
            {
                continue;
            }

            Moments mu;
            mu = moments(contours[i], false);
            current_center = Point2f(mu.m10/mu.m00, mu.m01/mu.m00);

            if(current_center.y < center.y)
            {
                center.x = current_center.x;
                center.y = current_center.y;

                Rect bound = boundingRect(contours[i]);
                corners[0] = Point(bound.x, bound.y);
                corners[1] = Point(bound.x + bound.width, bound.y);
                corners[2] = Point(bound.x, bound.y + bound.height);
                corners[3] = Point(bound.x + bound.width, bound.y + bound.height);
            }
        }

        line(draw, corners[0], corners[1], Scalar(0, 255, 0), 2);
        line(draw, corners[1], corners[3], Scalar(0, 255, 0), 2);
        line(draw, corners[2], corners[3], Scalar(0, 255, 0), 2);
        line(draw, corners[2], corners[0], Scalar(0, 255, 0), 2);

        solvePnPRansac(objectPoints, corners, cameraMatrix, distCoeffs, rvec1, tvec1);

        // Write outputs on image
        sprintf(str, "Calculated:");
        putText(draw, str, Point(480, 20),  CV_FONT_HERSHEY_SIMPLEX, .75, Scalar(0, 255, 0), 1);

        sprintf(str, "T1 = %.1f", tvec1.at<double>(0,0)/12);
        putText(draw, str, Point(480, 40),  CV_FONT_HERSHEY_SIMPLEX, .75, Scalar(0, 255, 0), 1);

        sprintf(str, "T2 = %.1f", tvec1.at<double>(1,0)/12);
        putText(draw, str, Point(480, 60),  CV_FONT_HERSHEY_SIMPLEX, .75, Scalar(0, 255, 0), 1);

        sprintf(str, "T3 = %.1f", tvec1.at<double>(2,0)/12);
        putText(draw, str, Point(480, 80),  CV_FONT_HERSHEY_SIMPLEX, .75, Scalar(0, 255, 0), 1);

        sprintf(str, "R1 = %.1f", rvec1.at<double>(0,0)*180/CV_PI);
        putText(draw, str, Point(480, 100),  CV_FONT_HERSHEY_SIMPLEX, .75, Scalar(0, 255, 0), 1);

        sprintf(str, "R2 = %.1f", rvec1.at<double>(1,0)*180/CV_PI);
        putText(draw, str, Point(480, 120),  CV_FONT_HERSHEY_SIMPLEX, .75, Scalar(0, 255, 0), 1);

        sprintf(str, "R3 = %.1f", rvec1.at<double>(2,0)*180/CV_PI);
        putText(draw, str, Point(480, 140),  CV_FONT_HERSHEY_SIMPLEX, .75, Scalar(0, 255, 0), 1);

        c = cvWaitKey(10);

        // Exit by pressing "escape key"
        if (c == 27)
            break;

        // Display image
        imshow("Final", draw);
    }
}

vector<Point3f> getObjectData()
{
    vector<Point3f> points;

    points.push_back(Point3f(-31.0, 10.0, 0.0));
    points.push_back(Point3f(31.0, 10.0, 0.0));
    points.push_back(Point3f(-31.0, -10.0, 0.0));
    points.push_back(Point3f(31.0, -10.0, 0.0));

    return points;
}

Mat setCameraMatrix()
{
    Mat matrix(3, 3, CV_32FC1);
    matrix.at<float>(0,0) = 580.0;
    matrix.at<float>(0,1) = 0.0;
    matrix.at<float>(0,2) = 0.0;
    matrix.at<float>(1,0) = 0.0;
    matrix.at<float>(1,1) = 580.0;
    matrix.at<float>(1,2) = 0.0;
    matrix.at<float>(2,0) = 320.0;
    matrix.at<float>(2,1) = 244.0;
    matrix.at<float>(2,2) = 1.0;

    return matrix;
}
