/*Alliance Wall
  v3.2 (post internationals 2013)
  Hunter Park
  Take image from Kinect, process it
  Tracks Squares
  Computes distance and rotation
  Distinguish between 3pt and 2pt targets
  Region of Interest
  Upper limit
  Load Image capability
*/

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include "libfreenect_cv.h"
#include <time.h>
#include "timer.h"
#include <math.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <QtNetwork>

using namespace std;

#define IR_IMAGE_WIDTH      640
#define IR_IMAGE_HEIGHT     488
#define Target_Height       110.125    //inches
#define Bracket_Distance    462//389.5   inches
#define Bracket_Height      31.6875    //inches
#define Sensor_Height       0.6125     //inches
#define Sensor_Distance     1.5        //inches
#define PNG_TIME            5000
#define Ratio_Test          2.4
#define ROI_HEIGHT          300


#define SAVE_PNG    0			// Save IR image to file
#define DO_UDP      0			// Send UDP messages to cRIO
#define WRITE_LOG   0		    // Write vision solution to log file
#define SHOW_IMAGE  1           // Show image
#define LOAD_FILE   0    		// Load images from file instead of from Kinect

CvPoint pt[4];
CvPoint targetpt[4];
CvPoint2D32f Target;

char str[50];
char c;

char dirname[100];
char pngname[100];
char logname[100];

int minContourArea = 1000;     //subject to change

IplImage* ir_img = cvCreateImage( cvSize( IR_IMAGE_WIDTH, IR_IMAGE_HEIGHT ), IPL_DEPTH_8U, 1);
IplImage* img = cvCreateImage( cvSize( IR_IMAGE_WIDTH, IR_IMAGE_HEIGHT ), IPL_DEPTH_8U, 1);
IplImage* out = cvCreateImage(cvSize(IR_IMAGE_WIDTH, IR_IMAGE_HEIGHT), IPL_DEPTH_8U, 1);

#if SHOW_IMAGE
IplImage* rgb_img = cvCreateImage(cvSize(IR_IMAGE_WIDTH, IR_IMAGE_HEIGHT), IPL_DEPTH_8U, 3);
#endif

CvMemStorage* storage = cvCreateMemStorage(0);
CvMemStorage* Kernel = cvCreateMemStorage(0);

CvFont font;

float length(float x1, float y1, float x2, float y2)
{
    return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
}

void T2B_L2R(CvPoint* pt)
{
    int temp_x;
    int temp_y;

    int i, swapped;

    do {
        swapped = 0;
        for (i = 1; i < 4; i++)
        {
            if (pt[i-1].y > pt[i].y)
            {
                temp_x = pt[i-1].x;
                temp_y = pt[i-1].y;
                pt[i-1].x = pt[i].x;
                pt[i-1].y = pt[i].y;
                pt[i].x = temp_x;
                pt[i].y = temp_y;
                swapped = 1;
            }
        }
    }
    while (swapped == 1);

    // Make sure top two points are left to right
    if (pt[0].x > pt[1].x)
    {
        temp_x = pt[0].x;
        temp_y = pt[0].y;
        pt[0].x = pt[1].x;
        pt[0].y = pt[1].y;
        pt[1].x = temp_x;
        pt[1].y = temp_y;
    }

    // Make sure bottom two points are left to right
    if (pt[2].x > pt[3].x)
    {
        temp_x = pt[2].x;
        temp_y = pt[2].y;
        pt[2].x = pt[3].x;
        pt[2].y = pt[3].y;
        pt[3].x = temp_x;
        pt[3].y = temp_y;
    }
}

CvSeq* findSquares4(IplImage* out, float* LargestRatio, int* TargetType)
{
    float PreviousBiggestRatio = 0;
    float FinalX;
    float FinalY;
    double Ratio;
    double RatioXT;
    double RatioXB;
    double RatioYR;
    double RatioYL;
    CvSeq* contours;
    CvSeq* result;
    CvSeq* Square = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvPoint), storage);
    Target.x = 0.0;
    Target.y = 0.0;

    cvFindContours(out, storage, &contours, sizeof(CvContour),
                   CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

    // test each contour
    while(contours)
    {
        result = cvApproxPoly(contours, sizeof(CvContour), storage,
                              CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0);

        if(result->total == 4 &&
                fabs(cvContourArea(result,CV_WHOLE_SEQ)) > minContourArea &&
                cvCheckContourConvexity(result))
        {

            if (contours->v_next == NULL)       //no interier contour
            {
                contours = contours->h_next;   //go to next tree
                continue;
            }

            CvMoments *moments = (CvMoments*)malloc(sizeof(CvMoments));
            cvMoments (contours, moments, 1);

            double moment10 = cvGetSpatialMoment(moments, 1, 0);
            double moment01 = cvGetSpatialMoment(moments, 0, 1);
            double area = cvGetSpatialMoment(moments, 0, 0);

            double posX = 0;
            double posY = 0;

            if (area > 0)
            {
                posX = moment10/area;
                posY = moment01/area;
            }
            else
            {
                posX = 0.0;
                posY = 0.0;
            }

            CvSeqReader reader;

            //initialize reader of the sequence
            cvStartReadSeq(result, &reader, 0);

            memcpy(pt, reader.ptr, result->elem_size);
            CV_NEXT_SEQ_ELEM(result->elem_size, reader);
            memcpy(pt + 1, reader.ptr, result->elem_size);
            CV_NEXT_SEQ_ELEM(result->elem_size, reader);
            memcpy(pt + 2, reader.ptr, result->elem_size);
            CV_NEXT_SEQ_ELEM(result->elem_size, reader);
            memcpy(pt + 3, reader.ptr, result->elem_size);
            CV_NEXT_SEQ_ELEM(result->elem_size, reader);

            T2B_L2R(pt);

            CvPoint2D32f Center;
            Center.x = (posX - IR_IMAGE_WIDTH/2);
            Center.y = -(posY - IR_IMAGE_HEIGHT/2);

            if (Center.y > 160)
            {
                contours = contours->h_next;   //go to next tree
                continue;
            }

            RatioXT = length(pt[0].x, pt[0].y, pt[1].x, pt[1].y);
            RatioXB = length(pt[2].x, pt[2].y, pt[3].x, pt[3].y);
            RatioYL = length(pt[0].x, pt[0].y, pt[2].x, pt[2].y);
            RatioYR = length(pt[1].x, pt[1].y, pt[3].x, pt[3].y);

            if ( (RatioYL + RatioYR) < 10.0)
            {
                Ratio = 0;
            }
            else
            {
                Ratio = (RatioXT + RatioXB)/(RatioYL + RatioYR);
            }
            if ( Ratio > 4.2 )           //not a contour of interest
            {
                contours = contours->h_next;
            }

            if ( Ratio > PreviousBiggestRatio)
            {
                PreviousBiggestRatio = Ratio;
                *LargestRatio = PreviousBiggestRatio;
                Target = Center;
                Square = result;
                FinalX = posX;
                FinalY = posY;
                targetpt[0] = pt[0];
                targetpt[1] = pt[1];
                targetpt[2] = pt[2];
                targetpt[3] = pt[3];
                if (*LargestRatio > Ratio_Test)
                {
                    *TargetType = 3;
                }
                else
                {
                    *TargetType = 2;
                }
            }
        }

        // Take next tree
        contours = contours->h_next;
    }

#if SHOW_IMAGE

    if (*TargetType == 3)
    {
        cvCircle(rgb_img, cvPoint(FinalX, FinalY), 3, cvScalar(255, 0, 0), 1, 8, 0);
        cvDrawContours(rgb_img, Square, cvScalar(255, 0, 0), cvScalar(255, 0, 0), 0, 2, CV_AA);
    }
    else
    {
        cvCircle(rgb_img, cvPoint(FinalX, FinalY), 3, cvScalar(0, 0, 255), 1, 8, 0);
        cvDrawContours(rgb_img, Square, cvScalar(0, 0, 255), cvScalar(0, 0, 255), 0, 2, CV_AA);
    }

    sprintf(str, "Closest Center = (%.2f, %.2f)", Target.x, (Target.y));
    cvPutText(rgb_img, str, cvPoint(10, 20), &font, cvScalar(255,0,255));

    sprintf(str, "Ratio = %.1f", Ratio);
    cvPutText(rgb_img, str, cvPoint(10, 40), &font, cvScalar(255,0,255));
#endif

    cvClearMemStorage(storage);

    return Square;
}

int main(int argc, char *argv[])
{
#if SHOW_IMAGE
    cvNamedWindow("Threshold", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Finished", CV_WINDOW_AUTOSIZE);

    cvMoveWindow("Threshold", 750, 10);
    cvMoveWindow("Finished", 10, 10);

    cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX_SMALL, 0.75, 0.75, 0, 1, CV_AA);
#endif

    int png_count = 0;
    int imageSaved = 0;
    int imageNum = 0;
    int hadIR = 0;
    int imgNum = 20;
    int lastNum = 0;
    int TargetType;

    char dirname[100];
    char pngname[100];
    char logname[100];

    double frame_time_ms;
    double ave_frame_time_ms;
    double do_png_time;

    float Xrot;
    float Yrot;
    float distance;
    float LargestRatio;

    float DeltaHeight = Target_Height - (Bracket_Height + Sensor_Height);
    float Kinect_Distance = Bracket_Distance - Sensor_Distance;
    float Kinect_Angle = (atan2(DeltaHeight, Kinect_Distance)*180/CV_PI);

    double systime = 0;
    static double startTick = cvGetTickCount();
    double cpufreq = cvGetTickFrequency();

    ofstream logfile;

    IplConvKernel* Verticle;

    int Value [] = {0, 1, 0,
                    0, 1, 0,
                    0, 1, 0};

    QUdpSocket udpSocket;

    if (SAVE_PNG || WRITE_LOG) {
        time_t rawtime;
        struct tm* timeinfo;
        int len;

        time(&rawtime);
        timeinfo = localtime(&rawtime);
        len = strftime(dirname,50,"%Y%m%d_%H%M%S",timeinfo);
        if (len > 0) {
            printf("Error creating directory name");
        }

        mkdir(dirname, 0777);

        if (WRITE_LOG) {
            sprintf(logname, "./%s/logfile.csv",dirname);
            logfile.open(logname);
            logfile << "systime_sec" << ", "                    // Incremental system time since start (sec)
                    << "distance_ft" << ", "                    // Distance to target (feet)
                    << "Xrot_deg" << ", "                       // X rotation (deg)
                    << "TargetType" << ", "                     // TargetType - 0 = none, 2 = 2pt, 3 = 3pt
                    << "UL_corner_x" << ", "                    // Upper left corner - x-coordinate (pixels)
                    << "UL_corner_y" << ", "                    // Upper left corner - y-coordinate (pixels)
                    << "UR_corner_x" << ", "                    // Upper right corner - x-coordinate (pixels)
                    << "UR_corner_y" << ", "                    // Upper right corner - y-coordinate (pixels)
                    << "LL_corner_x" << ", "                    // Lower left corner - x-coordinate (pixels)
                    << "LL_corner_y" << ", "                    // Lower left corner - y-coordinate (pixels)
                    << "LR_corner_x" << ", "                    // Lower right corner - x-coordinate (pixels)
                    << "LR_corner_y" << ", "                    // Lower right corner - y-coordinate (pixels)
                    << "Target_x" << ", "                       // Center of target - x-coordinate relative to center of screen (pixels)
                    << "Target_y" << ", "                       // Center of target - y-coordinate relative to center of screen (pixels)
                    << "AspectRatio" << ", "                    // Aspect ratio of taret
                    << "imageSaved" << ", "                     // Image saved logical
                    << "imageNum" << endl;                      // Image number
            logfile.close();
        }
    }

    DECLARE_TIMING(IR_Timer);
    START_TIMING(IR_Timer);

    while ( 1 )
    {
        //reinitialize values to 0
        TargetType = 0;
        distance = 0.0;
        Yrot = 0.0;
        Xrot = 0.0;
        LargestRatio = 0.0;
        Target.x = 0.0;
        Target.y = 0.0;

        if (LOAD_FILE)
        {
            if (imgNum < 1)
            {
                imgNum = 1;
            }

            if (lastNum != imgNum)
            {
                sprintf(pngname, "./20130225_221543/ir_img_%d.png",imgNum);

                ir_img = cvLoadImage(pngname, .5);
                lastNum = imgNum;
            }
        } else {
            ir_img = freenect_sync_get_ir_cv(0);
        }

        if (!ir_img) {
            printf("Error grabbing IR video.");

            // Shut down vision processor by unplugging Kinect cable
            if (hadIR == 1) {
                if (LOAD_FILE) {
                    imgNum = 1;
                }

                while(1) {
                    system("sudo shutdown -h now");
                    cvWaitKey(2000);
                }
            } else {    // Vision processor was booted without Kinect plugged in so quit program
                break;
            }
        } else {
            hadIR = 1;
        }

#if LOAD_FILE
        // Write image name to output image
        cvPutText(rgb_img, pngname, cvPoint(20, 20), &font, cvScalar(0, 255, 0));
#endif

        // Check for key press
        if (LOAD_FILE)
        {
            c = cvWaitKey(-1);
        }
        else
        {
            c = cvWaitKey(10);
        }

        if (c == 32 || c == 84)            // Switch to next PNG by pressing space bar
        {
            imgNum ++;
        } else if (c == 0x31)              //1 key pressed
        {
            imgNum --;
        }

        // Calculate elapsed system time (sec)
        if (cpufreq > 0) {
            systime = (cvGetTickCount() - startTick) / (cpufreq * 1000000.0);
        }


        // Set ir_img ROI
        CvRect ROI_rect = cvRect(1, 1, IR_IMAGE_WIDTH, ROI_HEIGHT);
        cvSetImageROI(ir_img, ROI_rect);
        cvSetImageROI(out, ROI_rect);

        //Isolate the vision targets
        cvThreshold(ir_img, out, 80, 256, CV_THRESH_BINARY);

        cvResetImageROI(ir_img);
        cvResetImageROI(out);

        Verticle = cvCreateStructuringElementEx(3, 3, 1, 1, CV_SHAPE_CUSTOM, Value);

        cvErode(out, out, Verticle, 1);
        cvDilate(out, out, Verticle, 1);

#if SHOW_IMAGE
        cvShowImage( "Threshold", out );
        cvCvtColor(ir_img, rgb_img, CV_GRAY2BGR);
#endif

        //copy to find squares with
        cvCopyImage(out, img);

        findSquares4(img, &LargestRatio, &TargetType);

#if SHOW_IMAGE
        //draw crosshairs
        cvLine(rgb_img, cvPoint(IR_IMAGE_WIDTH/2, 0), cvPoint(IR_IMAGE_WIDTH/2, IR_IMAGE_HEIGHT), cvScalar(0,255,255), 1);
        cvLine(rgb_img, cvPoint(0, IR_IMAGE_HEIGHT/2), cvPoint(IR_IMAGE_WIDTH, IR_IMAGE_HEIGHT/2), cvScalar(0,255,255), 1);
#endif

        if (TargetType == 0)
        {
            Xrot = 0.0;
            Yrot = 0.0;
            distance = 0.0;
        }
        else
        {
            Xrot = Target.x/(IR_IMAGE_WIDTH/2)*29;         //kinect fov: 58 degrees horizontal pixels:640
            Yrot = Target.y/(IR_IMAGE_HEIGHT/2)*22.5;         //kinect fov: 45 degrees vertical pixels:488

            if ((Kinect_Angle + Yrot) == 0) {
                distance = 0.0;
            } else {
                distance = (DeltaHeight / (tan((Kinect_Angle + Yrot)*CV_PI/180)))/12;
            }
        }

#if SHOW_IMAGE
        sprintf(str, "Ratchet Rockers 1706");
        cvPutText(rgb_img, str, cvPoint(420, 20), &font, cvScalar(255,0,255));

        sprintf(str, "Distance = %.2f", distance);
        cvPutText(rgb_img, str, cvPoint(10, 60), &font, cvScalar(255,0,255));

        sprintf(str, "XRot  = %.1f", Xrot);
        cvPutText(rgb_img, str, cvPoint(10, 80), &font, cvScalar(255, 0, 255));

        sprintf(str, "YRot  = %.1f", Yrot);
        cvPutText(rgb_img, str, cvPoint(10, 100), &font, cvScalar(255, 0, 255));

        sprintf(str, "Theta = %.1f", Kinect_Angle);
        cvPutText(rgb_img, str, cvPoint(10, 120), &font, cvScalar(255, 0, 255));

        sprintf(str, "Target Type = %d", TargetType);
        cvPutText(rgb_img, str, cvPoint(10, 140), &font, cvScalar(255, 0, 255));
#endif


#if WRITE_LOG
        logfile.open(logname, fstream::app);
        logfile << setiosflags(ios::fixed) << setprecision(4) << systime << ", "  // Incremental system time since start (sec)
                << distance << ", "                         // Distance to basket (feet)
                << Xrot << ", "                             // X rotation (deg)
                << TargetType << ", "                       // TargetType - 0 = none, 2 = 2pt, 3 = 3pt
                << targetpt[0].x << ", "                    // Upper left corner - x-coordinate (pixels)
                << targetpt[0].y << ", "                    // Upper left corner - y-coordinate (pixels)
                << targetpt[1].x << ", "                    // Upper right corner - x-coordinate (pixels)
                << targetpt[1].y << ", "                    // Upper right corner - y-coordinate (pixels)
                << targetpt[2].x << ", "                    // Lower left corner - x-coordinate (pixels)
                << targetpt[2].y << ", "                    // Lower left corner - y-coordinate (pixels)
                << targetpt[3].x << ", "                    // Lower right corner - x-coordinate (pixels)
                << targetpt[3].y << ", "                    // Lower right corner - y-coordinate (pixels)
                << Target.x << ", "                         // Center of target - x-coordinate relative to center of screen (pixels)
                << Target.y << ", "                         // Center of target - y-coordinate relative to center of screen (pixels)
                << LargestRatio << ", "                     // Aspect ratio of taret
                << imageSaved << ", "                       // Image saved logical
                << imageNum << endl;                        // Imge number
        logfile.close();
        imageSaved = 0;
#endif

        // Write FPS on output image
        STOP_TIMING(IR_Timer);
        frame_time_ms = GET_TIMING(IR_Timer);
        ave_frame_time_ms = (GET_AVERAGE_TIMING(IR_Timer));
        if (frame_time_ms > 0 && ave_frame_time_ms > 0) {
#if SHOW_IMAGE
#if !LOAD_FILE
            sprintf(str, "Current FPS = %.1f", 1000/frame_time_ms);
            cvPutText(rgb_img, str, cvPoint(10, 160),  &font, cvScalar(255, 0, 255, 0));
            sprintf(str, "Average FPS = %.1f", 1000/ave_frame_time_ms);
            cvPutText(rgb_img, str, cvPoint(10, 180),  &font, cvScalar(255, 0, 255, 0));
#endif
#endif
        }
        START_TIMING(IR_Timer);

        do_png_time += frame_time_ms;
        if (do_png_time > PNG_TIME) {
            if (SAVE_PNG) {
                png_count+=1;
                sprintf(pngname, "./%s/ir_img_%d.png",dirname, png_count);
                cvSaveImage(pngname, ir_img, 0);
                do_png_time = 0.0;
                imageSaved = 1;
                imageNum ++;
            }
        }

#if SHOW_IMAGE
        cvShowImage( "Finished", rgb_img );

        int c = cvWaitKey(10);

        if (c == 27)   //escape key
        {
            printf("Key pressed, exitting code ");
            break;
        }
#endif

#if DO_UDP
        if (TargetType == 3) {
            QByteArray datagram = QByteArray::number(distance) + " "
                    + QByteArray::number(Xrot) + " "
                    + QByteArray::number((double)TargetType) + " ";

            udpSocket.writeDatagram(datagram.data(), datagram.size(), QHostAddress(0x0A110602), 80);
        }
#endif
        cvClearMemStorage(storage);
    }

#if SHOW_IMAGE
    cvReleaseImage(&rgb_img);
#endif

    cvReleaseImage(&out);
    cvReleaseImage(&img);

    cvDestroyWindow("Threshold");
    cvDestroyWindow("Finished");
    cvReleaseMemStorage (&storage);

    cvReleaseStructuringElement(&Verticle);

    return 0;
}
