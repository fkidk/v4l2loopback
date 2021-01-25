#include <iostream>
#include <thread>
using namespace std;

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

// use this for getting info on camera: v4l2-ctl --all -d /dev/video0
/*
                             "video/x-raw, width=(int)1280, height=(int)720, " \
                             "format=(string)BGRx ! " \
                             "videoconvert ! " \
                             "video/x-raw, width=(int)1280, height=(int)720, " \
                             "format=(string)BGR ! " \
                             "videoconvert ! " \
*/

int main() {
    std::string pipelineIn = "v4l2src device=\"/dev/video0\" ! queue ! " \
                             "video/x-raw, width=(int)1280, height=(int)720, " \
                             "format=(string)YUY2, framerate=(fraction)10/1 ! " \
                             "videoconvert ! " \
                             "appsink";

/*
                              "video/x-raw, width=(int)1280, height=(int)720, " \
                              "format=(string)YUY2, framerate=(fraction)10/1 ! " \
*/
    std::string pipelineOut = "appsrc num-buffers=1000 ! " \
                              "videoconvert ! " \
                              "vdpau, bitrate=8000000, SliceIntraRefreshEnable=true ! " \
                              "video/x-h264, stream-format=byte-stream ! h264parse ! qtmux ! " \
                              "v4l2sink device=\"/dev/video10\"";

    std::cout << "Using pipeline in : \n\t" << pipelineIn << "\n";
    std::cout << "Using pipeline out: \n\t" << pipelineOut << "\n";

    cv::VideoCapture read_frame(pipelineIn, cv::CAP_GSTREAMER);
    cv::VideoWriter write_frame;

    std::cout << "FPS: " << read_frame.get(cv::CAP_PROP_FPS) << "\n";

    int codec = cv::VideoWriter::fourcc('Y', 'U', 'Y', '2');
    write_frame.open(pipelineOut, codec, 10.0, cv::Size(1280, 720), true);

    cv::Mat img;

    while(true) {
        read_frame.read(img);

//        cv::cvtColor(img, img, cv::COLOR_YUV2RGB_YUY2);
/*
        cv::imshow("test", img);
        if (cv::waitKey(30) == (char)27) {
            break;
        }
*/
        write_frame.write(img);
    }

    read_frame.release();
    return 0;
}