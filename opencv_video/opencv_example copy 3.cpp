#include <iostream>
#include <thread>
#include <chrono>
using namespace std;

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;

//Add CUDA support
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
using namespace cv::cuda;

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>

#define VIDEO_DEVICE "/dev/video10"
#define FRAME_FORMAT V4L2_PIX_FMT_YUV420
#define FRAME_WIDTH 1280
#define FRAME_HEIGHT 720
#define CHANNELS 3
#define SIZE_IMGE (FRAME_WIDTH * 2 * FRAME_HEIGHT) * CHANNELS

std::string gstreamer_pipeline (int sensor_id=0,
                                int sensor_mode=3,
                                int capture_width=1280,
                                int capture_height=720,
                                int display_width=1280,
                                int display_height=720,
                                int framerate=30,
                                int flip_method=0) {
    return "nvarguscamerasrc sensor-id=" + std::to_string(sensor_id) + " sensor-mode=" + std::to_string(sensor_mode) + " ! " \
           "video/x-raw(memory:NVMM), " \
           "width=(int)" + std::to_string(capture_width) + ", " \
           "height=(int)" + std::to_string(capture_height) + ", " \
           "format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) +"/1 ! " \
           "nvvidconv flip-method=" + std::to_string(flip_method) + " ! " \
           "video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" + std::to_string(display_height) + ", format=(string)BGRx ! " \
           "videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

int camera(std::string pipeline, cv::cuda::GpuMat *cuda_img, cv::cuda::Stream *cuda_stream, bool *stop_thread) {
  std::cout << "Using pipeline: \n\t" << pipeline << "\n";

  cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
  if(!cap.isOpened()) {
	  std::cout<<"Failed to open camera."<<std::endl;
    cap.release();
	  return (-1);
  }

  cv::Mat img;

  while(true) {
    if (!cap.read(img)) {
		  std::cout<<"Capture read error"<<std::endl;
		  break;
	  }
    if (*stop_thread) {
      break;
    }
    cuda_img->upload(img, *cuda_stream);
    cuda_stream->waitForCompletion();
//    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  cap.release();
  return 0;
}

int open_video() {
	struct v4l2_format v;

  const char*video_device=VIDEO_DEVICE;
	int dev_fd = open(video_device, O_RDWR);
	if (dev_fd == -1) {
		std::cout << "cannot open video device\n";
	}
	v.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
	if (ioctl(dev_fd, VIDIOC_G_FMT, &v) == -1){
		std::cout << "cannot setup video device\n";
	}
	v.fmt.pix.width = FRAME_WIDTH * 2;
	v.fmt.pix.height = FRAME_HEIGHT;
	v.fmt.pix.pixelformat = FRAME_FORMAT;
	v.fmt.pix.sizeimage = SIZE_IMGE;
	v.fmt.pix.field = V4L2_FIELD_NONE;
  v.fmt.pix.colorspace = V4L2_COLORSPACE_SRGB;
	if (ioctl(dev_fd, VIDIOC_S_FMT, &v) == -1){
		std::cout << "cannot setup video device\n";
	}

	return dev_fd;
}

int csi() {
  int sensor_mode = 4 ;
  int capture_width = FRAME_WIDTH ;
  int capture_height = FRAME_HEIGHT ;
  int display_width = FRAME_WIDTH ;
  int display_height = FRAME_HEIGHT ;
  int framerate = 60 ;
  int flip_method = 0 ;

  std::string pipeline0 = gstreamer_pipeline(0, 
                                            sensor_mode,
                                            capture_width,
	                                          capture_height,
                                            display_width,
                                            display_height,
                                            framerate,
                                            flip_method);

  std::string pipeline1 = gstreamer_pipeline(1,
                                            sensor_mode,
                                            capture_width,
	                                          capture_height,
                                            display_width,
                                            display_height,
                                            framerate,
                                            flip_method);

  cv::cuda::GpuMat cuda_rear_img, cuda_front_img, dst_rear_img, dst_front_img;

  cv::cuda::Stream cuda_rear_stream, cuda_front_stream, cuda_merged_stream;
  
  bool stop_threads = false;

  thread rear(camera, pipeline0, &cuda_rear_img, &cuda_rear_stream, &stop_threads);
  thread front(camera, pipeline1, &cuda_front_img, &cuda_front_stream, &stop_threads);

  cv::Point2f pts1[] = {cv::Point2f( 240,   0),
                        cv::Point2f( 920,   0),
                        cv::Point2f( 227, 720),
                        cv::Point2f(1275, 720)};
  cv::Point2f pts2[] = {cv::Point2f(   0,   0),
                        cv::Point2f(FRAME_WIDTH, 0),
                        cv::Point2f(   0, FRAME_HEIGHT),
                        cv::Point2f(FRAME_WIDTH, FRAME_HEIGHT)};
  for (int i = 0; i < sizeof(pts1)/sizeof(*pts1); i++) {
      pts1[i] *=  FRAME_WIDTH / 1280;
  }
  cv::Mat M_rear = cv::getPerspectiveTransform(pts1, pts2);

  cv::Point2f pts3[] = {cv::Point2f( 455,   0),
                        cv::Point2f(1105,   0),
                        cv::Point2f(   0, 705),
                        cv::Point2f( 980, 720)};
  for (int i = 0; i < sizeof(pts3)/sizeof(*pts3); i++) {
      pts3[i] *=  FRAME_WIDTH / 1280;
  }
  cv::Mat M_front = cv::getPerspectiveTransform(pts3, pts2);

/* START PREPARE VIDEO STREAM */
  int loc_dev = open_video();
/* END PREPARE VIDEO STREAM */

  while (true) {
    if (!cuda_rear_img.empty() && !cuda_front_img.empty()) {
      cv::cuda::warpPerspective(cuda_rear_img, dst_rear_img, M_rear, cuda_rear_img.size(), INTER_LINEAR , BORDER_CONSTANT, 0, cuda_rear_stream );
      cv::cuda::warpPerspective(cuda_front_img, dst_front_img, M_front, cuda_front_img.size(), INTER_LINEAR , BORDER_CONSTANT, 0, cuda_front_stream );
      cuda_rear_stream.waitForCompletion();
      cuda_front_stream.waitForCompletion();

      cv::Mat result_img;

      cv::cuda::GpuMat NewImg(capture_height, capture_width * 2, cuda_rear_img.type());
      dst_front_img.copyTo(NewImg(cv::Rect(0,0,dst_front_img.cols, dst_front_img.rows)), cuda_merged_stream);
      cuda_merged_stream.waitForCompletion();
      dst_rear_img.copyTo(NewImg(cv::Rect(dst_front_img.cols, 0, dst_rear_img.cols, dst_rear_img.rows)), cuda_merged_stream);
      cuda_merged_stream.waitForCompletion();
//      cv::cuda::cvtColor(NewImg, NewImg, cv::COLOR_BGR2RGBA, 0, cuda_merged_stream);
//      cuda_merged_stream.waitForCompletion();
      NewImg.download(result_img);

    	write(loc_dev, result_img.data, SIZE_IMGE);

/*      cv::imshow("Merged", result_img);
      if (cv::waitKey(30) == (char)27) {
        break;
      };*/
    }
  }

  stop_threads = true;

  destroyAllWindows();

  rear.join();
  front.join();

  return 0;
}

int main() {
    printShortCudaDeviceInfo(getDevice());
    int cuda_devices_number = getCudaEnabledDeviceCount();
    cout << "CUDA Device(s) Number: "<< cuda_devices_number << endl;
    DeviceInfo _deviceInfo;
    bool _isd_evice_compatible = _deviceInfo.isCompatible();
    cout << "CUDA Device(s) Compatible: " << _isd_evice_compatible << endl;

    return csi();
}
