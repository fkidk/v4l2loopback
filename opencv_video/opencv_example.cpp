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

#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <assert.h>

#define ROUND_UP_2(num)  (((num)+1)&~1)
#define ROUND_UP_4(num)  (((num)+3)&~3)
#define ROUND_UP_8(num)  (((num)+7)&~7)
#define ROUND_UP_16(num) (((num)+15)&~15)
#define ROUND_UP_32(num) (((num)+31)&~31)
#define ROUND_UP_64(num) (((num)+63)&~63)

#if 0
# define CHECK_REREAD
#endif

#define VIDEO_DEVICE "/dev/video10"
#if 1
# define FRAME_WIDTH  1280
# define FRAME_HEIGHT 720
#else
# define FRAME_WIDTH  512
# define FRAME_HEIGHT 512
#endif

#if 0
# define FRAME_FORMAT V4L2_PIX_FMT_YUYV
#else
# define FRAME_FORMAT V4L2_PIX_FMT_YVU420
#endif

static int debug=0;

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

int format_properties(const unsigned int format,
		const unsigned int width,
		const unsigned int height,
		size_t*linewidth,
		size_t*framewidth) {
size_t lw, fw;
	switch(format) {
	case V4L2_PIX_FMT_YUV420: case V4L2_PIX_FMT_YVU420:
		lw = width; /* ??? */
		fw = ROUND_UP_4 (width) * ROUND_UP_2 (height);
		fw += 2 * ((ROUND_UP_8 (width) / 2) * (ROUND_UP_2 (height) / 2));
	break;
	case V4L2_PIX_FMT_UYVY: case V4L2_PIX_FMT_Y41P: case V4L2_PIX_FMT_YUYV: case V4L2_PIX_FMT_YVYU:
		lw = (ROUND_UP_2 (width) * 2);
		fw = lw * height;
	break;
	default:
		return 0;
	}

	if(linewidth)*linewidth=lw;
	if(framewidth)*framewidth=fw;
	
	return 1;
}

void print_format(struct v4l2_format*vid_format) {
  printf("	vid_format->type                =%d\n",	vid_format->type );
  printf("	vid_format->fmt.pix.width       =%d\n",	vid_format->fmt.pix.width );
  printf("	vid_format->fmt.pix.height      =%d\n",	vid_format->fmt.pix.height );
  printf("	vid_format->fmt.pix.pixelformat =%d\n",	vid_format->fmt.pix.pixelformat);
  printf("	vid_format->fmt.pix.sizeimage   =%d\n",	vid_format->fmt.pix.sizeimage );
  printf("	vid_format->fmt.pix.field       =%d\n",	vid_format->fmt.pix.field );
  printf("	vid_format->fmt.pix.bytesperline=%d\n",	vid_format->fmt.pix.bytesperline );
  printf("	vid_format->fmt.pix.colorspace  =%d\n",	vid_format->fmt.pix.colorspace );
}

int init_video_feed(size_t *framesize, int *fdwr) {
	struct v4l2_capability vid_caps;
	struct v4l2_format vid_format;

	size_t linewidth = 0;

  const char*video_device=VIDEO_DEVICE;
	int ret_code = 0;

	int i;

	fdwr = (int*)open(video_device, O_RDWR);
	assert(fdwr >= 0);

	ret_code = ioctl(*fdwr, VIDIOC_QUERYCAP, &vid_caps);
	assert(ret_code != -1);

	memset(&vid_format, 0, sizeof(vid_format));

	ret_code = ioctl(*fdwr, VIDIOC_G_FMT, &vid_format);
  if(debug)print_format(&vid_format);

	vid_format.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
	vid_format.fmt.pix.width = FRAME_WIDTH;
	vid_format.fmt.pix.height = FRAME_HEIGHT;
	vid_format.fmt.pix.pixelformat = FRAME_FORMAT;
	vid_format.fmt.pix.sizeimage = framesize;
	vid_format.fmt.pix.field = V4L2_FIELD_NONE;
	vid_format.fmt.pix.bytesperline = linewidth;
	vid_format.fmt.pix.colorspace = V4L2_COLORSPACE_SRGB;

  if(debug)print_format(&vid_format);
	ret_code = ioctl(*fdwr, VIDIOC_S_FMT, &vid_format);

	assert(ret_code != -1);

	if(debug)printf("frame: format=%d\tsize=%lu\n", FRAME_FORMAT, framesize);
  print_format(&vid_format);

	if(!format_properties(vid_format.fmt.pix.pixelformat,
                        vid_format.fmt.pix.width, vid_format.fmt.pix.height,
                        &linewidth,
                        framesize)) {
		printf("unable to guess correct settings for format '%d'\n", FRAME_FORMAT);
	}

  return 0;
}

int write_video_feed(cv::Mat *img, size_t * framesize, int *fdwr) {
	write(*fdwr, img, *framesize);
	pause();
  return 0;
}


int close_video_feed(int *fdwr) {
#ifdef CHECK_REREAD
	do {
	/* check if we get the same data on output */
	int fdr = open(video_device, O_RDONLY);
	read(fdr, check_buffer, framesize);
	for (i = 0; i < framesize; ++i) {
		if (buffer[i] != check_buffer[i])
			assert(0);
	}
	close(fdr);
	} while(0);
#endif

	close(*fdwr);
  return 0;
}

int csi() {
  int sensor_mode = 4 ;
  int capture_width = 1280 ;
  int capture_height = 720 ;
  int display_width = 1280 ;
  int display_height = 720 ;
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
                        cv::Point2f(capture_width, 0),
                        cv::Point2f(   0, capture_height),
                        cv::Point2f(capture_width, capture_height)};
  for (int i = 0; i < sizeof(pts1)/sizeof(*pts1); i++) {
      pts1[i] *=  capture_width / 1280;
  }
  cv::Mat M_rear = cv::getPerspectiveTransform(pts1, pts2);

  cv::Point2f pts3[] = {cv::Point2f( 455,   0),
                        cv::Point2f(1105,   0),
                        cv::Point2f(   0, 705),
                        cv::Point2f( 980, 720)};
  for (int i = 0; i < sizeof(pts3)/sizeof(*pts3); i++) {
      pts3[i] *=  capture_width / 1280;
  }
  cv::Mat M_front = cv::getPerspectiveTransform(pts3, pts2);

  size_t framesize = 0;
	int fdwr = 0;
  init_video_feed(&framesize, &fdwr);

  while (true) {
    if (!cuda_rear_img.empty() && !cuda_front_img.empty()) {
      cv::cuda::warpPerspective(cuda_rear_img, dst_rear_img, M_rear, cuda_rear_img.size(), INTER_LINEAR , BORDER_CONSTANT, 0, cuda_rear_stream );
      cv::cuda::warpPerspective(cuda_front_img, dst_front_img, M_front, cuda_front_img.size(), INTER_LINEAR , BORDER_CONSTANT, 0, cuda_front_stream );
      cuda_rear_stream.waitForCompletion();
      cuda_front_stream.waitForCompletion();

      cv::Mat result_img;

      cv::cuda::GpuMat NewImg(capture_height, capture_width * 2, cuda_rear_img.type());
      cuda_front_img.copyTo(NewImg(cv::Rect(0,0,cuda_front_img.cols, cuda_front_img.rows)), cuda_merged_stream);
      cuda_merged_stream.waitForCompletion();
      cuda_rear_img.copyTo(NewImg(cv::Rect(cuda_front_img.cols, 0, cuda_rear_img.cols, cuda_rear_img.rows)), cuda_merged_stream);
      cuda_merged_stream.waitForCompletion();
      NewImg.download(result_img);

      write_video_feed(&result_img, &framesize, &fdwr);

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
