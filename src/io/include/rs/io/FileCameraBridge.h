#ifndef __FILE_CAM_BRIDGE_H__
#define __FILE_CAM_BRIDGE_H__

#include <mutex>
#include <functional>

// ROS
#include <sensor_msgs/CameraInfo.h>

// OpenCV
#include <opencv2/opencv.hpp>

// RS
#include <rs/io/CamInterface.h>
#include <rs/scene_cas.h>


class FileCameraBridge : public CamInterface
{
public:
  FileCameraBridge(const boost::property_tree::ptree &pt): CamInterface(pt) {
    _newData = false;

    cameraInfo.width = 640;
    cameraInfo.height = 480;
    cameraInfo.roi.width = 640;
    cameraInfo.roi.height = 480;
    cameraInfo.roi.x_offset = 0;
    cameraInfo.roi.y_offset = 0;

    cameraInfo.K[0] = 525;
    cameraInfo.K[1] = 0;
    cameraInfo.K[2] = 319.75;
    cameraInfo.K[3] = 0;
    cameraInfo.K[4] = 525;
    cameraInfo.K[5] = 239.75;
    cameraInfo.K[6] = 0;
    cameraInfo.K[7] = 0;
    cameraInfo.K[8] = 1;

    // cameraInfo.P[0] = 525;
    // cameraInfo.P[1] = 0;
    // cameraInfo.P[2] = 319.75;
    // cameraInfo.P[3] = 0;
    // cameraInfo.P[4] = 0;
    // cameraInfo.P[5] = 525;
    // cameraInfo.P[6] = 239.75;
    // cameraInfo.P[7] = 0;
    // cameraInfo.P[8] = 0;
    // cameraInfo.P[9] = 0;
    // cameraInfo.P[10] = 1;
    // cameraInfo.P[11] = 0;

    // loadFrame("color_2.png", "depth_2.png");
    loadFrame("color_3.png", "depth_3.png");
    loadFrame("color_4.png", "depth_4.png");
    loadFrame("color_5.png", "depth.tiff");

    this->frame_rate = pt.get<double>("camera.frame_rate");
    auto worker = std::bind(&FileCameraBridge::threadWorker, this,
      std::chrono::milliseconds(int(1000/frame_rate)));
    this->frame_update_thread = std::thread(worker);
  }

  ~FileCameraBridge() {
    this->done = true;
    outInfo("Waiting for update thread to exit");
    if (this->frame_update_thread.joinable())
      this->frame_update_thread.join();
  }

  bool setData(uima::CAS &tcas, uint64_t ts = std::numeric_limits<uint64_t>::max()) override {
    std::lock_guard<std::mutex> lock(update_mtx);

    cv::Mat color, depth;
    std::tie(color, depth) = frames[current_frame_id];

    current_frame_id = (current_frame_id + 1) % frames.size();

    rs::SceneCas cas(tcas);
    cas.set(VIEW_COLOR_IMAGE, color);
    cas.set(VIEW_DEPTH_IMAGE, depth);

    cas.set(VIEW_CAMERA_INFO, cameraInfo);

    _newData = false;

    return true;
  }

private:
  void threadWorker(std::chrono::milliseconds period) {
    this->done = false;

    while (!done) {
      std::this_thread::sleep_for(period);
      {
        std::lock_guard<std::mutex> lock(update_mtx);
        _newData = true;
        // outInfo("New frame ready");
      }
    }
  }

  void loadFrame(std::string color_name, std::string depth_name) {
    std::string path = ros::package::getPath("robosherlock") + "/res/";

    cv::Mat color = cv::imread(path + color_name);
    cv::Mat depth = cv::imread(path + depth_name,  CV_LOAD_IMAGE_ANYDEPTH);


    cv::resize(color, color, cv::Size(640, 480), 0, 0, cv::INTER_NEAREST);
    cv::resize(depth, depth, cv::Size(640, 480), 0, 0, cv::INTER_NEAREST);

    // cv::threshold(depth, depth, 3, 255, cv::THRESH_TOZERO);

    if (depth.type() == CV_16UC1)
      depth.convertTo(depth, CV_16UC1, 3000./65535, 0);
    else
      depth.convertTo(depth, CV_16UC1, 3000./255, 0);

    this->frames.push_back(std::make_pair(color, depth));
  }

  std::mutex update_mtx;
  std::thread frame_update_thread;

  std::vector<std::tuple<cv::Mat, cv::Mat>> frames;
  bool done{true};

  int current_frame_id{0};

  double frame_rate {0.1};

  sensor_msgs::CameraInfo cameraInfo;
};

#endif // __FILE_CAM_BRIDGE_H__
