#include <numeric>

#include <uima/api.hpp>

// ROS
#include <ros/ros.h>
#include <ros/package.h>

#include <pcl/point_types.h>
#include <rs/types/all_types.h>
//RS
#include <rs/scene_cas.h>
#include <rs/utils/time.h>
#include <rs/DrawingAnnotator.h>


using namespace uima;

using Triangle = std::vector<int>;

struct Mesh {
  std::vector<cv::Point3f> points;
  std::vector<Triangle> triangles;
};

struct Pose {
  cv::Vec3f rot;
  cv::Vec3f trans;
};

using Silhouette = std::vector<cv::Point2i>;
using Silhouettef = std::vector<cv::Point2f>;

struct Footprint {
  cv::Mat img;
  Silhouette contour;
  Pose pose;
};

class EdgeModel {
  public: std::string name;
  public: Mesh mesh;

  public: void addFootprint(cv::Mat &footprint, const Silhouette &contour, const Pose &pose) {
    if (contour.size() > 0)
      this->items.push_back({footprint.clone(), contour, pose});
  }

  public: void saveToFile(std::string name) {
    std::ofstream ofs(name);

    for (auto &fp : this->items) {
      ofs << "pose: " << fp.pose.rot(0) << " " << fp.pose.rot(1) << " " << fp.pose.rot(2) << " "
                      << fp.pose.trans(0) << " " << fp.pose.trans(1) << std::endl;

      for (auto &pt : fp.contour) {
        ofs << pt.x << " " << pt.y << std::endl;
      }
    }

    ofs.close();
  }

  public: bool loadFromFile(std::string name) {
    std::ifstream ifs(name);

    // check if file exists
    if (!ifs.good())
      return false;

    Silhouette contour;
    Pose pose;

    for (std::string line; std::getline(ifs, line);) {
      std::istringstream iss(line);

      if (line.find("pose") == 0) {
        auto img = drawFootprint(contour);
        this->addFootprint(img, contour, pose);
        contour.clear();

        iss >> pose.rot(0) >> pose.rot(1) >> pose.rot(2) >> pose.trans(0) >> pose.trans(1);
      }
      else {
        cv::Point2i pt;
        iss >> pt.x >> pt.y;

        contour.push_back(pt);
      }
    }

    auto img = drawFootprint(contour);
    this->addFootprint(img, contour, pose);

    return true;
  }

  protected: cv::Mat drawFootprint(Silhouette &contour) {
    if (contour.size() == 0)
      return cv::Mat();

    cv::Rect b_rect = cv::boundingRect(contour);

    cv::Mat result = cv::Mat::zeros(b_rect.height + 8, b_rect.width + 8, CV_8UC1);

    for (auto &pt : contour)
      cv::circle(result, pt, 1, cv::Scalar(255), -1);

    return result;
  }

  public: std::vector<Footprint> items;
};

struct Camera {
  cv::Mat matrix = cv::Mat::eye(3, 3, CV_32FC1);
  std::vector<float> ks;
};

class ContourFittingClassifier : public DrawingAnnotator
{
public:
  ContourFittingClassifier(): DrawingAnnotator(__func__) {

  }

  TyErrorId initialize(AnnotatorContext &ctx)
  {
    outInfo("initialize");
    ctx.extractValue("cachePath", cache_path);
    ctx.extractValue("rotationAxisSamples", rotation_axis_samples);
    ctx.extractValue("rotationAngleSamples", rotation_angle_samples);
    ctx.extractValue("silhouetteImageSize", silhouette_image_size);
    ctx.extractValue("silhouetteMarginSize", silhouette_margin_size);
    ctx.extractValue("icpIterationsProcrustes", icp_iteration_procrustes);

    std::vector<std::string*> filenames;
    ctx.extractValue("referenceShapes", filenames);

    for (auto &fname : filenames) {
      try {
        auto training_mesh = readTrainingMesh(*fname);

        ::EdgeModel edge_model;
        edge_model = getSampledFootprints(
          training_mesh,
          silhouette_camera,
          silhouette_image_size,
          silhouette_margin_size,
          rotation_axis_samples,
          rotation_angle_samples,
          translation_samples);

        this->edge_models[*fname] = edge_model;
      } catch (const std::exception &ex) {
        outError(ex.what());
      }

      delete fname;
    }

    if (this->edge_models.empty())
      outError("No valid meshes found");

    return UIMA_ERR_NONE;
  }

  TyErrorId destroy()
  {
    outInfo("destroy");
    return UIMA_ERR_NONE;
  }

  // TyErrorId process(CAS &tcas, ResultSpecification const &res_spec)
  // {
  //   outInfo("process start");
  //   rs::StopWatch clock;
  //   rs::SceneCas cas(tcas);
  //   pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGBA>);
  //   outInfo("Cache path =  " << cache_path);
  //   cas.get(VIEW_CLOUD,*cloud_ptr);

  //   outInfo("Cloud size: " << cloud_ptr->points.size());
  //   outInfo("took: " << clock.getTime() << " ms.");
  //   return UIMA_ERR_NONE;
  // }

  TyErrorId processWithLock(CAS &tcas, ResultSpecification const &res_spec) override
  {
    outInfo("process start");
    rs::SceneCas cas(tcas);

    outInfo("Cache path =  " << cache_path);


    return UIMA_ERR_NONE;
  }


protected:
  void drawImageWithLock(cv::Mat &disp) override {
    auto seg_size = silhouette_image_size+silhouette_margin_size*2+1;
    cv::Mat whole_image = cv::Mat::zeros(seg_size*rotation_axis_samples,
      seg_size*rotation_angle_samples, CV_8UC1);

    if (!this->edge_models.empty()) {
      auto edge_model_it = this->edge_models.begin();
      // TODO: cyclic std::advance(edge_model_it) each N seconds

      int i = 0;
      for (auto kv : edge_model_it->second.items) {
        auto ix = (i % rotation_angle_samples)*seg_size;
        auto iy = (i / rotation_angle_samples)*seg_size;
        i++;

        kv.img.copyTo(whole_image.colRange(ix, ix+kv.img.cols).
          rowRange(iy, iy+kv.img.rows));
      }
    }

    cv::cvtColor(whole_image, disp, CV_GRAY2BGR);
  }

  ::Mesh readTrainingMesh(std::string _filename) {
    std::vector<cv::Point3f> points; 
    std::vector<Triangle> triangles; 

    std::string filename = ros::package::getPath("robosherlock") + _filename;
    std::ifstream ifs(filename);

    if (!ifs.good())
      throw std::runtime_error("File '"+filename+"' not found");

    enum class PLYSection : int { HEADER=0, VERTEX, FACE};
    std::map<PLYSection, int> counts;

    PLYSection cur_section = PLYSection::HEADER;
    for (std::string line; std::getline(ifs, line);) {
      if (cur_section == PLYSection::HEADER) {
        if (line.find("element face") == 0)
          counts[PLYSection::FACE] = std::atoi(line.substr(line.rfind(" ")).c_str());
        if (line.find("element vertex") == 0)
          counts[PLYSection::VERTEX] = std::atoi(line.substr(line.rfind(" ")).c_str());
        if (line.find("end_header") == 0) {
          cur_section = PLYSection::VERTEX;
          outInfo("Vertices: " << counts[PLYSection::VERTEX]);
          outInfo("Faces: " << counts[PLYSection::FACE]);
        }
      }
      else if (cur_section == PLYSection::VERTEX) {
        if (0 < counts[cur_section]) {
          std::istringstream iss(line);

          cv::Point3f pt;
          iss >> pt.x >> pt.y >> pt.z;

          points.push_back(pt);
          --counts[cur_section];
        }
        else
          cur_section = PLYSection::FACE;
      }
      if (cur_section == PLYSection::FACE) {
        if (0 == counts[cur_section]--)
          break;

        std::istringstream iss(line);

        int n_verts, i1, i2, i3;
        iss >> n_verts >> i1 >> i2 >> i3;
        assert(n_verts == 3);

        triangles.push_back({i1, i2, i3});
      }
    }

    assert(counts[PLYSection::VERTEX] == 0);
    assert(counts[PLYSection::FACE] == 0);

    return {points, triangles};
  }

  static ::EdgeModel getSampledFootprints(const Mesh &mesh, Camera &cam, int im_size,
     int marg_size, int rot_axis_samples, int rot_angle_samples, int trans_samples) {

    ::EdgeModel e_model;

    auto it = std::max_element(mesh.points.cbegin(), mesh.points.cend(),
      [](const cv::Point3f &a, const cv::Point3f &b) {
        return cv::norm(a) < cv::norm(b); });
    float mlen = cv::norm(*it);

    const auto pi = 3.1415926f;
    for (int r_ax_i = 0; r_ax_i < rot_axis_samples; ++r_ax_i) {
      float axis_inc = pi * r_ax_i / rot_axis_samples;
      cv::Vec3f axis{std::cos(axis_inc), std::sin(axis_inc), 0};

      for (int r_ang_i = 0; r_ang_i < rot_angle_samples; ++r_ang_i ) {
        float theta = pi * r_ang_i / rot_angle_samples;
        auto rodrigues = axis*theta;

        outInfo("Training sample (" << r_ax_i << ";" << r_ang_i << ")");

        // avoid translation sampling for now
        Pose mesh_pose{rodrigues, {0,0,-mlen*3.f}};

        auto footprint = getFootprint(mesh, mesh_pose, cam, im_size, marg_size);
        e_model.addFootprint(footprint.img, footprint.contour, footprint.pose);
      }
    }

    return e_model;
  }

  static cv::Rect_<float> getBoundingRect(::Silhouettef &sil) {
    cv::Rect_<float> b_rect;

    auto h_it = std::minmax_element(sil.begin(), sil.end(),
      [](const cv::Point2f &a, const cv::Point2f &b) {
        return a.x < b.x;});
    auto v_it = std::minmax_element(sil.begin(), sil.end(),
      [](const cv::Point2f &a, const cv::Point2f &b) {
        return a.y < b.y;});

    b_rect.x = h_it.first->x;
    b_rect.y = v_it.first->y;
    b_rect.width = h_it.second->x - b_rect.x;
    b_rect.height = v_it.second->y - b_rect.y;

    return b_rect;
  }

  static ::Footprint getFootprint(const ::Mesh &mesh,
      const Pose &pose, ::Camera &cam,const int im_size, const int marg_size) {
    // project points on a plane
    std::vector<cv::Point2f> points2d;
    cv::projectPoints(mesh.points, pose.rot, pose.trans, cam.matrix, cam.ks, points2d);

    // find points2d bounding rect
    cv::Rect_<float> b_rect;
    // b_rect = cv::boundingRect2f(points2d); // available since 2.4.something
    b_rect = getBoundingRect(points2d);

    auto larger_size = std::max(b_rect.width, b_rect.height);

    auto rate = static_cast<float>(im_size)/larger_size;
    cv::Size fp_mat_size(b_rect.width*rate + 1, b_rect.height*rate + 1);

    cv::Mat footprint = cv::Mat::zeros(fp_mat_size, CV_8UC1);

    // std::cout << fp_mat_size << std::endl;
    // std::cout << b_rect << std::endl;
    // map point onto plane
    for (auto &point : points2d) {
      cv::Point2i xy = (point - b_rect.tl())*rate;

      assert(xy.x >= 0);
      assert(xy.y >= 0);
      assert(xy.x <= footprint.cols);
      assert(xy.y <= footprint.rows);

      xy.x = std::min(xy.x, footprint.cols-1);
      xy.y = std::min(xy.y, footprint.rows-1);

      point = xy;
    }

    for (const auto &tri : mesh.triangles) {
      std::vector<cv::Point2i> poly{
          points2d[tri[0]],
          points2d[tri[1]],
          points2d[tri[2]]};

      cv::fillConvexPoly(footprint, poly, cv::Scalar(255));
    }

    cv::copyMakeBorder(footprint, footprint,
      marg_size, marg_size, marg_size, marg_size,
      cv::BORDER_CONSTANT | cv::BORDER_ISOLATED, cv::Scalar(0));

    cv::Mat tmp = footprint.clone();
    std::vector<Silhouette> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(tmp, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    assert(contours.size() == 1);

    cv::Mat mkernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(footprint, footprint, cv::MORPH_GRADIENT, mkernel,
      cv::Point(-1,-1), 1);

    // cv::imshow("footprint", footprint);
    // cv::waitKey(100);

    return {footprint, contours[0], pose};
  }

private:
  std::string cache_path{"/tmp"};
  int rotation_axis_samples{10};
  int rotation_angle_samples{10};
  int translation_samples{1};
  int silhouette_image_size{240};
  int silhouette_margin_size{4};
  int icp_iteration_procrustes{100};

  std::map<std::string, EdgeModel> edge_models;

  ::Camera silhouette_camera;
};

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(ContourFittingClassifier)