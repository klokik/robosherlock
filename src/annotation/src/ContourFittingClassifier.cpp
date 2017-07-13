#include <numeric>
#include <set>

#include <uima/api.hpp>

// ROS
#include <ros/ros.h>
#include <ros/package.h>

#define PCL_SEGFAULT_WORKAROUND 1

#include <pcl/point_types.h>

#if !PCL_SEGFAULT_WORKAROUND
#include <pcl/registration/icp.h>
#else
#include "libicp/src/icpPointToPoint.h"
#endif

//RS
#include <rs/types/all_types.h>
#include <rs/scene_cas.h>
#include <rs/utils/time.h>
#include <rs/DrawingAnnotator.h>
#include <rs/segmentation/ImageSegmentation.h>

using namespace uima;

using Triangle = std::vector<int>;

struct Mesh {
  std::vector<cv::Point3f> points;
  std::vector<cv::Vec3f> normals;
  std::vector<Triangle> triangles;
};

struct Pose {
  cv::Vec3f rot;
  cv::Vec3f trans;
};

using Silhouette = std::vector<cv::Point2i>;
using Silhouettef = std::vector<cv::Point2f>;

cv::Point2f transform(cv::Mat &M, const cv::Point2f &pt);
cv::Point3f transform(cv::Mat &M, const cv::Point3f &pt);
cv::Vec3f transform(cv::Mat &M, const cv::Vec3f &vec);
::Silhouettef transform(cv::Mat &M, const ::Silhouettef &sil);
::Silhouettef transform(cv::Mat &M, const ::Silhouette &sil);
std::vector<cv::Point3f> transform(cv::Mat &M, const std::vector<cv::Point3f> &points);
cv::Rect_<float> getBoundingRect(::Silhouettef &sil);

struct Footprint {
  cv::Mat img;
  Silhouettef contour;
  Pose pose;
  // cv::Mat camera_matrix;
};

class EdgeModel {
  public: std::string name;
  public: Mesh mesh;

  public: void addFootprint(cv::Mat &footprint, const Silhouettef &contour, const Pose &pose) {
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

    ::Silhouettef contour;
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
        cv::Point2f pt;
        iss >> pt.x >> pt.y;

        contour.push_back(pt);
      }
    }

    auto img = drawFootprint(contour);
    this->addFootprint(img, contour, pose);

    return true;
  }

  protected: cv::Mat drawFootprint(Silhouettef &contour) {
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

struct PoseHypothesis {
  std::string model_name;
  Pose pose;
  double probability;
};

class PoseRanking {
  public: void addElement(::PoseHypothesis &hypothesis) {
    this->hypotheses.insert(hypothesis);
  }

  public: ::PoseHypothesis getTop() const {
    return *this->hypotheses.rbegin();
  }

  public: std::vector<::PoseHypothesis> getTop(const size_t n) const {
    auto num = std::min(n, hypotheses.size());

    std::vector<::PoseHypothesis> result(this->hypotheses.rbegin(), std::next(this->hypotheses.rbegin(), num));

    return result;
  }

/*  public: void normalize() {
    double prob_sum = std::accumulate(this->hypotheses.begin(), this->hypotheses.end(), 0,
      [](const double sum, const ::PoseHypothesis &a) {
        return sum + a.probability; });

    for (auto &hyp : this->hypotheses)
      hyp.probability /= prob_sum;
  }*/

  public: void filter(const double level) noexcept {
    while (true) {
      auto it = std::find_if(this->hypotheses.begin(), this->hypotheses.end(),
        [level](const ::PoseHypothesis &a) {
          return a.probability < level; });

      if (it == this->hypotheses.end())
        break;

      this->hypotheses.erase(it);
    }
  }

  private: class LessProbable {
    public: bool operator()(const ::PoseHypothesis &a, const ::PoseHypothesis &b) {
      return a.probability < b.probability;
    }
  };

  private: std::set<::PoseHypothesis, LessProbable> hypotheses;
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

    bool ignore_cache = true;
    // ctx.extractValue("ignoreSilhouetteCache", ignoreSilhouetteCache);

    std::vector<std::string*> filenames;
    ctx.extractValue("referenceShapes", filenames);

    for (auto &fname : filenames) {
      try {
        EdgeModel edge_model;

        bool cached;
        std::string cache_filename = this->cache_path + fname->substr(fname->rfind('/')) + ".txt";
        outInfo("Cache name " << cache_filename);

        auto training_mesh = readTrainingMesh(*fname);

        if (!(cached = edge_model.loadFromFile(cache_filename)) || ignore_cache) {
          edge_model = getSampledFootprints(
            training_mesh,
            silhouette_camera,
            silhouette_image_size,
            silhouette_margin_size,
            rotation_axis_samples,
            rotation_angle_samples,
            translation_samples);
        } else
          outInfo("File in cache: " + cache_filename);

        edge_model.mesh = training_mesh;

        outInfo("Cached: " << cached);

        this->edge_models[*fname] = edge_model;

        if (!cached || ignore_cache)
          edge_model.saveToFile(cache_filename);
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

  TyErrorId processWithLock(CAS &tcas, ResultSpecification const &res_spec) override
  {
    outInfo("process start");
    rs::StopWatch clock;
    rs::SceneCas cas(tcas);

    cv::Mat cas_image_rgb;
    cv::Mat cas_image_depth;

    if (!cas.get(VIEW_DEPTH_IMAGE, cas_image_depth)) {
      outError("No depth image");
      return UIMA_ERR_NONE;
    }

    if (!cas.get(VIEW_COLOR_IMAGE, cas_image_rgb)) {
      outError("No color image");
      return UIMA_ERR_NONE;
    }

    cv::resize(cas_image_rgb, image_rgb, cas_image_depth.size());

    rs::Scene scene = cas.getScene();
    std::vector<rs::TransparentSegment> t_segments;
    scene.identifiables.filter(t_segments);

    outInfo("Found " << t_segments.size() << " transparent segments");

    // Camera
    cv::Mat K_ref = (cv::Mat_<double>(3, 3) << 570.3422241210938, 0.0, 319.5, 0.0, 570.3422241210938, 239.5, 0.0, 0.0, 1.0);
    cv::Mat distortion = (cv::Mat_<double>(5, 1) << 0, 0, 0, 0, 0);

    this->segments.clear();
    // this->fitted_silhouettes.clear();
    this->labels.clear();
    // this->affine_mesh_transforms.clear();
    this->pose_mesh_transforms.clear();
    for (auto &t_segment : t_segments) {
      rs::Segment segment = t_segment.segment.get();

      ::Silhouettef silhouette;
      for (auto &rs_pt : segment.contour.get()) {
        cv::Point2i cv_pt;
        rs::conversion::from(rs_pt, cv_pt);

        silhouette.push_back(cv::Point2f(cv_pt.x, cv_pt.y));
      }

      outInfo("\tSilhouette of " << silhouette.size() << " points");

      ImageSegmentation::Segment i_segment;
      rs::conversion::from(segment, i_segment);
      this->segments.push_back(i_segment);

      ::PoseRanking ranking;

      for (const auto &kv : this->edge_models) {
        auto &mesh = kv.second.mesh;
        assert(mesh.points.size() >= 4);

        for (auto it = kv.second.items.cbegin(); it != kv.second.items.cend(); it = std::next(it)) {
          ::PoseHypothesis hypothesis;
          hypothesis.model_name = kv.first;

          cv::Mat pr_trans = procrustesTransform(it->contour, silhouette);

          std::vector<cv::Point2f> points_2d;

          cv::projectPoints(mesh.points, it->pose.rot, it->pose.trans, silhouette_camera.matrix, silhouette_camera.ks, points_2d);
          points_2d = transform(pr_trans, points_2d);

          cv::Mat r_vec(3, 1, cv::DataType<double>::type);
          cv::Mat t_vec(3, 1, cv::DataType<double>::type);

          cv::solvePnP(mesh.points, points_2d, K_ref, distortion, r_vec, t_vec);

          Pose pose_3d;

          for (int i = 0; i < 3; ++i) {
            pose_3d.rot(i) = r_vec.at<double>(i);
            pose_3d.trans(i) = t_vec.at<double>(i);
          }

          hypothesis.pose = pose_3d;

          double probability, confidence;
          Silhouettef trans_mesh_sil; // TODO: get transformed mesh silhouette
          std::tie(probability, confidence) = getChamferDistance(trans_mesh_sil, silhouette, cas_image_depth.size());

          hypothesis.probability = probability*confidence;

          ranking.addElement(hypothesis);
        }
      }

      // ranking.normalize();

      // TODO: plot ranking distribution

      ranking.filter(0.1); // TODO: find the right confidence level

      auto hypothesis = ranking.getTop();

      this->labels.push_back(hypothesis.model_name);
      this->pose_mesh_transforms.push_back(hypothesis.pose);

      outInfo("\tProbably it is " << hypothesis.model_name << "; score: " << hypothesis.probability);


      cv::Mat affine_3d_transform(3, 4, CV_32FC1);
      cv::Rodrigues(hypothesis.pose.rot, affine_3d_transform.colRange(0, 3));
      affine_3d_transform.at<float>(0, 3) = hypothesis.pose.trans(0);
      affine_3d_transform.at<float>(1, 3) = hypothesis.pose.trans(1);
      affine_3d_transform.at<float>(2, 3) = hypothesis.pose.trans(2);

      std::cout << affine_3d_transform << std::endl;
      this->affine_mesh_transforms.push_back(affine_3d_transform);

      // break;
    }

    outInfo("took: " << clock.getTime() << " ms.");
    return UIMA_ERR_NONE;
  }


protected:
  void drawImageWithLock(cv::Mat &disp) override {
    disp = image_rgb;
    ImageSegmentation::drawSegments2D(disp, this->segments, this->labels, 1, 0.5);

/*    for (const auto &sil : this->fitted_silhouettes) {
      for (auto &pt : sil)
        cv::circle(disp, pt, 1, cv::Scalar(255, 0, 255), -1);
    }*/

    int i = 0;
    for (const auto &name : this->labels) {
      outInfo("draw");
      Camera cam;
      drawMesh(disp, cam, this->edge_models[name].mesh, this->affine_mesh_transforms[i], this->pose_mesh_transforms[i]);
      ++i;
    }
  }

  static void drawMesh(cv::Mat &dst, Camera &cam, Mesh &mesh, cv::Mat cam_sp_transform, Pose &pose) {
    std::vector<cv::Point3f> vertice;
    std::vector<cv::Vec3f> normal;

    vertice.reserve(mesh.points.size());
    normal.reserve(mesh.normals.size());

    assert(vertice.size() == normal.size());

    auto rect33 = cv::Rect(0, 0, 3, 3);
    cv::Mat cam_sp_rot = cv::Mat::eye(3, 4, CV_32FC1);
    cam_sp_transform(rect33).copyTo(cam_sp_rot(rect33));

    for (const auto &vtx : mesh.points)
      vertice.push_back(vtx);//transform(cam_sp_transform, vtx));

    for (const auto &nrm : mesh.normals)
      normal.push_back(transform(cam_sp_rot, nrm));

    std::vector<cv::Point2f> vertice_2d;

    cv::Mat draw_cam_matrix = (cv::Mat_<double>(3, 3) << 570.3422241210938, 0.0, 319.5, 0.0, 570.3422241210938, 239.5, 0.0, 0.0, 1.0);
    cv::projectPoints(vertice, pose.rot, pose.trans, draw_cam_matrix, {}, vertice_2d);

    cv::Vec3f light = cv::normalize(cv::Vec3f(1, 1, 1));

    float alpha = 0.3f;
    for (const auto &tri : mesh.triangles) {
      cv::Vec3f avg_normal = (normal[tri[0]] + normal[tri[1]] + normal[tri[2]]) / 3;

      float brightness = avg_normal.dot(light);
      cv::Scalar color = cv::Scalar(255*brightness, 255*brightness, 255*brightness);

      std::vector<cv::Point2i> poly{
          vertice_2d[tri[0]],
          vertice_2d[tri[1]],
          vertice_2d[tri[2]]};

      cv::Mat mask = cv::Mat::zeros(dst.size(), CV_8UC3);
      cv::fillConvexPoly(mask, poly, color);

      dst += mask*alpha;
    }

    // for (const auto &pt2 : vertice_2d) {
    //   cv::circle(dst, pt2, 1, cv::Scalar(0, 255, 0), -1);
    // }
  }

  ::Mesh readTrainingMesh(std::string _filename) {
    std::vector<cv::Point3f> points;
    std::vector<cv::Vec3f> normals;
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
          outInfo("Vertices/normals: " << counts[PLYSection::VERTEX]);
          outInfo("Faces: " << counts[PLYSection::FACE]);
        }
      }
      else if (cur_section == PLYSection::VERTEX) {
        if (0 < counts[cur_section]) {
          std::istringstream iss(line);

          cv::Point3f pt;
          cv::Point3f nrm;
          iss >> pt.x >> pt.y >> pt.z >> nrm.x >> nrm.y >> nrm.z;

          points.push_back(pt);
          normals.push_back(nrm);
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

    return {points, normals, triangles};
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
        Pose mesh_pose{rodrigues, {0, 0, -mlen*3.f}};

        auto footprint = getFootprint(mesh, mesh_pose, cam, im_size, marg_size);
        e_model.items.push_back(footprint);
      }
    }

    return e_model;
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
    std::vector<::Silhouette> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(tmp, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    assert(contours.size() == 1);

    cv::Mat bounding_matrix_inv = (cv::Mat_<float>(3, 3) << 1/rate, 0, b_rect.x - marg_size/rate,
                                                            0, 1/rate, b_rect.y - marg_size/rate, 0 , 0, 1);
    Silhouettef contour = transform(bounding_matrix_inv, contours[0]);

    cv::Mat mkernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(footprint, footprint, cv::MORPH_GRADIENT, mkernel,
      cv::Point(-1,-1), 1);

    return {footprint, contour, pose};
  }

  static ::Silhouettef normalizeSilhouette(const ::Silhouettef &shape) {
    ::Silhouettef result;
    for (const auto &pt : shape)
      result.push_back(pt);

    cv::Point2f mean = std::accumulate(shape.cbegin(), shape.cend(), cv::Point2f());
    mean *= (1.f / shape.size());

    float std_dev = 0;

    for (auto &pt : result) {
      pt = pt - mean;
      std_dev += std::pow(cv::norm(pt), 2);
    }

    std_dev = std::sqrt(std_dev / shape.size());

    for (auto &pt : result)
      pt *= 1.f / std_dev;

    return result;
  }

#if !PCL_SEGFAULT_WORKAROUND
  static void silhouetteToPC(::Silhouettef &sil, pcl::PointCloud<pcl::PointXYZ> &pc) {
    pc.width = sil.size();
    pc.height = 1;
    pc.is_dense = false;

    pc.resize(pc.width * pc.height);

    for (size_t i = 0; i < sil.size(); ++i) {
      pc.points[i] = {sil[i].x, sil[i].y, 0};
    }
  }

  static void PCToSilhouette(pcl::PointCloud<pcl::PointXYZ> &pc, ::Silhouettef &sil) {
    sil.clear();

    assert(pc.height == 1);

    for (size_t i = 0; i < pc.width; ++i) {
      sil.push_back(cv::Point2f(pc.points[i].x, pc.points[i].y));
    }
  }

  static std::pair<::Silhouettef, double> fitICP(::Silhouettef &test, ::Silhouettef &model) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cl_test(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cl_model(new pcl::PointCloud<pcl::PointXYZ>);

    silhouetteToPC(test, *cl_test);
    silhouetteToPC(model, *cl_model);

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(cl_test);
    icp.setInputTarget(cl_model);

    pcl::PointCloud<pcl::PointXYZ> cl_result;

    icp.align(cl_result);

    assert(icp.hasConverged());

    double score = icp.getFitnessScore();

    ::Silhouettef result;
    PCToSilhouette(cl_result, result);

    return {result, score};
  }

#else

  static std::pair<cv::Mat, double> fitICP(::Silhouettef &test, ::Silhouettef &model) {

    std::vector<double> test_arr;
    std::vector<double> model_arr;

    for (auto &pt : test) {
      test_arr.push_back(pt.x);
      test_arr.push_back(pt.y);
    }

    for (auto &pt : model) {
      model_arr.push_back(pt.x);
      model_arr.push_back(pt.y);
    }

    int dim = 2; // 2D

    Matrix rot = Matrix::eye(2); // libipc matrix
    Matrix trans(2,1);           // libipc matrix

    IcpPointToPoint icp(&test_arr[0], test.size(), dim);
    double score = icp.fit(&model_arr[0], model.size(), rot, trans, -1);

    cv::Mat cv_trf(3, 3, CV_32FC1, cv::Scalar(0.f));
    cv_trf.at<float>(0, 0) = rot.val[0][0];
    cv_trf.at<float>(0, 1) = rot.val[1][0];
    cv_trf.at<float>(1, 0) = rot.val[0][1];
    cv_trf.at<float>(1, 1) = rot.val[1][1];

    // assume that translation is small enough
    cv_trf.at<float>(0, 2) = 0; //trans.val[0][0];
    cv_trf.at<float>(1, 2) = 0; //trans.val[1][0];
    cv_trf.at<float>(2, 2) = 1.f;

    return {cv_trf, score};
  }
#endif

  static double getFitnessScore(const ::Silhouettef &a, const ::Silhouettef &b) {
    ::Silhouettef na = normalizeSilhouette(a);
    ::Silhouettef nb = normalizeSilhouette(b);

    auto transform__score = fitICP(na, nb);

    return transform__score.second;
  }

  std::tuple<std::string, size_t, double> getBestFitnessResult(const Silhouettef &shape) {
    std::string best_model_name;
    size_t best_pose_index{0};
    double best_fitness_score{std::numeric_limits<double>::max()};

    for (const auto &kv : this->edge_models) {
      for (auto it = kv.second.items.cbegin(); it != kv.second.items.cend(); it = std::next(it)) {
        // outInfo("pose " << std::distance(kv.second.items.cbegin(), it));
        auto score = getFitnessScore(shape, it->contour);

        if (score < best_fitness_score) {
          best_fitness_score = score;
          best_model_name = kv.first;
          best_pose_index = std::distance(kv.second.items.cbegin(), it);
        }
      }
    }

    return std::tuple<std::string, size_t, double>{best_model_name, best_pose_index, best_fitness_score};
  }

  static std::pair<cv::Vec2f, float> getMeanAndDev(const ::Silhouettef &sil) {
    cv::Point2f mean = std::accumulate(sil.cbegin(), sil.cend(), cv::Point2f());
    mean *= (1.f / sil.size());

    float std_dev = 0;
    for (auto &pt : sil)
      std_dev += std::pow(cv::norm(cv::Point2f(pt.x, pt.y) - mean), 2);

    std_dev = std::sqrt(std_dev / sil.size());

    return std::make_pair(mean, std_dev);
  }

  static cv::Mat procrustesTransform(const ::Silhouettef &sil, const ::Silhouettef &tmplt) {
    cv::Vec2f mean_s, mean_t;
    float deviation_s, deviation_t;

    // FIXME: avoid double mean/deviation computation
    std::tie(mean_s, deviation_s) = getMeanAndDev(sil);
    std::tie(mean_t, deviation_t) = getMeanAndDev(tmplt);

    auto ns = normalizeSilhouette(sil);
    auto nt = normalizeSilhouette(tmplt);

    cv::Mat icp_mat;
    std::tie(icp_mat, std::ignore) = fitICP(ns, nt);

    cv::Mat Ts_inv = (cv::Mat_<float>(3, 3) << 1, 0, -mean_s(0), 0, 1, -mean_s(1), 0 , 0, 1);
    cv::Mat Ss_inv = (cv::Mat_<float>(3, 3) << 1/deviation_s, 0, 0, 0, 1/deviation_s, 0, 0 , 0, 1);
    cv::Mat Rst = icp_mat;
    cv::Mat St = (cv::Mat_<float>(3, 3) << deviation_t, 0, 0, 0, deviation_t, 0, 0 , 0, 1);
    cv::Mat Tt = (cv::Mat_<float>(3, 3) << 1, 0, mean_t(0), 0, 1, mean_t(1), 0 , 0, 1);

    cv::Mat sil_to_tmplt_transformation = Tt * St * Rst * Ss_inv * Ts_inv;

    return sil_to_tmplt_transformation;
  }

  static std::pair<double, double> getChamferDistance(::Silhouettef &a, ::Silhouettef &b, cv::Size work_area) {
    throw std::runtime_error("Not implemented");

    size_t distance_sum = 0;
    size_t num_points_hit = 0;

    cv::Mat dist_map = cv::Mat::ones(work_area, CV_16UC1);
    for (auto it = b.begin(); it != std::prev(b.end()); it = std::next(it))
      cv::line(dist_map, *it, *std::next(it), cv::Scalar(0));
    cv::line(dist_map, b.back(), b.front(), cv::Scalar(0));

    cv::distanceTransform(dist_map, dist_map, CV_DIST_L2, 3);

    auto work_rect = cv::Rect(cv::Point(0, 0), work_area);
    for (const auto &pt : a) {
      auto pti = cv::Point(pt.x, pt.y);
      if (work_rect.contains(pti)) {
        distance_sum += dist_map.at<int16_t>(pti);
        num_points_hit += 1;
      }
    }

    double confidence = static_cast<double>(num_points_hit) / a.size();

    if (confidence == 0)
      throw std::runtime_error("input contour is too large or contains no points");

    double distance = static_cast<double>(distance_sum) / num_points_hit;

    return std::make_pair(distance, confidence);
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

  std::vector<ImageSegmentation::Segment> segments;
  // std::vector<::PoseRanking> rankings;
  // std::vector<Silhouettef> fitted_silhouettes;
  std::vector<cv::Mat> affine_mesh_transforms;
  std::vector<::Pose> pose_mesh_transforms;
  std::vector<std::string> labels;

  cv::Mat image_rgb;

  ::Camera silhouette_camera;
};

cv::Point2f transform(cv::Mat &M, const cv::Point2f &pt) {
  cv::Mat vec(3, 1, CV_32FC1);

  vec.at<float>(0, 0) = pt.x;
  vec.at<float>(1, 0) = pt.y;
  vec.at<float>(2, 0) = 1.f;

  cv::Mat dst = M * vec;

  return cv::Point2f(dst.at<float>(0, 0), dst.at<float>(1, 0));
}

cv::Point3f transform(cv::Mat &M, const cv::Point3f &pt) {
  cv::Mat vec(4, 1, CV_32FC1);

  vec.at<float>(0, 0) = pt.x;
  vec.at<float>(1, 0) = pt.y;
  vec.at<float>(2, 0) = pt.z;
  vec.at<float>(3, 0) = 1.f;

  cv::Mat dst = M * vec;

  return cv::Point3f(dst.at<float>(0, 0), dst.at<float>(1, 0), dst.at<float>(2, 0));
}

cv::Vec3f transform(cv::Mat &M, const cv::Vec3f &vec) {
  cv::Point3f pt(vec[0], vec[1], vec[2]);

  return transform(M, pt);
}

::Silhouettef transform(cv::Mat &M, const ::Silhouettef &sil) {
  Silhouettef result;

  for (const auto &pt : sil)
    result.push_back(transform(M, pt));

  return result;
}

::Silhouettef transform(cv::Mat &M, const ::Silhouette &sil) {
  Silhouettef result;

  for (const auto &pt : sil) {
    cv::Point2f ptf(pt.x, pt.y);

    result.push_back(transform(M, ptf));
  }

  return result;
}

std::vector<cv::Point3f> transform(cv::Mat &M, const std::vector<cv::Point3f> &points) {
  std::vector<cv::Point3f> result;
  result.reserve(points.size());

  for (const auto &pt : points)
    result.push_back(transform(M, pt));

  return result;
}

cv::Rect_<float> getBoundingRect(::Silhouettef &sil) {
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

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(ContourFittingClassifier)