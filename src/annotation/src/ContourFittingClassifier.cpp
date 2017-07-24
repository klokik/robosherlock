#include <numeric>
#include <set>

#include <uima/api.hpp>

// ROS
#include <ros/ros.h>
#include <ros/package.h>

#define PCL_SEGFAULT_WORKAROUND 1

#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>


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

struct Camera {
  cv::Mat matrix = cv::Mat::eye(3, 3, CV_32FC1);
  std::vector<float> ks;
};

using Silhouette = std::vector<cv::Point2i>;
using Silhouettef = std::vector<cv::Point2f>;

cv::Point2f transform(const cv::Mat &M, const cv::Point2f &pt);
cv::Point3f transform(const cv::Mat &M, const cv::Point3f &pt);
cv::Vec3f transform(const cv::Mat &M, const cv::Vec3f &vec);
::Silhouettef transform(const cv::Mat &M, const ::Silhouettef &sil);
::Silhouettef transform(const cv::Mat &M, const ::Silhouette &sil);
std::vector<cv::Point3f> transform(const cv::Mat &M, const std::vector<cv::Point3f> &points);
::Silhouettef normalizeSilhouette(const ::Silhouettef &shape);
cv::Rect_<float> getBoundingRect(const ::Silhouettef &sil);
cv::Mat poseToAffine(const ::Pose &pose);
::Pose operator+(const ::Pose &a, const Pose &b);
::Pose operator*(const double a, const Pose &b);
std::tuple<::Pose, double> fit2d3d(::Mesh &mesh, ::Pose &init_pose, ::Silhouettef &template_2d, ::Camera &camera);
::Silhouettef getCannySilhouette(cv::Mat &grayscale, cv::Rect &input_roi);

struct Footprint {
  cv::Mat img;
  Silhouettef contour;
  Silhouettef normalized_contour;
  Pose pose;
  // cv::Mat camera_matrix;
};

class EdgeModel {
  public: std::string name;
  public: Mesh mesh;
  public: Mesh edge_mesh;

  public: void addFootprint(cv::Mat &footprint, const Silhouettef &contour, const Pose &pose) {
    if (contour.size() > 0) {
      auto normalized_contour = normalizeSilhouette(contour);
      this->items.push_back({footprint.clone(), contour, normalized_contour, pose});
    }
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

  // in l_infinity sense
  public: double normalize() {
    double prob_norm = std::accumulate(this->hypotheses.begin(), this->hypotheses.end(), 0.,
      [](const double acc, const ::PoseHypothesis &a) {
        return std::max(acc, a.probability); });

    decltype(this->hypotheses) normalised;

    for (auto &hyp : this->hypotheses) {
      auto tmp = hyp;
      tmp.probability /= prob_norm;

      normalised.insert(tmp);
    }

    this->hypotheses = normalised;

    return prob_norm;
  }

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

  public: size_t size() const noexcept{
    return this->hypotheses.size();
  }

  private: class LessProbable {
    public: bool operator()(const ::PoseHypothesis &a, const ::PoseHypothesis &b) {
      return a.probability < b.probability;
    }
  };

  private: std::set<::PoseHypothesis, LessProbable> hypotheses;
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

        auto offset = fname->find('|');
        assert(offset != std::string::npos);

        std::string mesh_fname = fname->substr(0, offset);
        std::string edge_fname = fname->substr(offset+1, fname->size());

        bool cached;
        std::string cache_filename = this->cache_path + mesh_fname.substr(mesh_fname.rfind('/')) + ".txt";
        outInfo("Cache name " << cache_filename);

        auto training_mesh = readTrainingMesh(mesh_fname);
        auto edge_mesh = readTrainingMesh(edge_fname);

        if (ignore_cache || !(cached = edge_model.loadFromFile(cache_filename))) {
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
        edge_model.edge_mesh = edge_mesh;

        outInfo("Cached: " << cached);

        this->edge_models[mesh_fname] = edge_model;

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
    cv::Mat image_grayscale;

    if (!cas.get(VIEW_DEPTH_IMAGE, cas_image_depth)) {
      outError("No depth image");
      return UIMA_ERR_NONE;
    }

    if (!cas.get(VIEW_COLOR_IMAGE, cas_image_rgb)) {
      outError("No color image");
      return UIMA_ERR_NONE;
    }

    cv::resize(cas_image_rgb, image_rgb, cas_image_depth.size());
    cv::cvtColor(image_rgb, image_grayscale, CV_BGR2GRAY);

    if (!cas.get(VIEW_CLOUD, *view_cloud)) {
      outError("No view point cloud");
      return UIMA_ERR_NONE;
    }

    rs::Scene scene = cas.getScene();
    std::vector<rs::TransparentSegment> t_segments;
    scene.identifiables.filter(t_segments);

    outInfo("Found " << t_segments.size() << " transparent segments");

    // Camera
    cv::Mat K_ref = (cv::Mat_<double>(3, 3) << 570.3422241210938, 0.0, 319.5, 0.0, 570.3422241210938, 239.5, 0.0, 0.0, 1.0);
    cv::Mat distortion = (cv::Mat_<double>(5, 1) << 0, 0, 0, 0, 0);

    ::Camera world_camera;
    world_camera.matrix = K_ref;
    //world_camera.ks = {};

    this->segments.clear();
    // this->fitted_silhouettes.clear();
    this->labels.clear();
    this->pose_hypotheses.clear();
    this->histograms.clear();

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

      ::Silhouettef surface_edges = getCannySilhouette(image_grayscale, i_segment.rect);
      if (surface_edges.size() == 0)
        continue;

      this->segments.push_back(i_segment);

      ::PoseRanking ranking;

      // will be reinitialised by first chamfer distance call
      cv::Mat dist_transform(0, 0, CV_32FC1);
      getChamferDistance(silhouette, silhouette, cas_image_depth.size(), dist_transform);

      for (const auto &kv : this->edge_models) {
        auto &mesh = kv.second.mesh;
        assert(mesh.points.size() >= 4);

        #pragma omp parallel for
        for (auto it = kv.second.items.cbegin(); it < kv.second.items.cend(); ++it) {
          ::PoseHypothesis hypothesis;
          hypothesis.model_name = kv.first;

          double pr_fitness_score = 0;
          cv::Mat pr_trans = procrustesTransform(it->contour, silhouette, &pr_fitness_score);

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

          double distance, confidence;
          auto trans_mesh_fp = getFootprint(mesh, pose_3d, world_camera, 64, 4);
          std::tie(distance, confidence) = getChamferDistance(trans_mesh_fp.contour, silhouette, cas_image_depth.size(), dist_transform);

          hypothesis.probability = (1 / (distance + 0.01)) * confidence;
          // hypothesis.probability = (1 / (pr_fitness_score + 0.01));

          #pragma omp critical
          ranking.addElement(hypothesis);
        }
      }

      ranking.filter(0.001); // TODO: find the right confidence level

      if (ranking.size() == 0)
        outInfo("Low probable silhouette");

      auto max_pr = ranking.normalize();

      double norm_confidence_level = 0.90;  // TODO: find the right confidence level

      // plot ranking distribution
      auto ranks = ranking.getTop(100);
      int hsize = ranks.size();
      cv::Mat hist(hsize, hsize, CV_8UC3, cv::Scalar(0, 0, 0));
      int i = 0;
      for (auto &re : ranks) {
        float x = i;
        float y = hsize - re.probability*hsize;
        auto color = (re.probability > norm_confidence_level ? cv::Scalar(0, 128, 0) : cv::Scalar(0, 0, 128));
        cv::line(hist, cv::Point(x, hsize-1), cv::Point(x, y), color, 1);
        i++;
      }
      this->histograms.push_back(hist);

      ranking.filter(norm_confidence_level);
      this->pose_hypotheses.push_back(ranking.getTop(100));

      if (ranking.size() > 0) {
        auto top_hypothesis = ranking.getTop();
        this->labels.push_back(top_hypothesis.model_name);

        outInfo("Found " << ranking.size() << " probable poses");
        outInfo("\tProbably it is " << top_hypothesis.model_name << "; score: " << max_pr);

        outInfo("Surface edges contains: " << surface_edges.size() << " points");
        for (auto &hyp : this->pose_hypotheses.back()) {
          ::Mesh &surface_edge_mesh = this->edge_models[hyp.model_name].edge_mesh;

          ::Pose new_pose;
          double cost = 0;

          outInfo("Running 2d-3d ICP ... ");
          std::tie(new_pose, cost) = ::fit2d3d(surface_edge_mesh, hyp.pose, surface_edges, world_camera);
          outInfo("\tdone: cost = " << cost);

          hyp.pose = new_pose;
          // assert(cost < 1 && cost >= 0);

          break;
        }
      }
      break;
    }

    outInfo("took: " << clock.getTime() << " ms.");
    return UIMA_ERR_NONE;
  }


protected:
  void drawImageWithLock(cv::Mat &disp) override {
    // disp = this->displ;
    // return;
    disp = image_rgb;

    cv::Mat gray;
    cv::cvtColor(image_rgb, gray, CV_BGR2GRAY);

    for (auto seg : this->segments) {
      auto roi = seg.rect;

      constexpr int margin = 5;
      roi.x -= margin;
      roi.y -= margin;
      roi.width += 2*margin;
      roi.height += 2*margin;

      cv::Mat gray_roi = gray(roi);

      // Compute optimal thresh value
      cv::Mat not_used;
      double otsu_thresh_val = cv::threshold(gray_roi, not_used, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
      cv::Canny(gray_roi, gray_roi, otsu_thresh_val/3, otsu_thresh_val);
    }

    cv::cvtColor(gray, disp, CV_GRAY2BGR);

    // ImageSegmentation::drawSegments2D(disp, this->segments, this->labels, 1, 0.5);

    // return;
    // for (const auto &sil : this->fitted_silhouettes) {
    //   for (auto &pt : sil)
    //     cv::circle(disp, pt, 1, cv::Scalar(255, 0, 255), -1);
    // }

    for (size_t i = 0; i < this->labels.size(); ++i) {
      outInfo("draw");
      ::PoseHypothesis hyp;
      if (this->pose_hypotheses[i].size() > 0)
        hyp = this->pose_hypotheses[i].front();
      else
        continue;

      Camera cam;
      drawMesh(disp, cam, this->edge_models[this->labels[i]].mesh, hyp.pose);

      auto &seg_center = segments[i].center;
      cv::Rect hist_dst_rect(cv::Point(seg_center.x, seg_center.y + 5 + segments[i].rect.height/2), histograms[i].size());
      hist_dst_rect = hist_dst_rect & cv::Rect(cv::Point(), disp.size());
      disp(hist_dst_rect) *= 0.5;
      cv::Rect hist_src_rect = (hist_dst_rect - hist_dst_rect.tl()) & cv::Rect(cv::Point(), histograms[i].size());
      disp(hist_dst_rect) += histograms[i](hist_src_rect);
    }
  }

  static std::tuple<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr, std::vector<pcl::Vertices>>
    meshToPCLMesh(const ::Mesh &mesh, const cv::Mat &trans, const float tint) {
    std::vector<pcl::Vertices> polygons;

    for (const auto &tri : mesh.triangles) {
      pcl::Vertices vtc;
      vtc.vertices.push_back(tri[0]);
      vtc.vertices.push_back(tri[1]);
      vtc.vertices.push_back(tri[2]);

      polygons.push_back(vtc);
    }

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud {new pcl::PointCloud<pcl::PointXYZRGBA>};
    cloud->width = mesh.points.size();
    cloud->height = 1;
    cloud->is_dense = false;

    cloud->points.resize(cloud->width * cloud->height);

    for (size_t i = 0; i < mesh.points.size(); ++i) {
      auto pt = transform(trans, mesh.points[i]);
      auto &cpt = cloud->points[i];
      cpt.x = -pt.x;
      cpt.y = -pt.y;
      cpt.z = -pt.z;
      cpt.r = 255*tint;
      cpt.g = 255*tint;
      cpt.b = 255*tint;
      cpt.a = 255;
    }

    return std::tie(cloud, polygons);
  }

  void fillVisualizerWithLock(pcl::visualization::PCLVisualizer &visualizer, const bool firstRun) override
  {
    const std::string &cloudname = "ContourFittingClassifier";
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr outCloud;

    outCloud = this->view_cloud;

    if (firstRun)
      visualizer.addPointCloud(outCloud, cloudname);
    else {
      visualizer.updatePointCloud(outCloud, cloudname);
      visualizer.removeAllShapes();
    }

    int i = 0;
    for (const auto &seg_hyp : this->pose_hypotheses) {
      for (const auto &hyp : seg_hyp) {
        std::string pcl_mesh_name = "_" + std::to_string(i);
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud;
        std::vector<pcl::Vertices> polygons;

        cv::Mat affine_3d_transform = poseToAffine(hyp.pose);
        std::tie(cloud, polygons) = meshToPCLMesh(this->edge_models[hyp.model_name].mesh,
                                                  affine_3d_transform, (hyp.probability - 0.85) / 0.15);

        visualizer.removeShape(pcl_mesh_name);
        assert(visualizer.addPolygonMesh<pcl::PointXYZRGBA>(cloud, polygons, pcl_mesh_name));

        ++i;
      }
    }
  }

  static void drawMesh(cv::Mat &dst, Camera &cam, Mesh &mesh, Pose &pose) {
    std::vector<cv::Point3f> vertice;
    std::vector<cv::Vec3f> normal;

    vertice.reserve(mesh.points.size());
    // normal.reserve(mesh.normals.size());

    assert(vertice.size() == normal.size());

/*    cv::Mat cam_sp_transform = poseToAffine(pose);

    auto rect33 = cv::Rect(0, 0, 3, 3);
    cv::Mat cam_sp_rot = cv::Mat::eye(3, 4, CV_32FC1);
    cam_sp_transform(rect33).copyTo(cam_sp_rot(rect33));*/

    for (const auto &vtx : mesh.points)
      vertice.push_back(vtx);//transform(cam_sp_transform, vtx));

/*    for (const auto &nrm : mesh.normals)
      normal.push_back(transform(cam_sp_rot, nrm));*/

    std::vector<cv::Point2f> vertice_2d;

    cv::Mat draw_cam_matrix = (cv::Mat_<double>(3, 3) << 570.3422241210938, 0.0, 319.5, 0.0, 570.3422241210938, 239.5, 0.0, 0.0, 1.0);
    cv::projectPoints(vertice, pose.rot, pose.trans, draw_cam_matrix, {}, vertice_2d);

    cv::Vec3f light = cv::normalize(cv::Vec3f(1, 1, 1));

/*    float alpha = 0.3f;
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
    }*/

    for (const auto &pt2 : vertice_2d) {
      cv::circle(dst, pt2, 1, cv::Scalar(0, 255, 0), -1);
    }
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
    Silhouettef normalized_contour = normalizeSilhouette(contour);

    cv::Mat mkernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(footprint, footprint, cv::MORPH_GRADIENT, mkernel,
      cv::Point(-1,-1), 1);

    return {footprint, contour, normalized_contour, pose};
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

  static std::pair<cv::Mat, double> fitICP(const ::Silhouettef &test, const ::Silhouettef &model) {

    std::vector<double> test_arr;
    std::vector<double> model_arr;

    for (const auto &pt : test) {
      test_arr.push_back(pt.x);
      test_arr.push_back(pt.y);
    }

    for (const auto &pt : model) {
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

  static double getFitnessScoreN(const ::Silhouettef &a, const ::Silhouettef &b) {
    auto transform__score = fitICP(a, b);

    return transform__score.second;
  }

  std::tuple<std::string, size_t, double> getBestFitnessResultN(const Silhouettef &nshape) {
    std::string best_model_name;
    size_t best_pose_index{0};
    double best_fitness_score{std::numeric_limits<double>::max()};

    for (const auto &kv : this->edge_models) {
      std::vector<double> scores(kv.second.items.size());

      #pragma omp parallel for
      for (int i = 0; i < kv.second.items.size(); ++i) {
        scores[i] = getFitnessScoreN(nshape, kv.second.items[i].normalized_contour);
      }

      auto it = std::min_element(scores.cbegin(), scores.cend());
      if (*it < best_fitness_score) {
        best_fitness_score = *it;
        best_model_name = kv.first;
        best_pose_index = std::distance(scores.cbegin(), it);
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

  static cv::Mat procrustesTransform(const ::Silhouettef &sil, const ::Silhouettef &tmplt, double *fitness_score = nullptr) {
    cv::Vec2f mean_s, mean_t;
    float deviation_s, deviation_t;

    // FIXME: avoid double mean/deviation computation
    std::tie(mean_s, deviation_s) = getMeanAndDev(sil);
    std::tie(mean_t, deviation_t) = getMeanAndDev(tmplt);

    auto ns = normalizeSilhouette(sil);
    auto nt = normalizeSilhouette(tmplt);

    cv::Mat icp_mat;
    double fitness;
    std::tie(icp_mat, fitness) = fitICP(ns, nt);

    if (fitness_score)
      *fitness_score = fitness;

    cv::Mat Ts_inv = (cv::Mat_<float>(3, 3) << 1, 0, -mean_s(0), 0, 1, -mean_s(1), 0 , 0, 1);
    cv::Mat Ss_inv = (cv::Mat_<float>(3, 3) << 1/deviation_s, 0, 0, 0, 1/deviation_s, 0, 0 , 0, 1);
    cv::Mat Rst = icp_mat;
    cv::Mat St = (cv::Mat_<float>(3, 3) << deviation_t, 0, 0, 0, deviation_t, 0, 0 , 0, 1);
    cv::Mat Tt = (cv::Mat_<float>(3, 3) << 1, 0, mean_t(0), 0, 1, mean_t(1), 0 , 0, 1);

    cv::Mat sil_to_tmplt_transformation = Tt * St * Rst * Ss_inv * Ts_inv;

    return sil_to_tmplt_transformation;
  }

  std::pair<double, double> getChamferDistance(::Silhouettef &a, ::Silhouettef &b, cv::Size work_area, cv::Mat &dist_transform) {
    double distance_sum = 0;
    size_t num_points_hit = 0;

    if (dist_transform.cols == 0 || dist_transform.rows == 0) {
      cv::Mat sil_map_b = cv::Mat::ones(work_area, CV_8UC1);
      for (auto it = b.begin(); it != std::prev(b.end()); it = std::next(it))
        cv::line(sil_map_b, *it, *std::next(it), cv::Scalar(0));
      cv::line(sil_map_b, b.back(), b.front(), cv::Scalar(0));

      // auto rect_a = ::getBoundingRect(a);
      // auto rect_b = ::getBoundingRect(b);
      // cv::Rect roi = (rect_a | rect_b) & cv::Rect_<float>(cv::Point(0, 0), work_area);

      dist_transform = cv::Mat(sil_map_b.size(), CV_32FC1, cv::Scalar(0.f));
      // cv::distanceTransform(sil_map_b(roi), dist_transform(roi), CV_DIST_L2, 3);
      cv::distanceTransform(sil_map_b, dist_transform, CV_DIST_L2, 3);
    }

    auto work_rect = cv::Rect(cv::Point(0, 0), work_area);
    for (const auto &pt : a) {
      auto pti = cv::Point(pt.x, pt.y);
      if (work_rect.contains(pti)) {
        distance_sum += dist_transform.at<float>(pti);
        num_points_hit += 1;
      }
    }

    double confidence = static_cast<double>(num_points_hit) / a.size();

    if (confidence == 0)
      throw std::runtime_error("input contour is too large or contains no points");

    double distance = distance_sum / num_points_hit;

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
  std::vector<std::vector<::PoseHypothesis>> pose_hypotheses;
  // std::vector<Silhouettef> fitted_silhouettes;
  std::vector<std::string> labels;
  std::vector<cv::Mat> histograms;

  cv::Mat image_rgb;
  cv::Mat distance_mat = cv::Mat(480, 640, CV_8UC3);

  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr view_cloud {new pcl::PointCloud<pcl::PointXYZRGBA>};

  ::Camera silhouette_camera;
};

cv::Point2f transform(const cv::Mat &M, const cv::Point2f &pt) {
  cv::Mat vec(3, 1, CV_32FC1);

  vec.at<float>(0, 0) = pt.x;
  vec.at<float>(1, 0) = pt.y;
  vec.at<float>(2, 0) = 1.f;

  cv::Mat dst = M * vec;

  return cv::Point2f(dst.at<float>(0, 0), dst.at<float>(1, 0));
}

cv::Point3f transform(const cv::Mat &M, const cv::Point3f &pt) {
  cv::Mat vec(4, 1, CV_32FC1);

  vec.at<float>(0, 0) = pt.x;
  vec.at<float>(1, 0) = pt.y;
  vec.at<float>(2, 0) = pt.z;
  vec.at<float>(3, 0) = 1.f;

  cv::Mat dst = M * vec;

  return cv::Point3f(dst.at<float>(0, 0), dst.at<float>(1, 0), dst.at<float>(2, 0));
}

cv::Vec3f transform(const cv::Mat &M, const cv::Vec3f &vec) {
  cv::Point3f pt(vec[0], vec[1], vec[2]);

  return transform(M, pt);
}

::Silhouettef transform(const cv::Mat &M, const ::Silhouettef &sil) {
  Silhouettef result;

  for (const auto &pt : sil)
    result.push_back(transform(M, pt));

  return result;
}

::Silhouettef transform(const cv::Mat &M, const ::Silhouette &sil) {
  Silhouettef result;

  for (const auto &pt : sil) {
    cv::Point2f ptf(pt.x, pt.y);

    result.push_back(transform(M, ptf));
  }

  return result;
}

std::vector<cv::Point3f> transform(const cv::Mat &M, const std::vector<cv::Point3f> &points) {
  std::vector<cv::Point3f> result;
  result.reserve(points.size());

  for (const auto &pt : points)
    result.push_back(transform(M, pt));

  return result;
}

::Silhouettef normalizeSilhouette(const ::Silhouettef &shape) {
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

cv::Rect_<float> getBoundingRect(const ::Silhouettef &sil) {
  cv::Rect_<float> b_rect;

  auto h_it = std::minmax_element(sil.cbegin(), sil.cend(),
    [](const cv::Point2f &a, const cv::Point2f &b) {
      return a.x < b.x;});
  auto v_it = std::minmax_element(sil.cbegin(), sil.cend(),
    [](const cv::Point2f &a, const cv::Point2f &b) {
      return a.y < b.y;});

  b_rect.x = h_it.first->x;
  b_rect.y = v_it.first->y;
  b_rect.width = h_it.second->x - b_rect.x;
  b_rect.height = v_it.second->y - b_rect.y;

  return b_rect;
}

cv::Mat poseToAffine(const ::Pose &pose) {
  cv::Mat affine_3d_transform(3, 4, CV_32FC1);

  cv::Rodrigues(pose.rot, affine_3d_transform.colRange(0, 3));
  affine_3d_transform.at<float>(0, 3) = pose.trans(0);
  affine_3d_transform.at<float>(1, 3) = pose.trans(1);
  affine_3d_transform.at<float>(2, 3) = pose.trans(2);

  return affine_3d_transform;
}

::Pose operator+(const ::Pose &a, const Pose &b) {
  ::Pose result;

  result.rot = a.rot + b.rot;
  result.trans = a.trans + b.trans;

  return result;
}

::Pose operator*(const double a, const Pose &b) {
  ::Pose result;

  result.rot = b.rot * a;
  result.trans = b.trans * a;

  return result;
}

::Silhouettef projectSurfacePoints(::Mesh &mesh, ::Pose &pose, ::Camera &camera) {
  ::Silhouettef points_2d;
  cv::projectPoints(mesh.points, pose.rot, pose.trans, camera.matrix, camera.ks, points_2d);

  return points_2d;
}

pcl::KdTreeFLANN<pcl::PointXY> getKdTree(const ::Silhouettef &sil) {
  pcl::PointCloud<pcl::PointXY>::Ptr input_cloud {new pcl::PointCloud<pcl::PointXY>};

  input_cloud->width = sil.size();
  input_cloud->height = 1;
  input_cloud->is_dense = false;

  input_cloud->points.resize(input_cloud->width * input_cloud->height);

  for(size_t i = 0; i < input_cloud->size(); ++i) {
    input_cloud->points[i] = {sil[i].x, sil[i].y};
  }

  pcl::KdTreeFLANN<pcl::PointXY> kdtree;

  kdtree.setInputCloud(input_cloud);

  return kdtree;
}

cv::Point2f getNearestPoint(pcl::KdTree<pcl::PointXY> &template_kdtree, const cv::Point2f &pt) {
  pcl::PointXY search_point = {pt.x, pt.y};

  std::vector<int> indices;
  std::vector<float> l2_sqr_distances;

  assert(template_kdtree.nearestKSearch(search_point, 1, indices, l2_sqr_distances) == 1);

  auto cloud = template_kdtree.getInputCloud();
  auto out_pt = cloud->points[indices.front()];

  return cv::Point2f(out_pt.x, out_pt.y);
}

std::tuple<cv::Mat, cv::Mat> computeResidualsAndWeights(const ::Silhouettef &data, const ::Silhouettef &template_2d) {
  cv::Mat residuals(data.size(), 1, CV_32FC1);
  cv::Mat weights(data.size(), 1, CV_32FC1);

  auto template_kdtree = getKdTree(template_2d);

  int i = 0;
  for (const auto &pt : data) {
    auto template_pt = getNearestPoint(template_kdtree, pt);

    float distance = cv::norm(template_pt - pt);
    residuals.at<float>(i, 0) = distance;
    weights.at<float>(i, 0) = (distance <= 10); // or how do we check if point matches???

    i++;
  }

  return std::tie(residuals, weights);
}

template<class T, class Compare>
const T &clamp(const T &v, const T &lo, const T &hi, Compare comp) {
    return assert(!comp(hi, lo)),
        comp(v, lo) ? lo : comp(hi, v) ? hi : v;
}

template<class T>
const T &clamp(const T &v, const T &lo, const T &hi) {
    return clamp(v, lo, hi, std::less<T>());
}

cv::Mat computeJacobian(::Pose &pose, ::Mesh &mesh, float h, ::Silhouettef &template_2d, cv::Mat &weights, ::Camera &camera) {
  size_t dof = 6;

  std::vector<::Pose> pose_h_plus;
  std::vector<::Pose> pose_h_minus;

  Pose tmp = pose + Pose{{h, 0, 0}, {0, 0, 0}};
  pose_h_plus.push_back(tmp);
  tmp = pose + Pose{{-h, 0, 0}, {0, 0, 0}};
  pose_h_minus.push_back(tmp);

  tmp = pose + Pose{{0, h, 0}, {0, 0, 0}};
  pose_h_plus.push_back(tmp);
  tmp = pose + Pose{{0, -h, 0}, {0, 0, 0}};
  pose_h_minus.push_back(tmp);

  tmp = pose + Pose{{0, 0, h}, {0, 0, 0}};
  pose_h_plus.push_back(tmp);
  tmp = pose + Pose{{0, 0, -h}, {0, 0, 0}};
  pose_h_minus.push_back(tmp);

  tmp = pose + Pose{{0, 0, 0}, {h, 0, 0}};
  pose_h_plus.push_back(tmp);
  tmp = pose + Pose{{0, 0, 0}, {-h, 0, 0}};
  pose_h_minus.push_back(tmp);

  tmp = pose + Pose{{0, 0, 0}, {0, h, 0}};
  pose_h_plus.push_back(tmp);
  tmp = pose + Pose{{0, 0, 0}, {0, -h, 0}};
  pose_h_minus.push_back(tmp);

  tmp = pose + Pose{{0, 0, 0}, {0, 0, h}};
  pose_h_plus.push_back(tmp);
  tmp = pose + Pose{{0, 0, 0}, {0, 0, -h}};
  pose_h_minus.push_back(tmp);

  Silhouettef d_ref = projectSurfacePoints(mesh, pose, camera);

  auto template_kd = getKdTree(template_2d);

  cv::Mat jacobian(mesh.points.size(), dof, CV_32FC1);
  for (size_t j = 0; j < dof; ++j) {
    Silhouettef d_plus = projectSurfacePoints(mesh, pose_h_plus[j], camera);
    Silhouettef d_minus = projectSurfacePoints(mesh, pose_h_minus[j], camera);

    for (size_t i = 0; i < d_plus.size(); ++i) {
      auto d_i_plus = getNearestPoint(template_kd, d_plus[i]);
      auto d_i_minus = getNearestPoint(template_kd, d_minus[i]);

      float dei_daj = weights.at<float>(i, 0) * (cv::norm(d_i_plus - d_ref[i]) - cv::norm(d_i_minus - d_ref[i])) / (2 * h);

      // assert(dei_daj < 1000);
      // assert(dei_daj > -1000);

      jacobian.at<float>(i, j) = clamp(dei_daj, -1000.f, 1000.f);
    }
  }

  return jacobian;
}

std::tuple<::Pose, double> fit2d3d(::Mesh &mesh, ::Pose &init_pose, ::Silhouettef &template_2d, ::Camera &camera) {
  ::Pose current_pose = init_pose;
  float lambda = 1e-2f;

  double cost = std::numeric_limits<double>::max();

  outInfo("Trace 1");

  bool done {false};
  while (!done) {
    // outInfo("Trace 2");
    Silhouettef sil_2d = projectSurfacePoints(mesh, current_pose, camera);

    // outInfo("Trace 3");
    cv::Mat residuals;
    cv::Mat weights;
    std::tie(residuals, weights) = computeResidualsAndWeights(sil_2d, template_2d);

    // outInfo("Trace 4");
    auto h = 1e-3;
    cv::Mat jacobian = computeJacobian(current_pose, mesh, h, template_2d, weights, camera);
    if (cv::countNonZero(jacobian) == 0) {
      outInfo("Already at best approximation, or `h` is too small");
      cost = cv::norm(residuals, cv::NORM_L2);
      break;
    }
    // outInfo("Jacobian: " << jacobian);

    // outInfo("Trace 5");
    cv::Mat pinv_jacobian = jacobian.inv(cv::DECOMP_SVD);
    cv::Mat delta_pose_mat = pinv_jacobian * residuals;

    // outInfo("Trace 6");
    ::Pose delta_pose;
    delta_pose.rot = cv::Vec3f(delta_pose_mat.at<float>(0, 0), delta_pose_mat.at<float>(1, 0), delta_pose_mat.at<float>(2, 0));
    delta_pose.trans = cv::Vec3f(delta_pose_mat.at<float>(3, 0), delta_pose_mat.at<float>(4, 0), delta_pose_mat.at<float>(5, 0));

    // outInfo("Trace 7");
    current_pose = current_pose + lambda*delta_pose;

    // outInfo("Trace 8");
    cost = cv::norm(residuals, cv::NORM_L2);
    done = (cost < 1e-3);

    outInfo("cost: " << cost);
  }

  return std::tie(current_pose, cost);
}

::Silhouettef getCannySilhouette(cv::Mat &grayscale, cv::Rect &input_roi) {
  auto roi = input_roi;

  constexpr int margin = 5;
  roi.x -= margin;
  roi.y -= margin;
  roi.width += 2*margin;
  roi.height += 2*margin;

  cv::Mat gray_roi = grayscale(roi);

  // Compute optimal thresh value
  cv::Mat not_used;
  double otsu_thresh_val = cv::threshold(gray_roi, not_used, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
  cv::Canny(gray_roi, gray_roi, otsu_thresh_val/3, otsu_thresh_val);

  ::Silhouettef result;

  outInfo("non Zero " << cv::countNonZero(gray_roi));

  for(int row = 0; row < gray_roi.rows; ++row) {
    uint8_t *ptr = gray_roi.ptr(row);
    for(int col = 0; col < gray_roi.cols; ++col, ++ptr) {
      if (*ptr != 0)
        result.push_back(cv::Point2f(row + roi.x, col + roi.y));
    }
  }

  return result;
}

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(ContourFittingClassifier)