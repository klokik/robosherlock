#pragma once

#include <pcl/point_types.h>
// #include <pcl/search/kdtree.h>


/*#define PCL_SEGFAULT_WORKAROUND 0

#if !PCL_SEGFAULT_WORKAROUND
#include <pcl/registration/icp.h>
#else
#include "libicp/src/icpPointToPoint.h"
#endif*/



struct PoseRT {
  cv::Vec3f rot;
  cv::Vec3f trans;
};

class Camera {
  public: cv::Mat matrix = cv::Mat::eye(3, 3, CV_32FC1);
  public: std::vector<float> distortion;

  public: void setFromMsgs(const sensor_msgs::CameraInfo &camInfo) {
    float *it = this->matrix.ptr<float>(0);
    for (size_t i = 0; i < 9; ++i, ++it)
      *it = camInfo.K[i];

    distortion.clear();
    for(size_t i = 0; i < camInfo.D.size(); ++i)
      this->distortion.push_back(camInfo.D[i]);
  }
};


namespace GeometryCV {

  cv::Mat poseRTToAffine(const ::PoseRT &pose) {
    cv::Mat affine_3d_transform(3, 4, CV_32FC1);

    cv::Rodrigues(pose.rot, affine_3d_transform.colRange(0, 3));
    affine_3d_transform.at<float>(0, 3) = pose.trans(0);
    affine_3d_transform.at<float>(1, 3) = pose.trans(1);
    affine_3d_transform.at<float>(2, 3) = pose.trans(2);

    return affine_3d_transform;
  }

  ::PoseRT operator+(const ::PoseRT &a, const ::PoseRT &b) {
    ::PoseRT result;

    result.rot = a.rot + b.rot;
    result.trans = a.trans + b.trans;

    return result;
  }

  ::PoseRT operator*(const double a, const ::PoseRT &b) {
    ::PoseRT result;

    result.rot = b.rot * a;
    result.trans = b.trans * a;

    return result;
  }

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

  cv::Vec3f transform(const ::PoseRT &pose, const cv::Vec3f &vec) {
    cv::Mat M = poseRTToAffine(pose);

    return transform(M, vec);
  }

  std::vector<cv::Point2f> transform(const cv::Mat &M, const std::vector<cv::Point2f> &pts) {
    std::vector<cv::Point2f> result;

    for (const auto &pt : pts)
      result.push_back(transform(M, pt));

    return result;
  }

  std::vector<cv::Point2f> transform(const cv::Mat &M, const std::vector<cv::Point2i> &pts) {
    std::vector<cv::Point2f> result;

    for (const auto &pt : pts) {
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

  std::vector<cv::Vec3f> transform(const cv::Mat &M, const std::vector<cv::Vec3f> &points) {
    std::vector<cv::Vec3f> result;
    result.reserve(points.size());

    for (const auto &pt : points)
      result.push_back(transform(M, pt));

    return result;
  }

  cv::Rect_<float> getBoundingRect(const std::vector<cv::Point2f> &pts) {
    cv::Rect_<float> b_rect;

    auto h_it = std::minmax_element(pts.cbegin(), pts.cend(),
      [](const cv::Point2f &a, const cv::Point2f &b) {
        return a.x < b.x;});
    auto v_it = std::minmax_element(pts.cbegin(), pts.cend(),
      [](const cv::Point2f &a, const cv::Point2f &b) {
        return a.y < b.y;});

    b_rect.x = h_it.first->x;
    b_rect.y = v_it.first->y;
    b_rect.width = h_it.second->x - b_rect.x;
    b_rect.height = v_it.second->y - b_rect.y;

    return b_rect;
  }

/*  pcl::search::KdTree<pcl::PointXY> getKdTree(const std::vector<cv::Point2f> &sil) {
    pcl::PointCloud<pcl::PointXY>::Ptr input_cloud {new pcl::PointCloud<pcl::PointXY>};

    input_cloud->width = sil.size();
    input_cloud->height = 1;
    input_cloud->is_dense = false;

    input_cloud->points.resize(input_cloud->width * input_cloud->height);

    for(size_t i = 0; i < input_cloud->size(); ++i) {
      input_cloud->points[i] = {sil[i].x, sil[i].y};
    }

    pcl::search::KdTree<pcl::PointXY> kdtree(false);

    kdtree.setInputCloud(input_cloud);

    return kdtree;
  }

  cv::Point2f getNearestPoint(pcl::search::KdTree<pcl::PointXY> &template_kdtree, const cv::Point2f &pt) {
    pcl::PointXY search_point = {pt.x, pt.y};

    std::vector<int> indices;
    std::vector<float> l2_sqr_distances;

    assert(template_kdtree.nearestKSearch(search_point, 1, indices, l2_sqr_distances) == 1);

    auto cloud = template_kdtree.getInputCloud();
    auto out_pt = cloud->points[indices.front()];

    return cv::Point2f(out_pt.x, out_pt.y);
  }

  std::tuple<cv::Mat, cv::Mat> compute2dDisparityResidualsAndWeights(const std::vector<cv::Point2f> &data, pcl::search::KdTree<pcl::PointXY> &template_kdtree) {
    cv::Mat residuals(data.size(), 1, CV_32FC1);
    cv::Mat weights(data.size(), 1, CV_32FC1);

    int i = 0;
    for (const auto &pt : data) {
      auto template_pt = getNearestPoint(template_kdtree, pt);

      float distance = cv::norm(template_pt - pt);
      residuals.at<float>(i, 0) = distance * distance;
      weights.at<float>(i, 0) = (distance <= 5); // or how do we check if point matches???

      i++;
    }

    return std::tie(residuals, weights);
  }*/

  std::vector<cv::Point2f> projectPoints(const std::vector<cv::Point3f> &points, const ::PoseRT &pose, const ::Camera &camera) {
    std::vector<cv::Point2f> points_2d;
    cv::projectPoints(points, pose.rot, pose.trans, camera.matrix, camera.distortion, points_2d);

    return points_2d;
  }

  cv::Vec3f projectRotationOnAxis(const cv::Vec3f &rodrigues, const cv::Vec3f &axis) {
    // find axes rotation transformation to align object's up to plane normal
    cv::Vec3f objects_up_local(0, 1, 0);
    ::PoseRT object_to_camera_rotation {rodrigues, cv::Vec3f(0, 0, 0)};
    auto objects_up_camspace = GeometryCV::transform(object_to_camera_rotation, objects_up_local);

    double phi = std::acos(axis.ddot(objects_up_camspace));

    cv::Vec3f up_to_n_rot_axis = objects_up_camspace.cross(axis);
    cv::Vec3f rodrigues_up_to_n = up_to_n_rot_axis * (phi / cv::norm(up_to_n_rot_axis));

    cv::Mat up_to_n_rotation;
    cv::Rodrigues(rodrigues_up_to_n, up_to_n_rotation);


    cv::Mat initial_rotation;
    cv::Rodrigues(rodrigues, initial_rotation);
    cv::Mat final_rotation = up_to_n_rotation * initial_rotation;

    cv::Mat result;
    cv::Rodrigues(final_rotation, result);

    return result;
  }

  ::PoseRT offsetPose(const ::PoseRT &input, const int dof_id, const float offset) {
    ::PoseRT pose_delta {{0, 0, 0}, {0, 0, 0}};

    switch (dof_id / 3) {
    case 0:
      pose_delta.rot[dof_id] += offset;
      break;

    case 1:
      pose_delta.trans[dof_id % 3] += offset;
      break;

    default:
      outError("Invalid dof_id: " << dof_id << ", acceptable range is [1-6]");
    }

    return input + pose_delta;
  }

/*  cv::Mat computeProximityJacobianForPoseRT(const ::PoseRT &pose, const std::vector<cv::Point3f> &points_3d, const float h, const std::vector<cv::Point2f> &template_2d, const cv::Mat &weights, const ::Camera &camera) {
    size_t dof = 6;

    auto template_kd = getKdTree(template_2d);

    cv::Mat jacobian(points_3d.size(), dof, CV_32FC1);
    for (size_t j = 0; j < dof; ++j) {
      ::PoseRT pose_h_plus  = offsetPose(pose, j, h);
      ::PoseRT pose_h_minus = offsetPose(pose, j, -h);

      std::vector<cv::Point2f> d_plus  = projectPoints(points_3d, pose_h_plus, camera);
      std::vector<cv::Point2f> d_minus = projectPoints(points_3d, pose_h_minus, camera);

      #pragma omp parallel for
      for (size_t i = 0; i < d_plus.size(); ++i) {
        auto d_i_plus = getNearestPoint(template_kd, d_plus[i]);
        auto d_i_minus = getNearestPoint(template_kd, d_minus[i]);

        double d1 = std::pow(cv::norm(d_i_plus - d_plus[i]), 2);
        double d2 = std::pow(cv::norm(d_i_minus - d_minus[i]), 2);

        float dei_daj = weights.at<float>(i) * (d1 - d2) / (2 * h);

        jacobian.at<float>(i, j) = dei_daj;
      }
    }

    return jacobian;
  }

  std::tuple<::PoseRT, double, cv::Mat> fit2d3d(const std::vector<cv::Point3f> &points, const ::PoseRT &init_pose,
      const std::vector<cv::Point2f> &template_2d, const ::Camera &camera, const size_t iterations_limit, cv::Vec3f normal_constraint = cv::Vec3f(0, 0, 0)) {
    ::PoseRT current_pose = init_pose;
    float learning_rate = 2;
    double limit_epsilon = 1e-5;
    size_t iterations_left = iterations_limit;
    size_t stall_counter = 0;
    double h = 1e-3;

    double last_error = 0;
    cv::Mat jacobian;

    auto template_kdtree = getKdTree(template_2d);

    bool done {false};
    while (!done && iterations_left) {

      std::vector<cv::Point2f> sil_2d = projectPoints(points, current_pose, camera);

      cv::Mat residuals;
      cv::Mat weights;
      std::tie(residuals, weights) = compute2dDisparityResidualsAndWeights(sil_2d, template_kdtree);

      double num {0};
      double sum {0};
      for (int i = 0; i < residuals.rows; ++i) {
        if (residuals.at<float>(i) < 0.9f) {
          sum += residuals.at<float>(i);
          num++;
        }
      }

      double ref_error = std::numeric_limits<double>::max();
      if (num != 0)
        ref_error = std::sqrt(sum) / num;

      outInfo("ref_error: " << ref_error << " (" << num << "/" << weights.rows << ")") ;

      if (std::abs(last_error - ref_error) < limit_epsilon) {
        if (stall_counter == 5) {
          outInfo("Done");
          done = true;
        }
        stall_counter++;
      }
      else
        stall_counter = 0;

      last_error = ref_error;

      jacobian = computeProximityJacobianForPoseRT(current_pose, points, h, template_2d, weights, camera);
      if (cv::countNonZero(jacobian) == 0 || cv::sum(weights)[0] == 0) {
        outInfo("Already at best approximation, or `h` is too small");
        ref_error = cv::norm(residuals, cv::NORM_L2);
        break;
      }
      // jacobian = jacobian / cv::norm(jacobian, cv::NORM_INF);
      // outInfo("Jacobian: " << jacobian);

      cv::Mat delta_pose_mat;
      cv::solve(jacobian, residuals, delta_pose_mat, cv::DECOMP_SVD);

      ::PoseRT delta_pose;
      delta_pose.rot = cv::Vec3f(delta_pose_mat.at<float>(0), delta_pose_mat.at<float>(1), delta_pose_mat.at<float>(2));
      delta_pose.trans = cv::Vec3f(delta_pose_mat.at<float>(3), delta_pose_mat.at<float>(4), delta_pose_mat.at<float>(5));


      current_pose = current_pose + (-1 * learning_rate) * delta_pose;

      // apply up-direction constraint if present
      if (cv::norm(normal_constraint) > 0.1)
        current_pose.rot = projectRotationOnAxis(current_pose.rot, normal_constraint);

      --iterations_left;
    }

    return std::tie(current_pose, last_error, jacobian);
  }*/

  std::pair<double, double> getChamferDistance(std::vector<cv::Point2f> &a, std::vector<cv::Point2f> &b, cv::Size work_area, cv::Mat &dist_transform) {
    double distance_sum = 0;
    size_t num_points_hit = 0;

    if (dist_transform.empty()) {
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

    double distance;

    if (confidence != 0)
      distance = distance_sum / num_points_hit;
    else {
      outWarn("input contour is too large or contains no points");
      distance = std::numeric_limits<double>::max();
    }

    return std::make_pair(distance, confidence);
  }

  std::vector<cv::Point2f> normalizePoints(const std::vector<cv::Point2f> &pts) {
    std::vector<cv::Point2f> result;
    for (const auto &pt : pts)
      result.push_back(pt);

    cv::Point2f mean = std::accumulate(pts.cbegin(), pts.cend(), cv::Point2f());
    mean *= (1.f / pts.size());

    float std_dev = 0;

    for (auto &pt : result) {
      pt = pt - mean;
      std_dev += std::pow(cv::norm(pt), 2);
    }

    std_dev = std::sqrt(std_dev / pts.size());

    for (auto &pt : result)
      pt *= 1.f / std_dev;

    return result;
  }

#if !PCL_SEGFAULT_WORKAROUND
  void vectorToPointCloud(const std::vector<cv::Point2f> &points, pcl::PointCloud<pcl::PointXYZ> &pc) {
    pc.width = points.size();
    pc.height = 1;
    pc.is_dense = false;

    pc.resize(pc.width * pc.height);

    for (size_t i = 0; i < points.size(); ++i) {
      pc.points[i] = {points[i].x, points[i].y, 0};
    }
  }

  void PointCloudToVector(pcl::PointCloud<pcl::PointXYZ> &pc, std::vector<cv::Point2f> &points) {
    points.clear();

    assert(pc.height == 1);

    for (size_t i = 0; i < pc.width; ++i) {
      points.push_back(cv::Point2f(pc.points[i].x, pc.points[i].y));
    }
  }

  std::pair<cv::Mat, double> fitICP(const std::vector<cv::Point2f> &test,const std::vector<cv::Point2f> &model) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cl_test(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cl_model(new pcl::PointCloud<pcl::PointXYZ>);

    vectorToPointCloud(test, *cl_test);
    vectorToPointCloud(model, *cl_model);

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(cl_test);
    icp.setInputTarget(cl_model);

    pcl::PointCloud<pcl::PointXYZ> cl_result;

    icp.align(cl_result);

    assert(icp.hasConverged());

    double score = icp.getFitnessScore();


    Eigen::Matrix4f eig_trf = icp.getFinalTransformation ();

    cv::Mat cv_trf(3, 3, CV_32FC1, cv::Scalar(0.f));
    cv_trf.at<float>(0, 0) = eig_trf(0, 0);
    cv_trf.at<float>(0, 1) = eig_trf(0, 1);
    cv_trf.at<float>(1, 0) = eig_trf(1, 0);
    cv_trf.at<float>(1, 1) = eig_trf(1, 1);

    // assume that translation is small enough
    cv_trf.at<float>(0, 2) = 0; //eig_trf(0, 3);
    cv_trf.at<float>(1, 2) = 0; //eig_trf(1, 3);
    cv_trf.at<float>(2, 2) = 1.f;

    return {cv_trf, score};
  }

#else

  std::pair<cv::Mat, double> fitICP(const std::vector<cv::Point2f> &test, const std::vector<cv::Point2f> &model) {

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

  std::pair<cv::Vec2f, float> getMeanAndStdDev(const std::vector<cv::Point2f> &sil) {
    cv::Point2f mean = std::accumulate(sil.cbegin(), sil.cend(), cv::Point2f());
    mean *= (1.f / sil.size());

    float std_dev = 0;
    for (auto &pt : sil)
      std_dev += std::pow(cv::norm(cv::Point2f(pt.x, pt.y) - mean), 2);

    std_dev = std::sqrt(std_dev / sil.size());

    return std::make_pair(mean, std_dev);
  }

  cv::Mat fitProcrustes2d(const std::vector<cv::Point2f> &sil, const std::vector<cv::Point2f> &tmplt, double *fitness_score = nullptr) {
    cv::Vec2f mean_s, mean_t;
    float deviation_s, deviation_t;

    // FIXME: avoid double mean/deviation computation
    std::tie(mean_s, deviation_s) = getMeanAndStdDev(sil);
    std::tie(mean_t, deviation_t) = getMeanAndStdDev(tmplt);

    auto ns = normalizePoints(sil);
    auto nt = normalizePoints(tmplt);

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
}