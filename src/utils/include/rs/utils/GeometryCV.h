
namespace GeometryCV {
  pcl::KdTreeFLANN<pcl::PointXY> getKdTree(const std::vector<cv::Point2f> &sil) {
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

  std::tuple<cv::Mat, cv::Mat> compute2dDisparityResidualsAndWeights(const std::vector<cv::Point2f> &data, pcl::KdTree<pcl::PointXY> &template_kdtree) {
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
  }

  std::tuple<::PoseRT, double, cv::Mat> fit2d3d(::Mesh &mesh, const ::PoseRT &init_pose,
      const std::vector<cv::Point2f> &template_2d, const ::Camera &camera, const size_t limit_iterations, cv::Vec3f normal_constraint = cv::Vec3f(0, 0, 0)) {
    ::PoseRT current_pose = init_pose;
    float learning_rate = 2;
    size_t limit_iterations = 50;
    double limit_epsilon = 1e-5;
    size_t stall_counter = 0;
    double h = 1e-3;

    double last_error = 0;
    cv::Mat jacobian;

    auto template_kdtree = getKdTree(template_2d);

    bool done {false};
    while (!done && limit_iterations) {

      std::vector<cv::Point2f> sil_2d = projectSurfacePoints(mesh, current_pose, camera);

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
          if (limit_iterations != 1)
            this->surface_edges_blue.push_back(sil_2d);
        }
        stall_counter++;
      }
      else
        stall_counter = 0;

      last_error = ref_error;

      jacobian = computeProximityJacobianForPoseRT(current_pose, mesh, h, template_2d, weights, camera);
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

      // check if normal constraint present
      if (cv::norm(normal_constraint) > 0.1)
        current_pose.rot = alignRotationWithPlane(current_pose.rot, normal_constraint);

      --limit_iterations;
    }

    return std::tie(current_pose, last_error, jacobian);
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

  cv::Mat poseRTToAffine(const ::PoseRT &pose) {
    cv::Mat affine_3d_transform(3, 4, CV_32FC1);

    cv::Rodrigues(pose.rot, affine_3d_transform.colRange(0, 3));
    affine_3d_transform.at<float>(0, 3) = pose.trans(0);
    affine_3d_transform.at<float>(1, 3) = pose.trans(1);
    affine_3d_transform.at<float>(2, 3) = pose.trans(2);

    return affine_3d_transform;
  }

  std::vector<cv::Point2f> projectSurfacePoints(::Mesh &mesh, ::PoseRT &pose, ::Camera &camera) {
    std::vector<cv::Point2f> points_2d;
    cv::projectPoints(mesh.points, pose.rot, pose.trans, camera.matrix, camera.distortion, points_2d);

    return points_2d;
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

  cv::Mat computeProximityJacobianForPoseRT(::PoseRT &pose, ::Mesh &mesh, float h, std::vector<cv::Point2f> &template_2d, cv::Mat &weights, ::Camera &camera) {
    size_t dof = 6;

    auto template_kd = getKdTree(template_2d);

    cv::Mat jacobian(mesh.points.size(), dof, CV_32FC1);
    for (size_t j = 0; j < dof; ++j) {
      ::PoseRT pose_h_plus  = offsetPose(pose, j, h);
      ::PoseRT pose_h_minus = offsetPose(pose, j, -h);

      std::vector<cv::Point2f> d_plus  = projectSurfacePoints(mesh, pose_h_plus, camera);
      std::vector<cv::Point2f> d_minus = projectSurfacePoints(mesh, pose_h_minus, camera);

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
}