#pragma once

#include <rs/utils/GeometryCV.h>

namespace Drawing {
  constexpr uint16_t max_depth_16u = std::numeric_limits<uint16_t>::max();
  constexpr float max_depth_32f = std::numeric_limits<float>::max();

  template <typename C_t>
  void drawTriangleInterp(cv::Mat &z_buffer, cv::Mat &dst, const std::vector<cv::Point3f> &poly, const std::vector<C_t> &vals) {
    int min_x = std::max(0, (int)std::floor(std::min(std::min(poly[0].x, poly[1].x), poly[2].x)));
    int min_y = std::max(0, (int)std::floor(std::min(std::min(poly[0].y, poly[1].y), poly[2].y)));
    int max_x = std::min(z_buffer.cols, (int)std::ceil(std::max(std::max(poly[0].x, poly[1].x), poly[2].x)));
    int max_y = std::min(z_buffer.rows, (int)std::ceil(std::max(std::max(poly[0].y, poly[1].y), poly[2].y)));

    cv::Mat T = (cv::Mat_<float>(2, 2) <<
      (poly[0].x - poly[2].x), (poly[1].x - poly[2].x),
      (poly[0].y - poly[2].y), (poly[1].y - poly[2].y));
    cv::Mat iT = T.inv();
    cv::Mat r = (cv::Mat_<float>(2, 1));

    for (int j = min_y; j < max_y; ++j)
      for (int i = min_x; i < max_x; ++i) {
        r.at<float>(0) = i - poly[2].x;
        r.at<float>(1) = j - poly[2].y;

        cv::Mat bc = iT * r;

        float l1 = bc.at<float>(0);
        float l2 = bc.at<float>(1);
        float l3 = 1 - l1 - l2;

        if (l1 >=0 && l2 >=0 && l3 >=0 && l1 <=1 && l2 <=1 && l3 <=1) {
          float new_depth = std::abs(poly[0].z*l1 + poly[1].z*l2 + poly[2].z*l3);

          if (z_buffer.at<float>(j, i) > new_depth) {
            z_buffer.at<float>(j, i) = new_depth;

            if (vals.size()) {
              C_t new_val = l1*vals[0] + l2*vals[1] + l3*vals[2];
              dst.at<C_t>(j, i) = new_val;
            }
          }
        }
      }
  }

  void drawMeshDepth(
      cv::Mat &dst_32fc1,
      const std::vector<cv::Point3f> &points,
      const std::vector<std::vector<int>> &indices,
      const cv::Vec3f rot,
      const cv::Vec3f trans,
      const cv::Mat &cam,
      const std::vector<float> &ks) {
    cv::Mat cam_sp_transform = GeometryCV::poseRTToAffine({rot, trans});

    std::vector<cv::Point3f> vertice = GeometryCV::transform(cam_sp_transform, points);

    std::vector<cv::Point2f> vertice_2d;
    cv::projectPoints(vertice, cv::Vec3f(0, 0, 0), cv::Vec3f(0, 0, 0), cam, ks, vertice_2d);

    cv::Mat none;
    std::vector<float> vals;

    for (const auto &tri : indices) {
      std::vector<float> depth {
        vertice[tri[0]].z,
        vertice[tri[1]].z,
        vertice[tri[2]].z};

      std::vector<cv::Point3f> poly{
          cv::Vec3f(vertice_2d[tri[0]].x, vertice_2d[tri[0]].y, depth[0]),
          cv::Vec3f(vertice_2d[tri[1]].x, vertice_2d[tri[1]].y, depth[1]),
          cv::Vec3f(vertice_2d[tri[2]].x, vertice_2d[tri[2]].y, depth[2])};

      drawTriangleInterp(dst_32fc1, none, poly, vals);
    }
  }

  void drawMeshNormals(
      cv::Mat &dst_depth_32fc1,
      cv::Mat &dst_32fc3,
      const std::vector<cv::Point3f> &points,
      const std::vector<cv::Vec3f> &normals,
      const std::vector<std::vector<int>> &indices,
      const cv::Vec3f rot,
      const cv::Vec3f trans,
      const cv::Mat &cam,
      const std::vector<float> &ks,
      const bool flat = false) {
    cv::Mat cam_sp_transform = GeometryCV::poseRTToAffine({rot, trans});
    cv::Mat cam_sp_transform_rot = GeometryCV::poseRTToAffine({rot, cv::Vec3f(0, 0, 0)});

    std::vector<cv::Point3f> vertice = GeometryCV::transform(cam_sp_transform, points);
    std::vector<cv::Vec3f> normals_cs = GeometryCV::transform(cam_sp_transform_rot, normals);

    std::vector<cv::Point2f> vertice_2d;
    cv::projectPoints(vertice, cv::Vec3f(0, 0, 0), cv::Vec3f(0, 0, 0), cam, ks, vertice_2d);

    for (const auto &tri : indices) {
      std::vector<float> depth {
        vertice[tri[0]].z,
        vertice[tri[1]].z,
        vertice[tri[2]].z};

      std::vector<cv::Vec3f> norms {
        normals_cs[tri[0]],
        normals_cs[tri[1]],
        normals_cs[tri[2]]};

      std::vector<cv::Point3f> poly{
          cv::Vec3f(vertice_2d[tri[0]].x, vertice_2d[tri[0]].y, depth[0]),
          cv::Vec3f(vertice_2d[tri[1]].x, vertice_2d[tri[1]].y, depth[1]),
          cv::Vec3f(vertice_2d[tri[2]].x, vertice_2d[tri[2]].y, depth[2])};

      drawTriangleInterp(dst_depth_32fc1, dst_32fc3, poly, norms);
    }
  }
}