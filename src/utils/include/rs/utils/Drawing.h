#pragma once

#include <rs/utils/GeometryCV.h>


/// \namespace Drawing Drawing.h
/// \brief Helper drawing functions contains in this namespace
namespace Drawing {
  /// \brief Maximum z_buffer value for 16bit integer depth
  constexpr uint16_t max_depth_16u = std::numeric_limits<uint16_t>::max();

  /// \brief Maximum z_buffer value for float single-precision depth
  constexpr float max_depth_32f = std::numeric_limits<float>::max();

  /// \brief Draw an interpolated triangle with depth test
  /// \param[in,out] z_buffer   Floating point depth buffer
  /// \param[in,out] dst        Optional image to write interpolated values
  /// \param[in]     poly       3 vertices to draw
  /// \param[in]     vals       Optional 3 values to interpolate and write to `dst`
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

  /// \brief Rasterise a mesh and draw only depth buffer
  /// \param[in,out]  dst_32fc1  Floating point z-buffer, initialised to max_depth_*
  /// \param[in]      points     3d vertices
  /// \param[in]      indices    Triangle indices
  /// \param[in]      rot        Vertice rotation vector (rodrigues)
  /// \param[in]      trans      Vertice translation vector
  /// \param[in]      cam        Pinhole camera matrix
  /// \param[in]      ks         Camera distortion coefficients
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

  /// \brief Rasterise a mesh and draw depth buffer and normals
  /// \param[in,out]  dst_32fc1  Floating point z-buffer, initialised to max_depth_*
  /// \param[in]      points     3d vertices
  /// \param[in]      normals    Vertex normals
  /// \param[in]      indices    Triangle indices
  /// \param[in]      rot        Vertice rotation vector (rodrigues)
  /// \param[in]      trans      Vertice translation vector
  /// \param[in]      cam        Pinhole camera matrix
  /// \param[in]      ks         Camera distortion coefficients
  /// \param[in]      flat       Use flat shading (single normal per triangle)
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

  /// \brief Draw histogram image
  /// \param[in] data       Vector of double values
  /// \param[in] bin_width  Width in px of single value column
  /// \param[in] height     Output image height
  /// \param[in] level      Color bins higher and lower than this value in different colors
  /// \return               Image with drawn histogram on it
  cv::Mat drawHistogram(const std::vector<double> &data, const int bin_width, const int height, const double level = 0.) {
    int bins_num = data.size();
    cv::Mat hist = cv::Mat::zeros(height, bins_num*bin_width, CV_8UC3);

    double max_val = *std::max_element(data.begin(), data.end());

    int i = 0;
    for (auto &bin : data) {
      float x0 = i * bin_width;
      float y0 = (1 - bin/max_val) * height;
      auto color = (bin > level ? cv::Scalar(0, 128, 0) : cv::Scalar(0, 0, 128));
      cv::rectangle(hist, cv::Rect(x0, y0, bin_width, height - y0 + 1), color, -1);
      i++;
    }

    return hist;
  }
}