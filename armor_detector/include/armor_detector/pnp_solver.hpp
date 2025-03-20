// Copyright (C) 2022 ChenJun
// Copyright (C) 2024 Zheng Yu
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__PNP_SOLVER_HPP_
#define ARMOR_DETECTOR__PNP_SOLVER_HPP_

#include <geometry_msgs/msg/point.hpp>
#include <opencv2/core.hpp>

// STD
#include <array>
#include <vector>

#include "armor_detector/armor.hpp"

// 添加 Eigen 头文件包含
#include <Eigen/Dense>

// 添加 ArmorElevation 类型前置声明
enum class ArmorElevation;

// 前置声明
class YawPnP;

namespace rm_auto_aim
{

class YawPnP {
public:
    YawPnP() {}
    YawPnP(ArmorElevation elevation) : elevation(elevation) {}
    
    // 修复构造函数参数类型
    YawPnP(const std::vector<cv::Point2f>& image_points,
          const std::vector<cv::Point3f>& world_points,
          const cv::Mat& camera_matrix,
          const cv::Mat& dist_coeffs);

    void setWorldPoints(const std::vector<cv::Point3f>& object_points);
    void setImagePoints(const std::vector<cv::Point2f>& image_points);

    double operator()(double append_yaw) const;

    ArmorElevation setElevation(double pitch);
    //ArmorElevation setElevation(rm_auto_aim::ArmorID armor_id);
    std::vector<Eigen::Vector4d> getMapping(double append_yaw) const;
    std::vector<Eigen::Vector2d> getProject(const std::vector<Eigen::Vector4d>& P_world) const;
    double getCost(const std::vector<Eigen::Vector2d>& P_project, double append_yaw) const;
    double getPixelCost(const std::vector<Eigen::Vector2d>& P_project, double append_yaw) const;
    double getAngleCost(const std::vector<Eigen::Vector2d>& P_project, double append_yaw) const;

    double getCost(double append_yaw) const;
    double getPixelCost(double append_yaw) const;
    double getAngleCost(double append_yaw) const;

    double getYawByPixelCost(double left, double right, double epsilon) const;
    double getYawByAngleCost(double left, double right, double epsilon) const;
    double getYawByMix(double pixel_yaw, double angle_yaw) const;


    double          sys_yaw;
    Eigen::Vector4d pose;
    ArmorElevation  elevation;

    std::vector<Eigen::Vector2d> P_pixel;      // 四点真实像素坐标
    std::vector<Eigen::Vector4d> P_world;      // 四点正对世界坐标

    Eigen::Matrix3d Kc;                        // 相机内参矩阵
    Eigen::Matrix4d T;                         // 图像坐标系在陀螺仪坐标系下的表示
    Eigen::Matrix4d T_inv;                     // 陀螺仪坐标系在图像坐标系下的表示
};

class PnPSolver
{
public:

  double getYaw(YawPnP* yaw_pnp, double yaw);
  // 将 armor_points 设为 public 
  std::vector<cv::Point3f> small_armor_points_;
  std::vector<cv::Point3f> large_armor_points_;

  PnPSolver(
    const std::array<double, 9> & camera_matrix,
    const std::vector<double> & distortion_coefficients);

  // Get 3d position
  bool solvePnP(const Armor & armor, cv::Mat & rvec, cv::Mat & tvec);

  // Calculate the distance between armor center and image center
  float calculateDistanceToCenter(const cv::Point2f & image_point);

private:
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;

  // Unit: mm
  static constexpr float SMALL_ARMOR_WIDTH = 132;
  static constexpr float SMALL_ARMOR_HEIGHT = 57;
  static constexpr float LARGE_ARMOR_WIDTH = 223;
  static constexpr float LARGE_ARMOR_HEIGHT = 57;

  // // Four vertices of armor in 3d
  // std::vector<cv::Point3f> small_armor_points_;
  // std::vector<cv::Point3f> large_armor_points_;
};




}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__PNP_SOLVER_HPP_
