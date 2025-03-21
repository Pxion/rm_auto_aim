// Copyright (C) 2022 ChenJun
// Copyright (C) 2024 Zheng Yu
// Licensed under the MIT License.

#include "armor_detector/pnp_solver.hpp"

#include <opencv2/calib3d.hpp>
#include <vector>

// 添加包含声明枚举的头文件
#include "armor_detector/armor.hpp"

namespace rm_auto_aim
{
PnPSolver::PnPSolver(
  const std::array<double, 9> & camera_matrix, const std::vector<double> & dist_coeffs)
: camera_matrix_(cv::Mat(3, 3, CV_64F, const_cast<double *>(camera_matrix.data())).clone()),
  dist_coeffs_(cv::Mat(1, 5, CV_64F, const_cast<double *>(dist_coeffs.data())).clone())
{
  // Unit: m
  constexpr double small_half_y = SMALL_ARMOR_WIDTH / 2.0 / 1000.0;
  constexpr double small_half_z = SMALL_ARMOR_HEIGHT / 2.0 / 1000.0;
  constexpr double large_half_y = LARGE_ARMOR_WIDTH / 2.0 / 1000.0;
  constexpr double large_half_z = LARGE_ARMOR_HEIGHT / 2.0 / 1000.0;

  // Start from bottom left in clockwise order
  // Model coordinate: x forward, y left, z up
  small_armor_points_.emplace_back(cv::Point3f(0, small_half_y, -small_half_z));
  small_armor_points_.emplace_back(cv::Point3f(0, small_half_y, small_half_z));
  small_armor_points_.emplace_back(cv::Point3f(0, -small_half_y, small_half_z));
  small_armor_points_.emplace_back(cv::Point3f(0, -small_half_y, -small_half_z));

  large_armor_points_.emplace_back(cv::Point3f(0, large_half_y, -large_half_z));
  large_armor_points_.emplace_back(cv::Point3f(0, large_half_y, large_half_z));
  large_armor_points_.emplace_back(cv::Point3f(0, -large_half_y, large_half_z));
  large_armor_points_.emplace_back(cv::Point3f(0, -large_half_y, -large_half_z));
}

bool PnPSolver::solvePnP(const Armor & armor, cv::Mat & rvec, cv::Mat & tvec)
{
  std::vector<cv::Point2f> image_armor_points;

  // Fill in image points
  image_armor_points.emplace_back(armor.left_light.bottom);
  image_armor_points.emplace_back(armor.left_light.top);
  image_armor_points.emplace_back(armor.right_light.top);
  image_armor_points.emplace_back(armor.right_light.bottom);

  // Solve pnp
  auto object_points = armor.type == ArmorType::SMALL ? small_armor_points_ : large_armor_points_;
  return cv::solvePnP(
    object_points, image_armor_points, camera_matrix_, dist_coeffs_, rvec, tvec, false,
    cv::SOLVEPNP_IPPE);
}

float PnPSolver::calculateDistanceToCenter(const cv::Point2f & image_point)
{
  float cx = camera_matrix_.at<double>(0, 2);
  float cy = camera_matrix_.at<double>(1, 2);
  return cv::norm(image_point - cv::Point2f(cx, cy));
}

double rm_auto_aim::PnPSolver:: getYaw(YawPnP* yaw_pnp, double yaw) {
    // 求解yaw
    // 通过角度代价函数计算偏航角（搜索范围-π/2~π/2，步长0.03弧度）
    double angle_yaw = yaw_pnp->getYawByAngleCost(-(M_PI / 2), (M_PI / 2), 0.03);
    // 通过像素代价函数计算偏航角（相同搜索范围和步长）
    double pixel_yaw = yaw_pnp->getYawByPixelCost(-(M_PI / 2), (M_PI / 2), 0.03);
    // 混合两种方法得到最终补偿值
    double append_yaw = yaw_pnp->getYawByMix(pixel_yaw, angle_yaw);
    // 释放yaw_pnp对象内存
    delete yaw_pnp;
    // 返回补偿后的总偏航角（附加补偿+原始偏航）
    return append_yaw + yaw;
}

double YawPnP::getYawByAngleCost(double left, double right, double epsilon) const {
    while (right - left > epsilon) {
        double mid1 = left + (right - left) / 3;
        double mid2 = right - (right - left) / 3;

        double f1 = getAngleCost(mid1);
        double f2 = getAngleCost(mid2);

        if (f1 < f2) {
            right = mid2;
        } else {
            left = mid1;
        }
    }
    return (left + right) / 2;
}

double YawPnP::getYawByPixelCost(double left, double right, double epsilon) const {
    while (right - left > epsilon) {
        double mid1 = left + (right - left) / 3;
        double mid2 = right - (right - left) / 3;

        double f1 = getPixelCost(mid1);
        double f2 = getPixelCost(mid2);

        if (f1 < f2) {
            right = mid2;
        } else {
            left = mid1;
        }
    }
    return (left + right) / 2;
}

double YawPnP::getYawByMix(double pixel_yaw, double angle_yaw) const {
    double mid = 0.3;
    double len = 0.1;
    
    double ratio = 0.5 + 0.5 * sin(M_PI * (fabs(pixel_yaw) - mid) / len);
    double append_yaw;

    if ((fabs(pixel_yaw) > (mid - len / 2)) && (fabs(pixel_yaw) < (mid + len / 2))) {
        append_yaw = ratio * pixel_yaw + (1 - ratio) * angle_yaw;
    } else if (fabs(pixel_yaw) <= (mid - len / 2)) {
        append_yaw = angle_yaw;
    } else {
        append_yaw = pixel_yaw;
    }
    return append_yaw;
}

void YawPnP::setWorldPoints(const std::vector<cv::Point3f>& object_points,const std::string& number) { 
    number_ = number;//设置装甲板数字类型
    // 添加空值检查
    if (object_points.empty()) {
        throw std::invalid_argument("Object points cannot be empty");
    }
    P_world.clear();
    for (const auto& p : object_points) {
        P_world.push_back(Eigen::Vector4d(0, -(p.x * 1e-3), -(p.y * 1e-3), 1));
    }
}

void YawPnP::setImagePoints(const std::vector<cv::Point2f>& image_points) { 
    P_pixel.clear();
    for (const auto& p : image_points) {
        P_pixel.push_back(Eigen::Vector2d(p.x, p.y));
    }
}

double YawPnP::getPixelCost(const std::vector<Eigen::Vector2d>& P_project, double append_yaw) const {
    if (P_pixel.size() != P_project.size() || P_project.size() < 4) return 0.0;

    int map[4] = {0, 1, 3, 2};

    double cost = 0.0;
    for (int i = 0; i < 4; i++) {
        int index_this = map[i];
        int index_next = map[(i + 1) % 4];
        Eigen::Vector2d pixel_line = P_pixel[index_next] - P_pixel[index_this];
        Eigen::Vector2d project_line = P_project[index_next] - P_project[index_this];

        double this_dist = (P_pixel[index_this] - P_project[index_this]).norm();
        double next_dist = (P_pixel[index_next] - P_project[index_next]).norm();
        double line_dist = fabs(pixel_line.norm() - project_line.norm());

        double pixel_dist = (0.5 * (this_dist + next_dist) + line_dist) / pixel_line.norm();
        cost += pixel_dist;
    }
    return cost;
}

double YawPnP::getAngleCost(const std::vector<Eigen::Vector2d>& P_project, double append_yaw) const {
    if (P_pixel.size() != P_project.size() || P_project.size() < 4) return 0.0;

    int map[4] = {0, 1, 3, 2};

    double cost = 0.0;
    for (int i = 0; i < 4; i++) {
        int index_this = map[i];
        int index_next = map[(i + 1) % 4];
        Eigen::Vector2d pixel_line = P_pixel[index_next] - P_pixel[index_this];
        Eigen::Vector2d project_line = P_project[index_next] - P_project[index_this];

        double cos_angle = pixel_line.dot(project_line) / (pixel_line.norm() * project_line.norm());
        double angle_dist = fabs(acos(cos_angle));

        cost += angle_dist;
    }
    return cost;
}

// 重载函数
double YawPnP::getPixelCost(double append_yaw) const {
    // 通过映射关系获取投影点
    auto P_world_rot = getMapping(append_yaw);
    auto P_project = getProject(P_world_rot);
    // 调用已有实现
    return getPixelCost(P_project, append_yaw);
}

double YawPnP::getAngleCost(double append_yaw) const {
    auto P_world_rot = getMapping(append_yaw);
    auto P_project = getProject(P_world_rot);
    return getAngleCost(P_project, append_yaw);
}

std::vector<Eigen::Vector4d> YawPnP::getMapping(double append_yaw) const {
    Eigen::Matrix4d M;
    std::vector<Eigen::Vector4d> P_mapping;

    double yaw = sys_yaw + append_yaw;
    double pitch;
    
    //根据装甲板数字类型设置俯仰角
    if (number_ == "outpost"){

        pitch = rm_auto_aim::ANGLE_DOWN_15;

    } else {

        pitch = rm_auto_aim::ANGLE_UP_15;
    }

    pitch = -pitch;
    M << cos(yaw) * cos(pitch), -sin(yaw), -sin(pitch) * cos(yaw), pose(0),
         sin(yaw) * cos(pitch),  cos(yaw), -sin(pitch) * sin(yaw), pose(1),
                    sin(pitch),         0,             cos(pitch), pose(2),
                             0,         0,                      0,       1;
    
    for (const auto& p : P_world) {
        P_mapping.push_back(M * p);
    }
    return P_mapping;
}

std::vector<Eigen::Vector2d> YawPnP::getProject(const std::vector<Eigen::Vector4d>& P_world) const {
    std::vector<Eigen::Vector2d> P_project;
    for (const auto& p : P_world) {
        Eigen::Vector3d p_camera = (T_inv * p).head(3);
        Eigen::Vector3d p_project = Kc * p_camera;
        P_project.push_back(p_project.head(2) / p_camera(2));
    }
    return P_project;
}



}  // namespace rm_auto_aim

