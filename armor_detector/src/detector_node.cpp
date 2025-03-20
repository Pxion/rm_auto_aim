// Copyright (C) 2022 ChenJun
// Copyright (C) 2024 Zheng Yu
// Licensed under the MIT License.

#include <cv_bridge/cv_bridge.h>
#include <rmw/qos_profiles.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/convert.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <image_transport/image_transport.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/duration.hpp>
#include <rclcpp/qos.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// STD
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "armor_detector/armor.hpp"
#include "armor_detector/detector_node.hpp"


#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/create_timer_ros.h>
// 添加 tf2 头文件
#include <tf2/utils.h>

namespace rm_auto_aim
{
ArmorDetectorNode::ArmorDetectorNode(const rclcpp::NodeOptions & options)
: Node("armor_detector", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting DetectorNode!");

  // 初始化时获取相机参数
  if (cam_info_) {
    camera_matrix_ = cv::Mat(3, 3, CV_64F, const_cast<double*>(cam_info_->k.data())).clone();
    dist_coeffs_ = cv::Mat(1, 5, CV_64F, const_cast<double*>(cam_info_->d.data())).clone();
  }

  // 初始化装甲板检测器
  detector_ = initDetector();

  // 创建装甲板检测结果发布者（使用传感器数据QoS配置）
  armors_pub_ = this->create_publisher<auto_aim_interfaces::msg::Armors>(
    "/detector/armors", rclcpp::SensorDataQoS());

  // Visualization Marker Publisher
  // See http://wiki.ros.org/rviz/DisplayTypes/Marker
  armor_marker_.ns = "armors";
  armor_marker_.action = visualization_msgs::msg::Marker::ADD;
  armor_marker_.type = visualization_msgs::msg::Marker::CUBE;
  armor_marker_.scale.x = 0.05;  // 立方体X轴尺寸
  armor_marker_.scale.z = 0.125; // 立方体Z轴尺寸
  armor_marker_.color.a = 1.0;   // 不透明度
  armor_marker_.color.g = 0.5;   // 基础颜色（绿色分量）
  armor_marker_.color.b = 1.0;   // 基础颜色（蓝色分量）
  armor_marker_.lifetime = rclcpp::Duration::from_seconds(0.1);

  // 文本标记物显示分类结果
  text_marker_.ns = "classification";
  text_marker_.action = visualization_msgs::msg::Marker::ADD;
  text_marker_.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  text_marker_.scale.z = 0.1;    // 文本大小
  text_marker_.color.a = 1.0;    // 文本不透明度
  text_marker_.color.r = 1.0;    // 文本颜色（白色）
  text_marker_.color.g = 1.0;
  text_marker_.color.b = 1.0;
  text_marker_.lifetime = rclcpp::Duration::from_seconds(0.1);

  // 创建标记物数组发布者
  marker_pub_ =
    this->create_publisher<visualization_msgs::msg::MarkerArray>("/detector/marker", 10);

  // 初始化TF2相关组件（用于坐标变换）
  tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  // 创建定时器接口（用于TF2超时管理）
  auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
    this->get_node_base_interface(), this->get_node_timers_interface());
  tf2_buffer_->setCreateTimerInterface(timer_interface);
  // 创建TF2监听器（自动接收坐标变换数据）
  tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);

  // 初始化调试模式参数
  debug_ = this->declare_parameter("debug", false);
  if (debug_) {
    createDebugPublishers();  // 创建调试信息发布者
  }

  // 初始化任务模式订阅（默认开启瞄准任务）
  is_aim_task_ = true;
  task_sub_ = this->create_subscription<std_msgs::msg::String>(
    "/task_mode", 10, std::bind(&ArmorDetectorNode::taskCallback, this, std::placeholders::_1));

  // 设置调试参数动态监控（支持运行时调试模式切换）
  debug_param_sub_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
  debug_cb_handle_ =
    debug_param_sub_->add_parameter_callback("debug", [this](const rclcpp::Parameter & p) {
      debug_ = p.as_bool();
      debug_ ? createDebugPublishers() : destroyDebugPublishers();
    });

  // 订阅相机信息（单次订阅，获取后自动取消订阅）
  cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
    "/camera_info", rclcpp::SensorDataQoS(),
    [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info) {
      cam_center_ = cv::Point2f(camera_info->k[2], camera_info->k[5]);  // 获取相机光心坐标
      cam_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*camera_info);
      pnp_solver_ = std::make_unique<PnPSolver>(camera_info->k, camera_info->d);  // 初始化PnP解算器
      cam_info_sub_.reset();  // 收到相机信息后取消订阅
    });

  // 订阅图像数据（主处理回调）
  img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    "/image_raw", rclcpp::SensorDataQoS(),
    std::bind(&ArmorDetectorNode::imageCallback, this, std::placeholders::_1));


}

// 任务模式回调函数（接收任务模式指令）
// 参数：task_msg - 包含任务模式字符串的消息
void ArmorDetectorNode::taskCallback(const std_msgs::msg::String::SharedPtr task_msg)
{
// 从消息中提取任务模式字符串
std::string task_mode = task_msg->data;

// 根据任务模式设置标志位
if (task_mode == "aim") {          // 收到瞄准任务指令
is_aim_task_ = true;             // 启用装甲板检测处理
} else {                           // 收到其他任务指令
is_aim_task_ = false;            // 停用装甲板检测处理
}
}

void ArmorDetectorNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr img_msg)
{
  // 检测图像中的装甲板（相机坐标系）
  auto camera_frame_armors = detectArmors(img_msg);

  // 当PnP解算器就绪且处于瞄准任务模式时处理检测结果
  if (pnp_solver_ != nullptr && is_aim_task_) {
    // 初始化消息头并清空之前的数据
    armors_msg_.header = armor_marker_.header = text_marker_.header = img_msg->header;
    armors_msg_.armors.clear();
    marker_array_.markers.clear();
    armor_marker_.id = 0;  // 重置标记ID
    text_marker_.id = 0;

    // 遍历所有检测到的装甲板
    auto_aim_interfaces::msg::Armor armor_msg;
    for (const auto & armor : camera_frame_armors) {
      cv::Mat rvec, tvec;
      bool success = pnp_solver_->solvePnP(armor, rvec, tvec);
      if (success) {
        // 填充装甲板基本信息
        armor_msg.type = ARMOR_TYPE_STR[static_cast<int>(armor.type)];
        armor_msg.number = armor.number;

        // 将旋转向量转换为旋转矩阵
        cv::Mat rotation_matrix;
        cv::Rodrigues(rvec, rotation_matrix);
        
        // 将旋转矩阵转换为四元数
        tf2::Matrix3x3 tf2_rotation_matrix(
          rotation_matrix.at<double>(0, 0), rotation_matrix.at<double>(0, 1),
          rotation_matrix.at<double>(0, 2), rotation_matrix.at<double>(1, 0),
          rotation_matrix.at<double>(1, 1), rotation_matrix.at<double>(1, 2),
          rotation_matrix.at<double>(2, 0), rotation_matrix.at<double>(2, 1),
          rotation_matrix.at<double>(2, 2));
        tf2::Quaternion tf2_q;
        tf2_rotation_matrix.getRotation(tf2_q);

        // 坐标系转换 -------------------------------------------------
        geometry_msgs::msg::PoseStamped camera_pose;
        camera_pose.header = img_msg->header;
        camera_pose.pose.position.x = tvec.at<double>(0);
        camera_pose.pose.position.y = tvec.at<double>(1);
        camera_pose.pose.position.z = tvec.at<double>(2);
        camera_pose.pose.orientation = tf2::toMsg(tf2_q);

        try {
          // 将位姿从相机坐标系转换到odom坐标系
          auto odom_pose = tf2_buffer_->transform(camera_pose, "odom");

          // 提取原始姿态的欧拉角
          double original_roll, original_pitch, original_yaw;
          tf2::Matrix3x3(tf2_q).getRPY(original_roll, original_pitch, original_yaw);

          // 创建YawPnP对象
          YawPnP* yaw_pnp = new YawPnP();

          // 提前定义 object_points 和 image_points
          auto object_points = armor.type == ArmorType::SMALL ? 
                              pnp_solver_->small_armor_points_ : 
                              pnp_solver_->large_armor_points_;
          std::vector<cv::Point2f> image_points = {
              armor.left_light.bottom,
              armor.left_light.top,
              armor.right_light.top,
              armor.right_light.bottom
          };

          // 设置yaw
          yaw_pnp->sys_yaw = original_yaw;

          // 设置装甲板四点坐标
          yaw_pnp->setWorldPoints(object_points);
          yaw_pnp->setImagePoints(image_points);

          // 通过类成员函数调用 getYaw
          double new_yaw = pnp_solver_->getYaw(yaw_pnp, original_yaw);

           // 生成新的四元数（保持原有roll/pitch）
          tf2::Quaternion q;
          q.setRPY(original_roll, original_pitch, new_yaw);  // 仅修改yaw
          q.normalize();
          
          // 更新位姿数据
          armor_msg.pose.position = odom_pose.pose.position;
          armor_msg.pose.orientation = tf2::toMsg(q);
          
        } catch (const tf2::TransformException & ex) {
          RCLCPP_ERROR(get_logger(), "坐标转换失败: %s", ex.what());
          continue;  // 跳过当前装甲板的处理
        }
        // -----------------------------------------------------------
  
      
        // 计算到图像中心的距离
        armor_msg.distance_to_image_center = pnp_solver_->calculateDistanceToCenter(armor.center);

        // 填充关键点坐标（保持原像素坐标系）
        armor_msg.kpts.clear();
        for (const auto & pt :
             {armor.left_light.top, armor.left_light.bottom, armor.right_light.bottom,
              armor.right_light.top}) {
          geometry_msgs::msg::Point point;
          point.x = pt.x;
          point.y = pt.y;
          armor_msg.kpts.emplace_back(point);
        }

        // 准备可视化标记（使用转换后的位姿）
        armor_marker_.id++;
        armor_marker_.scale.y = armor.type == ArmorType::SMALL ? 0.135 : 0.23;
        armor_marker_.pose = armor_msg.pose;
        
        text_marker_.id++;
        text_marker_.pose.position = armor_msg.pose.position;
        text_marker_.pose.position.y -= 0.1;
        text_marker_.text = armor.classfication_result;

        // 收集数据
        armors_msg_.armors.emplace_back(armor_msg);
        marker_array_.markers.emplace_back(armor_marker_);
        marker_array_.markers.emplace_back(text_marker_);
      } else {
        RCLCPP_WARN(this->get_logger(), "PnP failed!");
      }
    }

    // 发布odom坐标系下的装甲板信息
    armors_pub_->publish(armors_msg_);
    publishMarkers();
  }
}

std::unique_ptr<Detector> ArmorDetectorNode::initDetector()
{
  rcl_interfaces::msg::ParameterDescriptor param_desc;
  param_desc.integer_range.resize(1);
  param_desc.integer_range[0].step = 1;
  param_desc.integer_range[0].from_value = 0;
  param_desc.integer_range[0].to_value = 255;
  int binary_thres = declare_parameter("binary_thres", 160, param_desc);

  param_desc.description = "0-RED, 1-BLUE";
  param_desc.integer_range[0].from_value = 0;
  param_desc.integer_range[0].to_value = 1;
  auto detect_color = declare_parameter("detect_color", RED, param_desc);

  Detector::LightParams l_params = {
    .min_ratio = declare_parameter("light.min_ratio", 0.1),
    .max_ratio = declare_parameter("light.max_ratio", 0.4),
    .max_angle = declare_parameter("light.max_angle", 35.0),
    .min_fill_ratio = declare_parameter("light.min_fill_ratio", 0.8),
  };

  Detector::ArmorParams a_params = {
    .min_light_ratio = declare_parameter("armor.min_light_ratio", 0.7),
    .min_small_center_distance = declare_parameter("armor.min_small_center_distance", 0.8),
    .max_small_center_distance = declare_parameter("armor.max_small_center_distance", 3.2),
    .min_large_center_distance = declare_parameter("armor.min_large_center_distance", 3.2),
    .max_large_center_distance = declare_parameter("armor.max_large_center_distance", 5.5),
    .max_angle = declare_parameter("armor.max_angle", 35.0)};

  auto detector = std::make_unique<Detector>(binary_thres, detect_color, l_params, a_params);

  // Init classifier
  auto pkg_path = ament_index_cpp::get_package_share_directory("armor_detector");
  auto model_path = pkg_path + "/model/mlp.onnx";
  auto label_path = pkg_path + "/model/label.txt";
  double threshold = this->declare_parameter("classifier_threshold", 0.7);
  std::vector<std::string> ignore_classes =
    this->declare_parameter("ignore_classes", std::vector<std::string>{"negative"});
  detector->classifier =
    std::make_unique<NumberClassifier>(model_path, label_path, threshold, ignore_classes);

  return detector;
}

std::vector<Armor> ArmorDetectorNode::detectArmors(
  const sensor_msgs::msg::Image::ConstSharedPtr & img_msg)
{
  // Convert ROS img to cv::Mat
  auto img = cv_bridge::toCvShare(img_msg, "rgb8")->image;

  // Update params
  detector_->binary_thres = get_parameter("binary_thres").as_int();
  detector_->detect_color = get_parameter("detect_color").as_int();
  detector_->classifier->threshold = get_parameter("classifier_threshold").as_double();

  auto armors = detector_->detect(img);

  auto final_time = this->now();
  auto latency = (final_time - img_msg->header.stamp).seconds() * 1000;
  RCLCPP_DEBUG_STREAM(this->get_logger(), "Latency: " << latency << "ms");

  // Publish debug info
  if (debug_) {
    binary_img_pub_.publish(
      cv_bridge::CvImage(img_msg->header, "mono8", detector_->binary_img).toImageMsg());

    // Sort lights and armors data by x coordinate
    std::sort(
      detector_->debug_lights.data.begin(), detector_->debug_lights.data.end(),
      [](const auto & l1, const auto & l2) { return l1.center_x < l2.center_x; });
    std::sort(
      detector_->debug_armors.data.begin(), detector_->debug_armors.data.end(),
      [](const auto & a1, const auto & a2) { return a1.center_x < a2.center_x; });

    lights_data_pub_->publish(detector_->debug_lights);
    armors_data_pub_->publish(detector_->debug_armors);

    if (!armors.empty()) {
      auto all_num_img = detector_->getAllNumbersImage();
      number_img_pub_.publish(
        *cv_bridge::CvImage(img_msg->header, "mono8", all_num_img).toImageMsg());
    }

    detector_->drawResults(img);
    // Draw camera center
    cv::circle(img, cam_center_, 5, cv::Scalar(255, 0, 0), 2);
    // Draw latency
    std::stringstream latency_ss;
    latency_ss << "Latency: " << std::fixed << std::setprecision(2) << latency << "ms";
    auto latency_s = latency_ss.str();
    cv::putText(
      img, latency_s, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    result_img_pub_.publish(cv_bridge::CvImage(img_msg->header, "rgb8", img).toImageMsg());
  }

  return armors;
}

void ArmorDetectorNode::createDebugPublishers()
{
  lights_data_pub_ =
    this->create_publisher<auto_aim_interfaces::msg::DebugLights>("/detector/debug_lights", 10);
  armors_data_pub_ =
    this->create_publisher<auto_aim_interfaces::msg::DebugArmors>("/detector/debug_armors", 10);

  binary_img_pub_ = image_transport::create_publisher(this, "/detector/binary_img");
  number_img_pub_ = image_transport::create_publisher(this, "/detector/number_img");
  result_img_pub_ = image_transport::create_publisher(this, "/detector/result_img");
}

void ArmorDetectorNode::destroyDebugPublishers()
{
  lights_data_pub_.reset();
  armors_data_pub_.reset();

  binary_img_pub_.shutdown();
  number_img_pub_.shutdown();
  result_img_pub_.shutdown();
}

void ArmorDetectorNode::publishMarkers()
{
  using Marker = visualization_msgs::msg::Marker;
  armor_marker_.action = armors_msg_.armors.empty() ? Marker::DELETE : Marker::ADD;
  marker_array_.markers.emplace_back(armor_marker_);
  marker_pub_->publish(marker_array_);
}

}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::ArmorDetectorNode)
