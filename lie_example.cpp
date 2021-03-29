#include <iostream>
#include <manif/SE3.h>
#include <tf2_ros/transform_broadcaster.h>
#include <random>
#include <nav_msgs/msg/path.hpp>

class LieNode : public rclcpp::Node {
public:
    LieNode() : Node("lie_node") { };
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LieNode>();

  rclcpp::Rate r(20);

  tf2_ros::TransformBroadcaster br(node);

  rclcpp::Time start = node->now();

  geometry_msgs::msg::TransformStamped tf, tf_drifty;
  tf.header.frame_id = "map";
  tf.child_frame_id = "base_link";
  tf_drifty = tf;
  tf_drifty.child_frame_id = "base_link_drifty";

  // manif::SE3d a{manif::SE3d::Translation{1, 1, 0}, manif::SO3d::Identity()};
  manif::SE3d b{manif::SE3d::Translation{10, 0, 0}, manif::SO3d(Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ()))};
  manif::SE3d pose = b; //manif::SE3d::Identity();
  manif::SE3d pose_prev = b;
  manif::SE3d pose_drifty = b; //manif::SE3d::Identity();

  // Angeles Jorge

  manif::SE3d::Tangent c_log;
  //                   v,      w
  c_log.coeffs() << 0, 0, 1, 0, 0, 1; // se(3)
  double t_prev = 0;

  std::default_random_engine generator;
  std::normal_distribution<double> dist(0, 1);

  auto path_pub = node->create_publisher<nav_msgs::msg::Path>("path", 10);
  nav_msgs::msg::Path path;
  path.header.frame_id = "map";
  path.header.stamp = rclcpp::Time();

  double bias[6];
  for (int i = 0; i < 6; i++) {
      bias[i] = dist(generator);
  }

  const double delta_t = M_PI_4 * (8. / (503./4.));
  double t = 0;
  while(rclcpp::ok()) {
    rclcpp::Time now = node->now();
    tf.header.stamp = now;
    tf_drifty.header.stamp = now;

    std::cout << t << std::endl;

    if (t > M_PI * 8.) break;

    c_log.coeffs()(2) = std::sin(t / 4);

    pose_prev = pose;
    pose = (t - t_prev) * c_log + pose;

    manif::SE3d::Tangent c_log_drift = pose - pose_prev;
    double v_norm = c_log_drift.coeffs().head<3>().norm();
    for (int i = 0; i < 3; i++) {
        auto& coeff = c_log_drift.coeffs()(i);
        coeff += v_norm * 0.1 * bias[i];
    }
    double w_norm = c_log_drift.coeffs().tail<3>().norm();
    for (int i = 3; i < 6; i++) {
        auto& coeff = c_log_drift.coeffs()(i);
        coeff += w_norm * 0.03 * bias[i];
    }

    pose_drifty = pose_drifty + c_log_drift;

    auto* pose_current = &pose;
    auto* tf_current = &tf;

    for (int i = 0; i < 2; i++) {
        tf_current->transform.translation.x = pose_current->translation().x();
        tf_current->transform.translation.y = pose_current->translation().y();
        tf_current->transform.translation.z = pose_current->translation().z();

        tf_current->transform.rotation.x = pose_current->quat().x();
        tf_current->transform.rotation.y = pose_current->quat().y();
        tf_current->transform.rotation.z = pose_current->quat().z();
        tf_current->transform.rotation.w = pose_current->quat().w();

        if (i == 1) {
            path.poses.emplace_back();
            path.poses.back().header.frame_id = "base_link";
            path.poses.back().header.stamp = rclcpp::Time();

            path.poses.back().pose.position.x = pose_current->translation().x();
            path.poses.back().pose.position.y = pose_current->translation().y();
            path.poses.back().pose.position.z = pose_current->translation().z();

            path.poses.back().pose.orientation.x = pose_current->quat().x();
            path.poses.back().pose.orientation.y = pose_current->quat().y();
            path.poses.back().pose.orientation.z = pose_current->quat().z();
            path.poses.back().pose.orientation.w = pose_current->quat().w();

            path_pub->publish(path);
        }

        br.sendTransform(*tf_current);
        pose_current = &pose_drifty;
        tf_current = &tf_drifty;
    }


    t_prev = t;
    t += delta_t;
    r.sleep();
  }

  rclcpp::spin(node);
}
