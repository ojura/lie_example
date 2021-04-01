#include <iostream>
#include <tf2_ros/transform_broadcaster.h>
#include <random>
#include <nav_msgs/msg/path.hpp>
#include <ceres/ceres.h>
#include <manif/SE3.h>
#include <manif/ceres/ceres.h>

class LieNode : public rclcpp::Node {
 public:
  LieNode() : Node("lie_node"){};
};

void SE3ToGeometryMsgsPose(
    const manif::SE3d& pose, geometry_msgs::msg::PoseStamped& pose_stamped) {
  pose_stamped.pose.position.x = pose.translation().x();
  pose_stamped.pose.position.y = pose.translation().y();
  pose_stamped.pose.position.z = pose.translation().z();
  pose_stamped.pose.orientation.x = pose.quat().x();
  pose_stamped.pose.orientation.y = pose.quat().y();
  pose_stamped.pose.orientation.z = pose.quat().z();
  pose_stamped.pose.orientation.w = pose.quat().w();
}

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
  manif::SE3d b{
      manif::SE3d::Translation{10, 0, 0},
      manif::SO3d(Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ()))};
  manif::SE3d pose = b;  // manif::SE3d::Identity();
  manif::SE3d pose_prev = b;
  manif::SE3d pose_drifty = b;  // manif::SE3d::Identity();

  std::cout << "SE3d " << pose << " coeffs " << pose.coeffs() << " size "
            << pose.coeffs().size() << std::endl;

  // Angeles Jorge

  manif::SE3d::Tangent c_log;
  //                   v,      w
  c_log.coeffs() << 0, 0, 1, 0, 0, 1;  // se(3)
  double t_prev = 0;

  std::default_random_engine generator;
  std::normal_distribution<double> dist(0, 1);

  auto path_pub = node->create_publisher<nav_msgs::msg::Path>("path", 10);
  auto path_gt_pub = node->create_publisher<nav_msgs::msg::Path>("path_gt", 10);
  nav_msgs::msg::Path path;
  path.header.frame_id = "map";
  path.header.stamp = rclcpp::Time();

  double bias[6];
  for (int i = 0; i < 6; i++) {
    bias[i] = dist(generator);
  }

  const double delta_t = M_PI_4 * (8. / (503. / 4.));
  double t = 0;
  std::vector<manif::SE3d> poses_ground_truth, poses_drifty;
  poses_ground_truth.reserve(600);
  poses_drifty.reserve(600);
  while (rclcpp::ok()) {
    rclcpp::Time now = node->now();
    tf.header.stamp = now;
    tf_drifty.header.stamp = now;

    std::cout << t << std::endl;

    if (t > M_PI * 8.) break;

    c_log.coeffs()(2) = std::sin(t / 4);

    pose_prev = pose;
    pose = (t - t_prev) * c_log + pose;

    poses_ground_truth.push_back(pose);

    manif::SE3d::Tangent c_log_drift = pose - pose_prev;
    double v_norm = c_log_drift.coeffs().head<3>().norm();
    for (int i = 0; i < 3; i++) {
      auto& coeff = c_log_drift.coeffs()(i);
      coeff += v_norm * (0.1 * bias[i] + 0.01 * dist(generator));
    }
    double w_norm = c_log_drift.coeffs().tail<3>().norm();
    for (int i = 3; i < 6; i++) {
      auto& coeff = c_log_drift.coeffs()(i);
      coeff += w_norm * (0.03 * bias[i] + 0.003 * dist(generator));
    }

    pose_drifty = pose_drifty + c_log_drift;
    poses_drifty.push_back(pose_drifty);

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

      br.sendTransform(*tf_current);
      pose_current = &pose_drifty;
      tf_current = &tf_drifty;
    }

    t_prev = t;
    t += delta_t;
    // r.sleep();
  }

  auto path_gt = path;
  path_gt.poses.resize(poses_ground_truth.size());
  for (int i = 0; i < poses_ground_truth.size(); i++) {
    SE3ToGeometryMsgsPose(poses_ground_truth[i], path_gt.poses.at(i));
  }

  auto poses_to_be_optimized = poses_drifty;

  ceres::Problem::Options problem_options;
  ceres::Problem problem(problem_options);

  auto* parameterization = new ceres::AutoDiffLocalParameterization<
      manif::CeresLocalParameterizationFunctor<manif::SE3d>, 7, 6>();
  for (auto& p : poses_to_be_optimized) {
    problem.AddParameterBlock(p.data(), 7, parameterization);
  }

  problem.SetParameterBlockConstant(poses_to_be_optimized[0].data());

  for (int i = 1; i < poses_drifty.size(); i++) {
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<
            manif::CeresConstraintFunctor<manif::SE3d>, 6, 7, 7>(
            new manif::CeresConstraintSE3(poses_drifty[i] -
                                          poses_drifty[i - 1])),
        nullptr, poses_to_be_optimized[i - 1].data(),
        poses_to_be_optimized[i].data());
  }

  problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<manif::CeresConstraintSE3, 6, 7, 7>(
          new manif::CeresConstraintSE3(manif::SE3d::Tangent::Zero())),
      nullptr,
      poses_to_be_optimized.back().data(),
      poses_to_be_optimized[0].data());

  // Run the solver!
  path.poses.resize(poses_to_be_optimized.size());

  struct IterationCallbackFunctor : public ceres::IterationCallback {
    std::function<void(void)> callback;

    ceres::CallbackReturnType operator()(const ceres::IterationSummary&) {
      callback();
      return ceres::SOLVER_CONTINUE;
    }
  };

  IterationCallbackFunctor iteration_callback;
  iteration_callback.callback = [&]() {
      for (int i = 0; i < poses_to_be_optimized.size(); i++) {
       SE3ToGeometryMsgsPose(poses_to_be_optimized[i], path.poses.at(i));
      }
    for (int i = 0; i < 50; i++) {
        path_pub->publish(path);
        path_gt_pub->publish(path_gt);
        r.sleep();
      }
    };
  iteration_callback.callback();

  ceres::Solver::Summary summary;
  ceres::Solver::Options solver_options;
  solver_options.update_state_every_iteration = true;
  solver_options.minimizer_progress_to_stdout = true;
  solver_options.callbacks.push_back(&iteration_callback);
  ceres::Solve(solver_options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n";

  while (rclcpp::ok()) {
    path.header.stamp = node->now();
    path_pub->publish(path);
    path_gt_pub->publish(path_gt);
    r.sleep();
  }
}