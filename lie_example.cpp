#include <iostream>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <random>
#include <nav_msgs/msg/path.hpp>
#include <ceres/ceres.h>
#include <manif/SE3.h>
#include <manif/ceres/ceres.h>
#include <visualization_msgs/msg/marker.hpp>

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

void SE3ToTransformMsg(
    const manif::SE3d& pose, geometry_msgs::msg::TransformStamped& transform_stamped) {
  transform_stamped.transform.translation.x = pose.translation().x();
  transform_stamped.transform.translation.y = pose.translation().y();
  transform_stamped.transform.translation.z = pose.translation().z();
  transform_stamped.transform.rotation.x = pose.quat().x();
  transform_stamped.transform.rotation.y = pose.quat().y();
  transform_stamped.transform.rotation.z = pose.quat().z();
  transform_stamped.transform.rotation.w = pose.quat().w();
}

void TranslationToPoint(
    const manif::SE3d::Translation& translation, geometry_msgs::msg::Point& point) {
  point.x = translation.x();
  point.y = translation.y();
  point.z = translation.z();
}

template <typename LieGroup>
class ConstraintFunctor {
  using Tangent = typename LieGroup::Tangent;

  template <typename Scalar>
  using LieGroupTemplate = typename LieGroup::template LieGroupTemplate<Scalar>;

  template <typename Scalar>
  using TangentTemplate = typename Tangent::template TangentTemplate<Scalar>;

 public:
  MANIF_MAKE_ALIGNED_OPERATOR_NEW_COND_TYPE(Tangent)

  template <typename... Args>
  ConstraintFunctor(double omega_scale, Args&&... args)
      : measurement_(std::forward<Args>(args)...), omega_scale_(omega_scale) {}

  template <typename T>
  bool operator()(const T* const past_raw, const T* const futur_raw,
                  T* residuals_raw) const {
    const Eigen::Map<const LieGroupTemplate<T>> state_past(past_raw);
    const Eigen::Map<const LieGroupTemplate<T>> state_future(futur_raw);

    Eigen::Map<TangentTemplate<T>> residuals(residuals_raw);

    /// r = m - ( future (-) past )
    residuals = measurement_.template cast<T>() - (state_future - state_past);
    residuals.coeffs().template tail<3>() *= T(omega_scale_);

    /// r = exp( log(m)^-1 . ( past^-1 . future ) )

    //    residuals =
    //      measurement_.exp().template cast<T>()
    //        .between(state_past.between(state_future)).log();

    return true;
  }

 private:
  const Tangent measurement_;
  const double omega_scale_;
};


template <typename LieGroup>
class ObservationFunctor
{
  using Tangent  = typename LieGroup::Tangent;

  template <typename Scalar>
  using LieGroupTemplate = typename LieGroup::template LieGroupTemplate<Scalar>;

  template <typename Scalar>
  using TangentTemplate = typename Tangent::template TangentTemplate<Scalar>;

 public:
  MANIF_MAKE_ALIGNED_OPERATOR_NEW_COND_TYPE(Tangent)

  template <typename... Args>
  ObservationFunctor(double scale, const Eigen::Vector3d& landmark, Args&&... args)
      : landmark_(landmark), observation_(std::forward<Args>(args)...), scale_(scale)
  { }

  template<typename T>
  bool operator()(const T* const pose_raw,
                  T* residual_raw) const {
    const Eigen::Map<const LieGroupTemplate<T>> pose(pose_raw);

    Eigen::Map<Eigen::Matrix<T, 3, 1>> residual(residual_raw);

    /// r = m - ( future (-) past )
    residual = T(scale_) * (pose.act(observation_.template cast<T>()) - landmark_.template cast<T>());

    /// r = exp( log(m)^-1 . ( past^-1 . future ) )

    //    residuals =
    //      measurement_.exp().template cast<T>()
    //        .between(state_past.between(state_future)).log();

    return true;
  }

 private:
  const Eigen::Vector3d& landmark_;
  const Eigen::Vector3d observation_;
  double scale_;
};

using SE3Constraint = ConstraintFunctor<manif::SE3d>;
using SE3Observation = ObservationFunctor<manif::SE3d>;

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<LieNode>();

  rclcpp::Rate r(200);

  tf2_ros::TransformBroadcaster br(node);
  tf2_ros::StaticTransformBroadcaster br_static(node);

  rclcpp::Time start = node->now();

  geometry_msgs::msg::TransformStamped tf, tf_drifty;
  std::vector<geometry_msgs::msg::TransformStamped> static_tfs;
  tf.header.frame_id = "map";
  tf.child_frame_id = "base_link";
  tf_drifty = tf;
  tf_drifty.child_frame_id = "base_link_drifty";

  std::vector<manif::SE3d::Translation> landmarks;
  landmarks.reserve(3);

  static_tfs.emplace_back();
  auto& axis_tf = static_tfs.back();
  axis_tf.header.frame_id = "map";
  axis_tf.child_frame_id = "axis";
  manif::SE3d global_rotation_axis_location{manif::SE3d::Translation{30, 30, 0},
                                            manif::SO3d::Identity()};
  SE3ToTransformMsg(global_rotation_axis_location, axis_tf);

  static_tfs.emplace_back();
  auto& lm1 = static_tfs.back();
  lm1.header.frame_id = "map";
  lm1.child_frame_id = "lm1";
  landmarks.emplace_back(40, 40, 5);
  SE3ToTransformMsg({landmarks.back(), manif::SO3d::Identity()}, lm1);

  static_tfs.emplace_back();
  auto& lm2 = static_tfs.back();
  lm2.header.frame_id = "map";
  lm2.child_frame_id = "lm2";
  landmarks.emplace_back(30, 30 - 10 * sqrt(2), 0);
  SE3ToTransformMsg({landmarks.back(), manif::SO3d::Identity()}, lm2);

  static_tfs.emplace_back();
  auto& lm3 = static_tfs.back();
  lm3.header.frame_id = "map";
  lm3.child_frame_id = "lm3";
  landmarks.emplace_back(20, 20, 10);
  SE3ToTransformMsg({landmarks.back(), manif::SO3d::Identity()}, lm3);

  br_static.sendTransform(static_tfs);

  struct Observation {
    int landmark;
    int pose_id;
    Eigen::Vector3d relative_position;
  };

  std::vector<Observation> observations;

  // manif::SE3d a{manif::SE3d::Translation{1, 1, 0}, manif::SO3d::Identity()};
  manif::SE3d initial_with_respect_to_axis{
      manif::SE3d::Translation{10, 0, 0},
      manif::SO3d(Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ()))};
  manif::SE3d pose_with_respect_to_axis =
      initial_with_respect_to_axis;  // manif::SE3d::Identity();
  manif::SE3d pose =
      global_rotation_axis_location * initial_with_respect_to_axis;
  manif::SE3d pose_prev = pose;
  manif::SE3d pose_drifty = manif::SE3d::Identity();

  std::cout << "SE3d " << pose << " coeffs " << pose.coeffs() << " size "
            << pose.coeffs().size() << std::endl;

  manif::SE3d::Tangent c_log;
  //                   v,      w
  c_log.coeffs() << 0, 0, 1, 0, 0, 1;  // se(3)
  double t_prev = 0;

  std::default_random_engine generator;
  std::normal_distribution<double> dist(0, 1);

  auto path_pub = node->create_publisher<nav_msgs::msg::Path>("path", 10);
  auto path_gt_pub = node->create_publisher<nav_msgs::msg::Path>("path_gt", 10);
  auto marker_pub = node->create_publisher<visualization_msgs::msg::Marker>(
      "observations", 10);
  nav_msgs::msg::Path path;
  path.header.frame_id = "map";
  path.header.stamp = rclcpp::Time();
  auto path_gt = path;
  path.poses.reserve(600);
  path_gt.poses.reserve(600);

  double bias[6];
  for (int i = 0; i < 6; i++) {
    bias[i] = dist(generator);
  }

  const double delta_t = M_PI_4 * (8. / (503. / 4.));
  double t = 0;
  std::vector<manif::SE3d> poses_ground_truth, poses_drifty;
  poses_ground_truth.reserve(600);
  poses_drifty.reserve(600);

  visualization_msgs::msg::Marker observations_marker;
  observations_marker.header.frame_id = "map";
  observations_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
  observations_marker.color.a = 1.0;
  observations_marker.color.r = 1.0;
  observations_marker.color.g = 1.0;
  observations_marker.scale.x = 0.1;
  observations_marker.scale.y = 0.1;

  int pose_id = 0;
  int count_to_observations_moment = 0;
  while (rclcpp::ok()) {
    rclcpp::Time now = node->now();
    tf.header.stamp = now;
    tf_drifty.header.stamp = now;

    std::cout << t << std::endl;

    if (t > M_PI * 8.) break;

    c_log.coeffs()(2) = std::sin(t / 4);

    pose_prev = pose;
    pose_with_respect_to_axis =
        (t - t_prev) * c_log + pose_with_respect_to_axis;
    pose = global_rotation_axis_location * pose_with_respect_to_axis;

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

    if (count_to_observations_moment == 0) {
      for (int l = 0; l < landmarks.size(); l++) {
        observations.emplace_back(
            Observation{l, pose_id, pose.inverse().act(landmarks[l])});
        observations_marker.points.emplace_back();
        TranslationToPoint(pose.translation(),
                           observations_marker.points.back());
        observations_marker.points.emplace_back();
        TranslationToPoint(pose.act(observations.back().relative_position),
                           observations_marker.points.back());
        for (int i = 0; i < 3; i++) {
          observations.back().relative_position(i) += 0.03 * dist(generator);
        }
      }
      marker_pub->publish(observations_marker);
    }

    pose_drifty = pose_drifty + c_log_drift;
    poses_drifty.push_back(pose_drifty);

    path.poses.emplace_back();
    SE3ToGeometryMsgsPose(pose_drifty, path.poses.back());
    path_pub->publish(path);

    path_gt.poses.emplace_back();
    SE3ToGeometryMsgsPose(pose, path_gt.poses.back());
    path_gt_pub->publish(path_gt);

    auto* pose_current = &pose;
    auto* tf_current = &tf;

    for (int i = 0; i < 2; i++) {
      SE3ToTransformMsg(*pose_current, *tf_current);
      br.sendTransform(*tf_current);
      pose_current = &pose_drifty;
      tf_current = &tf_drifty;
    }

    t_prev = t;
    t += delta_t;
    count_to_observations_moment++;
    pose_id++;
    if (count_to_observations_moment == 30) {
      count_to_observations_moment = 0;
    }
    r.sleep();
  }

  std::cout << "Number of poses: " << poses_ground_truth.size() << std::endl;

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
        new ceres::AutoDiffCostFunction<SE3Constraint, 6, 7, 7>(
            new SE3Constraint(10., poses_drifty[i] - poses_drifty[i - 1])),
        nullptr, poses_to_be_optimized[i - 1].data(),
        poses_to_be_optimized[i].data());
  }

  problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<SE3Constraint, 6, 7, 7>(
          new SE3Constraint(10., manif::SE3d::Tangent::Zero())),
      nullptr, poses_to_be_optimized.back().data(),
      poses_to_be_optimized[0].data());

  // Run the solver!
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

  std::cout << "--------------" << std::endl;
  std::cout << "Adding observation residuals" << std::endl;

  for (const auto& observation : observations) {
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<SE3Observation, 3, 7>(
            new SE3Observation(10., landmarks[observation.landmark],
                               observation.relative_position)),
        nullptr, poses_to_be_optimized[observation.pose_id].data());
  }

  problem.SetParameterBlockVariable(poses_to_be_optimized[0].data());
  ceres::Solve(solver_options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n";

  while (rclcpp::ok()) {
    path.header.stamp = node->now();
    path_pub->publish(path);
    path_gt_pub->publish(path_gt);
    r.sleep();
  }
}