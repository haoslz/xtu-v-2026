#include "planner.hpp"

#include <vector>

#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

namespace auto_aim
{
Planner::Planner(const std::string & config_path)
: lock_id_(-1), comming_angle_(60 / 57.3), leaving_angle_(45 / 57.3)
{
  auto yaml = tools::load(config_path);
  yaw_offset_ = tools::read<double>(yaml, "yaw_offset") / 57.3;
  pitch_offset_ = tools::read<double>(yaml, "pitch_offset") / 57.3;
  fire_thresh_ = tools::read<double>(yaml, "fire_thresh");
  decision_speed_ = tools::read<double>(yaml, "decision_speed");
  high_speed_delay_time_ = tools::read<double>(yaml, "high_speed_delay_time");
  low_speed_delay_time_ = tools::read<double>(yaml, "low_speed_delay_time");
  
  // 读取小陀螺参数
  if (yaml["comming_angle"].IsDefined()) {
    comming_angle_ = tools::read<double>(yaml, "comming_angle") / 57.3;
  }
  if (yaml["leaving_angle"].IsDefined()) {
    leaving_angle_ = tools::read<double>(yaml, "leaving_angle") / 57.3;
  }

  setup_yaw_solver(config_path);
  setup_pitch_solver(config_path);
}

Plan Planner::plan(Target target, double bullet_speed, const Eigen::Matrix3d & R_gimbal2world)
{
  // 0. Check bullet speed
  if (bullet_speed < 10 || bullet_speed > 25) {
    bullet_speed = 12;
  }

  // 1. 【关键修复】使用持久锁定，防止装甲板在帧间切换
  auto armor_list = target.armor_xyza_list();
  int locked_armor_id = 0;
  auto min_dist = 1e10;
  
  // 找到最近的装甲板
  for (int i = 0; i < (int)armor_list.size(); i++) {
    auto dist = armor_list[i].head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      locked_armor_id = i;
    }
  }
  
  // 使用滞后锁定：只有当新的装甲板明显更近时才切换
  if (lock_id_ >= 0 && lock_id_ < (int)armor_list.size()) {
    double current_dist = armor_list[lock_id_].head<2>().norm();
    double hysteresis = 0.05;  // 50mm 滞后范围
    if (min_dist > current_dist - hysteresis) {
      locked_armor_id = lock_id_;  // 保持原有锁定
    }
  }
  lock_id_ = locked_armor_id;  // 更新锁定
  
  Eigen::Vector3d xyz = armor_list[locked_armor_id].head<3>();
  min_dist = xyz.head<2>().norm();
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z());
  target.predict(bullet_traj.fly_time);

  // 2. Get trajectory - 使用锁定的装甲板ID
  double yaw0;
  Trajectory traj;
  try {
    yaw0 = aim_by_id(target, bullet_speed, R_gimbal2world, locked_armor_id)(0);
    traj = get_trajectory_by_id(target, yaw0, bullet_speed, R_gimbal2world, locked_armor_id);
  } catch (const std::exception & e) {
    tools::logger()->warn("Unsolvable target {:.2f}", bullet_speed);
    return {false};
  }

  // 3. Solve yaw
  Eigen::VectorXd x0(2);
  x0 << traj(0, 0), traj(1, 0);
  tiny_set_x0(yaw_solver_, x0);

  yaw_solver_->work->Xref = traj.block(0, 0, 2, HORIZON);
  tiny_solve(yaw_solver_);

  // 4. Solve pitch
  x0 << traj(2, 0), traj(3, 0);
  tiny_set_x0(pitch_solver_, x0);

  pitch_solver_->work->Xref = traj.block(2, 0, 2, HORIZON);
  tiny_solve(pitch_solver_);

  Plan plan;
  plan.control = true;

  plan.target_yaw = tools::limit_rad(traj(0, HALF_HORIZON) + yaw0);
  plan.target_pitch = traj(2, HALF_HORIZON);

  double raw_yaw = tools::limit_rad(yaw_solver_->work->x(0, HALF_HORIZON) + yaw0);
  double raw_pitch = pitch_solver_->work->x(0, HALF_HORIZON);
  
  // 低通滤波平滑输出
  if (!filter_initialized_) {
    last_yaw_ = raw_yaw;
    last_pitch_ = raw_pitch;
    filter_initialized_ = true;
  }
  
  // 处理 yaw 角度跨越 ±π 的情况
  double yaw_diff = raw_yaw - last_yaw_;
  if (yaw_diff > CV_PI) yaw_diff -= 2 * CV_PI;
  if (yaw_diff < -CV_PI) yaw_diff += 2 * CV_PI;
  
  plan.yaw = tools::limit_rad(last_yaw_ + filter_alpha_ * yaw_diff);
  plan.pitch = last_pitch_ + filter_alpha_ * (raw_pitch - last_pitch_);
  
  last_yaw_ = plan.yaw;
  last_pitch_ = plan.pitch;
  
  plan.yaw_vel = yaw_solver_->work->x(1, HALF_HORIZON);
  plan.yaw_acc = yaw_solver_->work->u(0, HALF_HORIZON);
  plan.pitch_vel = pitch_solver_->work->x(1, HALF_HORIZON);
  plan.pitch_acc = pitch_solver_->work->u(0, HALF_HORIZON);

  auto shoot_offset_ = 2;
  plan.fire =
    std::hypot(
      traj(0, HALF_HORIZON + shoot_offset_) - yaw_solver_->work->x(0, HALF_HORIZON + shoot_offset_),
      traj(2, HALF_HORIZON + shoot_offset_) -
        pitch_solver_->work->x(0, HALF_HORIZON + shoot_offset_)) < fire_thresh_;
  return plan;
}

Plan Planner::plan(std::optional<Target> target, double bullet_speed, const Eigen::Matrix3d & R_gimbal2world)
{
  if (!target.has_value()) return {false};

  double delay_time =
    std::abs(target->ekf_x()[7]) > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;

  auto future = std::chrono::steady_clock::now() + std::chrono::microseconds(int(delay_time * 1e6));

  target->predict(future);

  return plan(*target, bullet_speed, R_gimbal2world);
}

void Planner::setup_yaw_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto max_yaw_acc = tools::read<double>(yaml, "max_yaw_acc");
  auto Q_yaw = tools::read<std::vector<double>>(yaml, "Q_yaw");
  auto R_yaw = tools::read<std::vector<double>>(yaml, "R_yaw");

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_yaw.data());
  Eigen::Matrix<double, 1, 1> R(R_yaw.data());
  tiny_setup(&yaw_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_yaw_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_yaw_acc);
  tiny_set_bound_constraints(yaw_solver_, x_min, x_max, u_min, u_max);

  yaw_solver_->settings->max_iter = 10;
}

void Planner::setup_pitch_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto max_pitch_acc = tools::read<double>(yaml, "max_pitch_acc");
  auto Q_pitch = tools::read<std::vector<double>>(yaml, "Q_pitch");
  auto R_pitch = tools::read<std::vector<double>>(yaml, "R_pitch");

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_pitch.data());
  Eigen::Matrix<double, 1, 1> R(R_pitch.data());
  tiny_setup(&pitch_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_pitch_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_pitch_acc);
  tiny_set_bound_constraints(pitch_solver_, x_min, x_max, u_min, u_max);

  pitch_solver_->settings->max_iter = 10;
}

Eigen::Matrix<double, 2, 1> Planner::aim(const Target & target, double bullet_speed,
                                          const Eigen::Matrix3d & R_gimbal2world)
{
  Eigen::Vector3d xyz;
  double yaw;
  auto min_dist = 1e10;

  for (auto & xyza : target.armor_xyza_list()) {
    auto dist = xyza.head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      xyz = xyza.head<3>();
      yaw = xyza[3];
    }
  }
  debug_xyza = Eigen::Vector4d(xyz.x(), xyz.y(), xyz.z(), yaw);

  // Convert to gimbal frame before computing azimuth/pitch
  Eigen::Vector3d xyz_in_gimbal = R_gimbal2world.transpose() * xyz;
  auto azim = std::atan2(xyz_in_gimbal.y(), xyz_in_gimbal.x());
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz_in_gimbal.z());
  if (bullet_traj.unsolvable) throw std::runtime_error("Unsolvable bullet trajectory!");

  return {tools::limit_rad(azim + yaw_offset_), -bullet_traj.pitch - pitch_offset_};
}

// 使用锁定的装甲板ID进行瞄准，避免在预测过程中切换装甲板
Eigen::Matrix<double, 2, 1> Planner::aim_by_id(const Target & target, double bullet_speed,
                                               const Eigen::Matrix3d & R_gimbal2world, int armor_id)
{
  auto armor_list = target.armor_xyza_list();
  if (armor_id < 0 || armor_id >= (int)armor_list.size()) {
    armor_id = 0;
  }
  
  auto xyza = armor_list[armor_id];
  Eigen::Vector3d xyz = xyza.head<3>();
  double yaw = xyza[3];
  auto min_dist = xyz.head<2>().norm();
  
  debug_xyza = Eigen::Vector4d(xyz.x(), xyz.y(), xyz.z(), yaw);

  // Convert to gimbal frame before computing azimuth/pitch
  Eigen::Vector3d xyz_in_gimbal = R_gimbal2world.transpose() * xyz;
  auto azim = std::atan2(xyz_in_gimbal.y(), xyz_in_gimbal.x());
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz_in_gimbal.z());
  if (bullet_traj.unsolvable) throw std::runtime_error("Unsolvable bullet trajectory!");

  return {tools::limit_rad(azim + yaw_offset_), -bullet_traj.pitch - pitch_offset_};
}

Trajectory Planner::get_trajectory(Target & target, double yaw0, double bullet_speed,
                                  const Eigen::Matrix3d & R_gimbal2world)
{
  Trajectory traj;

  target.predict(-DT * (HALF_HORIZON + 1));
  auto yaw_pitch_last = aim(target, bullet_speed, R_gimbal2world);

  target.predict(DT);  // [0] = -HALF_HORIZON * DT -> [HHALF_HORIZON] = 0
  auto yaw_pitch = aim(target, bullet_speed, R_gimbal2world);

  for (int i = 0; i < HORIZON; i++) {
    target.predict(DT);
  auto yaw_pitch_next = aim(target, bullet_speed, R_gimbal2world);

    auto yaw_vel = tools::limit_rad(yaw_pitch_next(0) - yaw_pitch_last(0)) / (2 * DT);
    auto pitch_vel = (yaw_pitch_next(1) - yaw_pitch_last(1)) / (2 * DT);

    traj.col(i) << tools::limit_rad(yaw_pitch(0) - yaw0), yaw_vel, yaw_pitch(1), pitch_vel;

    yaw_pitch_last = yaw_pitch;
    yaw_pitch = yaw_pitch_next;
  }

  return traj;
}

// 使用锁定的装甲板ID计算轨迹，确保整个预测过程中使用同一个装甲板
Trajectory Planner::get_trajectory_by_id(Target & target, double yaw0, double bullet_speed,
                                         const Eigen::Matrix3d & R_gimbal2world, int armor_id)
{
  Trajectory traj;

  target.predict(-DT * (HALF_HORIZON + 1));
  auto yaw_pitch_last = aim_by_id(target, bullet_speed, R_gimbal2world, armor_id);

  target.predict(DT);
  auto yaw_pitch = aim_by_id(target, bullet_speed, R_gimbal2world, armor_id);

  for (int i = 0; i < HORIZON; i++) {
    target.predict(DT);
    auto yaw_pitch_next = aim_by_id(target, bullet_speed, R_gimbal2world, armor_id);

    auto yaw_vel = tools::limit_rad(yaw_pitch_next(0) - yaw_pitch_last(0)) / (2 * DT);
    auto pitch_vel = (yaw_pitch_next(1) - yaw_pitch_last(1)) / (2 * DT);

    traj.col(i) << tools::limit_rad(yaw_pitch(0) - yaw0), yaw_vel, yaw_pitch(1), pitch_vel;

    yaw_pitch_last = yaw_pitch;
    yaw_pitch = yaw_pitch_next;
  }

  return traj;
}

Eigen::Vector3d Planner::choose_aim_target(const Target & target)
{
  std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
  auto armor_num = armor_xyza_list.size();
  
  // 如果装甲板未发生过跳变，直接返回第一个
  if (!target.jumped) return armor_xyza_list[0].head(3);

  Eigen::VectorXd ekf_x = target.ekf_x();
  // 计算整车旋转中心的球坐标yaw
  auto center_yaw = std::atan2(ekf_x[2], ekf_x[0]);

  // 计算每个装甲板的相对角度
  std::vector<double> delta_angle_list;
  for (int i = 0; i < armor_num; i++) {
    auto delta_angle = tools::limit_rad(armor_xyza_list[i][3] - center_yaw);
    delta_angle_list.emplace_back(delta_angle);
  }

  // 选择在可射击范围内的装甲板（±60°）
  std::vector<int> id_list;
  for (int i = 0; i < armor_num; i++) {
    if (std::abs(delta_angle_list[i]) > 60 / 57.3) continue;
    id_list.push_back(i);
  }
  
  if (id_list.empty()) {
    return armor_xyza_list[0].head(3);
  }

  // 锁定模式：防止在多个装甲板之间来回切换
  // 增加滞后范围(hysteresis)，防止目标旋转时频繁切换装甲板
  if (id_list.size() > 1) {
    int id0 = id_list[0], id1 = id_list[1];
    
    // 如果已经锁定了某个装甲板
    if (lock_id_ == id0 || lock_id_ == id1) {
      // 只有当另一个装甲板的角度明显更好时才切换（滞后范围：15°）
      double hysteresis_threshold = 15 / 57.3;  // 15度转换为弧度
      double delta0 = std::abs(delta_angle_list[id0]);
      double delta1 = std::abs(delta_angle_list[id1]);
      
      if (lock_id_ == id0 && delta1 < delta0 - hysteresis_threshold) {
        lock_id_ = id1;
      } else if (lock_id_ == id1 && delta0 < delta1 - hysteresis_threshold) {
        lock_id_ = id0;
      }
    } else {
      // 未处于锁定模式时，选择delta_angle绝对值较小的装甲板，进入锁定模式
      lock_id_ = (std::abs(delta_angle_list[id0]) < std::abs(delta_angle_list[id1])) ? id0 : id1;
    }

    return armor_xyza_list[lock_id_].head(3);
  }

  // 只有一个装甲板在可射击范围内时，重置锁定
  lock_id_ = -1;
  return armor_xyza_list[id_list[0]].head(3);
}

}  // namespace auto_aim
