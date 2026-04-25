#ifndef IO__GIMBAL_HPP
#define IO__GIMBAL_HPP

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>

#include "serial/serial.h"
#include "tools/thread_safe_queue.hpp"

namespace io
{
struct __attribute__((packed)) RxPacket
{
  uint8_t head[2] = {'S', 'P'};
  uint8_t mode;
  float q[4];
  float yaw;
  float yaw_vel;
  float pitch;
  float pitch_vel;
  float bullet_speed;
  uint16_t bullet_count;
  uint16_t crc16;
};

static_assert(sizeof(RxPacket) <= 64, "RxPacket too large");

struct __attribute__((packed)) TxPacket
{
  uint8_t head[2] = {'S', 'P'};
  uint8_t mode;  // 0: idle, 1: control no fire, 2: control & fire
  float yaw;
  float yaw_vel;
  float yaw_acc;
  float pitch;
  float pitch_vel;
  float pitch_acc;
  uint16_t crc16;
};

static_assert(sizeof(TxPacket) <= 64, "TxPacket too large");

// 用于 API 兼容（不含协议头和 CRC）
struct VisionToGimbal
{
  uint8_t mode;
  float yaw;
  float yaw_vel;
  float yaw_acc;
  float pitch;
  float pitch_vel;
  float pitch_acc;
};

using GimbalToVision = RxPacket;

enum class GimbalMode
{
  IDLE,        // 空闲
  AUTO_AIM,    // 自瞄
  SMALL_BUFF,  // 小符
  BIG_BUFF     // 大符
};

struct GimbalState
{
  float yaw = 0.0f;
  float yaw_vel = 0.0f;
  float pitch = 0.0f;
  float pitch_vel = 0.0f;
  float bullet_speed = 0.0f;
  uint16_t bullet_count = 0;
};

class Gimbal
{
public:
  Gimbal(const std::string & config_path);

  ~Gimbal();

  GimbalMode mode() const;
  GimbalState state() const;
  std::string str(GimbalMode mode) const;
  Eigen::Quaterniond q(std::chrono::steady_clock::time_point t);

  // 主要接口：发送全部8个参数
  void send(
    bool control, bool fire,
    float yaw, float yaw_vel, float yaw_acc,
    float pitch, float pitch_vel, float pitch_acc);
  void send(const VisionToGimbal & data);
  void send(const TxPacket & pkt);

private:
  serial::Serial serial_;

  std::thread thread_;
  std::atomic<bool> quit_ = false;
  mutable std::mutex mutex_;

  RxPacket rx_data_{};
  TxPacket tx_data_{};

  GimbalMode mode_ = GimbalMode::IDLE;
  GimbalState state_{};
  tools::ThreadSafeQueue<std::tuple<Eigen::Quaterniond, std::chrono::steady_clock::time_point>>
    queue_{1000};

  bool read(uint8_t * buffer, size_t size);
  void read_thread();
  void reconnect();
  std::vector<uint8_t> recv_buffer_;  // 新增：接收缓冲区
  size_t recv_pos_ = 0;               // 新增：缓冲区写入位置
};

}  // namespace io

#endif  // IO__GIMBAL_HPP
