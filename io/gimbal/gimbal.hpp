// gimbal.hpp
#ifndef IO__GIMBAL_HPP
#define IO__GIMBAL_HPP

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "serial/serial.h"
#include "tools/thread_safe_queue.hpp"

// 全局串口互斥锁（定义在 .cpp 中）
extern std::mutex serial_mutex_;

namespace io {

// === 云台 → 视觉（接收包）===
struct __attribute__((packed)) RxPacket {
    uint8_t head[2] = {'S', 'P'};
    uint8_t mode;
    float q[4];             // w, x, y, z
    float yaw;              // 单位需确认（度 or 弧度）
    float yaw_vel;
    float pitch;
    float pitch_vel;
    float bullet_speed;
    uint16_t bullet_count;
    uint16_t crc16;
};
static_assert(sizeof(RxPacket) <= 64, "RxPacket too large");

// === 视觉 → 云台（发送包）===
// 使用完整字段（8个float + mode + crc）
struct __attribute__((packed)) TxPacket {
    uint8_t head[2] = {'S', 'P'};
    uint8_t mode;           // 0: idle, 1: control no fire, 2: control & fire
    float yaw;
    float yaw_vel;
    float yaw_acc;
    float pitch;
    float pitch_vel;
    float pitch_acc;
    uint16_t crc16;
};
static_assert(sizeof(TxPacket) <= 64, "TxPacket too large");

enum class GimbalMode {
    IDLE,
    AUTO_AIM,
    SMALL_BUFF,
    BIG_BUFF
};

// 用于 API 兼容（不含协议头和 CRC）
struct VisionToGimbal {
    uint8_t mode;
    float yaw;
    float yaw_vel;
    float yaw_acc;
    float pitch;
    float pitch_vel;
    float pitch_acc;
};

struct GimbalState {
    float yaw;        // radians
    float yaw_vel;    // rad/s
    float pitch;      // radians
    float pitch_vel;  // rad/s
    float bullet_speed;
    uint16_t bullet_count;
};

class Gimbal {
public:
    explicit Gimbal(const std::string& config_path);
    ~Gimbal();

    GimbalMode mode() const;
    GimbalState state() const;
    std::string str(GimbalMode mode) const;
    Eigen::Quaterniond q(std::chrono::steady_clock::time_point t);

    // 主要接口：发送全部8个参数
    void send(
        bool control, bool fire,
        float yaw, float yaw_vel, float yaw_acc,
        float pitch, float pitch_vel, float pitch_acc
    );
    void send(const VisionToGimbal& data);
    // 兼容接口（可选）
    void send(const TxPacket& pkt);

private:
    serial::Serial serial_;
    std::thread thread_;
    std::atomic<bool> quit_{false};
    mutable std::mutex mutex_; // 保护 mode_, state_
    std::pair<Eigen::Quaterniond, std::chrono::steady_clock::time_point> data_ahead_;
std::pair<Eigen::Quaterniond, std::chrono::steady_clock::time_point> data_behind_;
    bool skip_crc_ = false;
    GimbalMode mode_ = GimbalMode::IDLE;
    GimbalState state_;
    std::chrono::steady_clock::time_point last_valid_packet_time_;
    // 队列存储 (四元数, 时间戳)
    tools::ThreadSafeQueue<std::pair<Eigen::Quaterniond, std::chrono::steady_clock::time_point>>
        queue_{1000};

    // 滑动窗口接收缓冲区
    std::vector<uint8_t> recv_buffer_;
    size_t recv_pos_ = 0;

    void read_thread();
    void reconnect();
};

} // namespace io

#endif // IO__GIMBAL_HPP