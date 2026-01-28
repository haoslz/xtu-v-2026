#include "gimbal.hpp"

#include "tools/crc.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

namespace io
{
Gimbal::Gimbal(const std::string& config_path)
  : serial_()  // 默认构造（serial::Serial 支持默认构造）
{
    auto yaml = tools::load(config_path);
    auto com_port = tools::read<std::string>(yaml, "com_port");

    try {
        serial_.setPort(com_port);
        serial_.setBaudrate(115200);
        serial_.setTimeout(0, 0, 10, 0, 0);
        serial_.open();
    } catch (const std::exception& e) {
        tools::logger()->error("[Gimbal] Failed to open serial: {}", e.what());
        exit(1);
    }

    thread_ = std::thread(&Gimbal::read_thread, this);
    tools::logger()->info("[Gimbal] Serial opened.");
}

Gimbal::~Gimbal()
{
  quit_ = true;
  if (thread_.joinable()) thread_.join();
  serial_.close();
}

GimbalMode Gimbal::mode() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return mode_;
}

GimbalState Gimbal::state() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

std::string Gimbal::str(GimbalMode mode) const
{
  switch (mode) {
    case GimbalMode::IDLE:
      return "IDLE";
    case GimbalMode::AUTO_AIM:
      return "AUTO_AIM";
    case GimbalMode::SMALL_BUFF:
      return "SMALL_BUFF";
    case GimbalMode::BIG_BUFF:
      return "BIG_BUFF";
    default:
      return "INVALID";
  }
}

Eigen::Quaterniond Gimbal::q(std::chrono::steady_clock::time_point t)
{
  while (true) {
    auto [q_a, t_a] = queue_.pop();
    auto [q_b, t_b] = queue_.front();
    auto t_ab = tools::delta_time(t_a, t_b);
    auto t_ac = tools::delta_time(t_a, t);
    auto k = t_ac / t_ab;
    Eigen::Quaterniond q_c = q_a.slerp(k, q_b).normalized();
    if (t < t_a) return q_c;
    if (!(t_a < t && t <= t_b)) continue;

    return q_c;
  }
}

void Gimbal::send(io::VisionToGimbal VisionToGimbal)
{
  tx_data_.mode = VisionToGimbal.mode;
  tx_data_.yaw = VisionToGimbal.yaw;
  tx_data_.yaw_vel = VisionToGimbal.yaw_vel;
  tx_data_.yaw_acc = VisionToGimbal.yaw_acc;
  tx_data_.pitch = VisionToGimbal.pitch;
  tx_data_.pitch_vel = VisionToGimbal.pitch_vel;
  tx_data_.pitch_acc = VisionToGimbal.pitch_acc;
  tx_data_.crc16 = tools::get_crc16(
    reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));

  try {
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

void Gimbal::send(
  bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
  float pitch_acc)
{
  tx_data_.mode = control ? (fire ? 2 : 1) : 0;
  tx_data_.yaw = yaw;
  tx_data_.yaw_vel = yaw_vel;
  tx_data_.yaw_acc = yaw_acc;
  tx_data_.pitch = pitch;
  tx_data_.pitch_vel = pitch_vel;
  tx_data_.pitch_acc = pitch_acc;
  tx_data_.crc16 = tools::get_crc16(
    reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));

  try {
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

bool Gimbal::read(uint8_t * buffer, size_t size)
{
  try {
    return serial_.read(buffer, size) == size;
  } catch (const std::exception & e) {
    // tools::logger()->warn("[Gimbal] Failed to read serial: {}", e.what());
    return false;
  }
}

void Gimbal::read_thread()
{
    tools::logger()->info("[Gimbal] read_thread started.");
    recv_buffer_.resize(1024);
    recv_pos_ = 0;

    while (!quit_) {
        uint8_t temp_buf[256];
        size_t n = 0;

        try {
            n = serial_.read(temp_buf, sizeof(temp_buf)); // 超时返回 0
        } catch (const std::exception& e) {
            tools::logger()->warn("[Gimbal] Serial read error: {}", e.what());
            reconnect();
            recv_pos_ = 0;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        if (n == 0) {
            // 超时无数据，短休眠
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        // 防缓冲区溢出
        if (recv_pos_ + n > recv_buffer_.size()) {
            tools::logger()->warn("[Gimbal] Buffer overflow! Resetting.");
            recv_pos_ = 0;
            continue;
        }

        // 追加新数据
        std::memcpy(recv_buffer_.data() + recv_pos_, temp_buf, n);
        recv_pos_ += n;

        // === 滑动窗口解析所有可能包 ===
        size_t processed = 0;
        bool found_any = false;

        for (size_t i = 0; i + sizeof(RxPacket) <= recv_pos_; ) {
            if (recv_buffer_[i] == 'S' && recv_buffer_[i + 1] == 'P') {
                RxPacket pkt;
                std::memcpy(&pkt, recv_buffer_.data() + i, sizeof(RxPacket));

                if (pkt.mode <= 3) {
                    uint16_t computed_crc = tools::get_crc16(
                        recv_buffer_.data() + i,
                        sizeof(RxPacket) - sizeof(pkt.crc16)
                    );
                    if (computed_crc == pkt.crc16) {
                        // ✅ 有效包
                        auto t = std::chrono::steady_clock::now();
                        Eigen::Quaterniond q(pkt.q[0], pkt.q[1], pkt.q[2], pkt.q[3]);
                        queue_.push({q, t});

                        {
                            std::lock_guard<std::mutex> lock(mutex_);
                            state_.yaw = pkt.yaw;
                            state_.yaw_vel = pkt.yaw_vel;
                            state_.pitch = pkt.pitch;
                            state_.pitch_vel = pkt.pitch_vel;
                            state_.bullet_speed = pkt.bullet_speed;
                            state_.bullet_count = pkt.bullet_count;

                            switch (pkt.mode) {
                                case 0: mode_ = GimbalMode::IDLE; break;
                                case 1: mode_ = GimbalMode::AUTO_AIM; break;
                                case 2: mode_ = GimbalMode::SMALL_BUFF; break;
                                case 3: mode_ = GimbalMode::BIG_BUFF; break;
                                default: mode_ = GimbalMode::IDLE;
                            }
                        }

                        found_any = true;
                        i += sizeof(RxPacket);
                        processed = i;
                        continue;
                    }
                }
            }
            i++; // 前进一步
        }

        // === 滑动缓冲区 ===
        if (found_any && processed < recv_pos_) {
            size_t remain = recv_pos_ - processed;
            std::memmove(recv_buffer_.data(), recv_buffer_.data() + processed, remain);
            recv_pos_ = remain;
        } else if (!found_any && recv_pos_ > 200) {
            // 长时间无有效包：保留最后 64 字节防断包
            size_t keep = 64;
            if (recv_pos_ > keep) {
                std::memmove(recv_buffer_.data(), recv_buffer_.data() + recv_pos_ - keep, keep);
                recv_pos_ = keep;
            }
        } else if (found_any) {
            recv_pos_ = 0; // 全部处理完
        }
    }

    tools::logger()->info("[Gimbal] read_thread stopped.");
}

void Gimbal::reconnect()
{
    tools::logger()->warn("[Gimbal] Attempting to reconnect serial...");

    if (serial_.isOpen()) {
        try {
            serial_.close();
        } catch (...) {}
    }

    for (int i = 0; i < 10 && !quit_; ++i) {
        try {
            serial_.open();
            tools::logger()->info("[Gimbal] Reconnected successfully.");
            recv_pos_ = 0; // 清空缓冲区
            return;
        } catch (...) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    tools::logger()->error("[Gimbal] Reconnect failed after 10 attempts.");
}
}