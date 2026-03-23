// gimbal.cpp
#include "gimbal.hpp"

#include "tools/crc.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

#include <cstring> // memcpy, memmove
#include <mutex>

// 定义全局串口互斥锁
std::mutex serial_mutex_;

namespace io {

Gimbal::Gimbal(const std::string& config_path)
    : quit_(false), recv_pos_(0) {
    last_valid_packet_time_ = std::chrono::steady_clock::now();
    auto yaml = tools::load(config_path);
    auto com_port = tools::read<std::string>(yaml, "com_port");

    // 修复：手动处理 skip_cboard_crc 默认值
    if (yaml["skip_cboard_crc"]) {
        skip_crc_ = tools::read<bool>(yaml, "skip_cboard_crc");
    } else {
        skip_crc_ = false;
    }

    try {
        serial_.setPort(com_port);
        serial_.setBaudrate(115200);
        serial_.setTimeout(0, 0, 10, 0, 0);
        serial_.open();
    } catch (const std::exception& e) {
        tools::logger()->error("[Gimbal] Failed to open serial '{}': {}", com_port, e.what());
        exit(1);
    }

    recv_buffer_.resize(1024);
    thread_ = std::thread(&Gimbal::read_thread, this);
    tools::logger()->info("[Gimbal] Serial opened on {}.", com_port);
}

Gimbal::~Gimbal() {
    quit_ = true;
    if (thread_.joinable()) thread_.join();
    if (serial_.isOpen()) serial_.close();
}

GimbalMode Gimbal::mode() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return mode_;
}

GimbalState Gimbal::state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

std::string Gimbal::str(GimbalMode mode) const {
    switch (mode) {
        case GimbalMode::IDLE:       return "IDLE";
        case GimbalMode::AUTO_AIM:   return "AUTO_AIM";
        case GimbalMode::SMALL_BUFF: return "SMALL_BUFF";
        case GimbalMode::BIG_BUFF:   return "BIG_BUFF";
        default:                     return "INVALID";
    }
}

// 简化版 q()：只返回最新四元数（因队列无 peek/wait_and_pop）
Eigen::Quaterniond Gimbal::q(std::chrono::steady_clock::time_point timestamp) {
    // 初始状态：若 data_behind_ 无效，先填充一个默认值
    static bool initialized = false;
    if (!initialized) {
        data_ahead_ = {Eigen::Quaterniond::Identity(), std::chrono::steady_clock::now()};
        data_behind_ = data_ahead_;
        initialized = true;
    }

    // 如果最新缓存数据仍早于请求时间，尝试从队列拉取新数据
    if (data_behind_.second < timestamp) {
        data_ahead_ = data_behind_; // 向前推进
    }

    // 从队列中不断取数据，直到 data_behind_.time > timestamp
    while (true) {
        std::pair<Eigen::Quaterniond, std::chrono::steady_clock::time_point> temp;
        if (!queue_.try_pop(temp)) {
            // 队列空了，停止拉取
            break;
        }
        if (temp.second > timestamp) {
            // 找到跨越时间戳的数据点
            data_behind_ = temp;
            break;
        } else {
            // 这个点还在目标时间之前，作为新的 ahead
            data_ahead_ = temp;
        }
    }

    // 现在：data_ahead_.time <= timestamp <= data_behind_.time

    const auto& [q_ahead, t_ahead] = data_ahead_;
    const auto& [q_behind, t_behind] = data_behind_;

    // 情况 1：两个时间点相同（或仅有一个有效点）
    if (t_ahead == t_behind) {
        return q_ahead.normalized();
    }

    // 情况 2：请求时间早于所有数据（应很少发生）
    if (timestamp <= t_ahead) {
        return q_ahead.normalized();
    }

    // 情况 3：请求时间晚于最新数据
    if (timestamp >= t_behind) {
        return q_behind.normalized();
    }

    // 情况 4：正常插值区间
    double dt_total = std::chrono::duration<double>(t_behind - t_ahead).count();
    double dt_partial = std::chrono::duration<double>(timestamp - t_ahead).count();
    double k = dt_partial / dt_total;

    // 球面线性插值（SLERP）
    return q_ahead.slerp(k, q_behind).normalized();
}

// ✅ 主要 send 接口：8个参数
void Gimbal::send(
    bool control, bool fire,
    float yaw, float yaw_vel, float yaw_acc,
    float pitch, float pitch_vel, float pitch_acc) {

    TxPacket tx_data{};
    tx_data.mode = control ? (fire ? 2 : 1) : 0;
    tx_data.yaw = yaw;
    tx_data.yaw_vel = yaw_vel;
    tx_data.yaw_acc = yaw_acc;
    tx_data.pitch = pitch;
    tx_data.pitch_vel = pitch_vel;
    tx_data.pitch_acc = pitch_acc;

    tx_data.crc16 = tools::get_crc16(
        reinterpret_cast<uint8_t*>(&tx_data),
        sizeof(tx_data) - sizeof(tx_data.crc16)
    );
    last_valid_packet_time_ = std::chrono::steady_clock::now();
    try {
        std::lock_guard<std::mutex> lock(serial_mutex_);
        serial_.write(reinterpret_cast<uint8_t*>(&tx_data), sizeof(tx_data));
    } catch (const std::exception& e) {
        tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
    }
}

// 兼容接口（可选）
void Gimbal::send(const VisionToGimbal& data) {
    TxPacket pkt{};
    pkt.mode = data.mode;
    pkt.yaw = data.yaw;
    pkt.yaw_vel = data.yaw_vel;
    pkt.yaw_acc = data.yaw_acc;
    pkt.pitch = data.pitch;
    pkt.pitch_vel = data.pitch_vel;
    pkt.pitch_acc = data.pitch_acc;

    pkt.crc16 = tools::get_crc16(
        reinterpret_cast<uint8_t*>(&pkt),
        sizeof(pkt) - sizeof(pkt.crc16)
    );

    last_valid_packet_time_ = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(serial_mutex_);
    serial_.write(reinterpret_cast<uint8_t*>(&pkt), sizeof(pkt));
}

// === 接收线程 ===
void Gimbal::read_thread() {
    tools::logger()->info("[Gimbal] read_thread started.");
    recv_buffer_.resize(1024);
    recv_pos_ = 0;

    while (!quit_) {
        uint8_t temp_buf[256];
        size_t n = 0;

        try {
            n = serial_.read(temp_buf, sizeof(temp_buf));
        } catch (const std::exception& e) {
            tools::logger()->warn("[Gimbal] Serial read error: {}", e.what());
            reconnect();
            recv_pos_ = 0;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        if (n == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        if (recv_pos_ + n > recv_buffer_.size()) {
            tools::logger()->warn("[Gimbal] Buffer overflow! Resetting.");
            recv_pos_ = 0;
            continue;
        }

        std::memcpy(recv_buffer_.data() + recv_pos_, temp_buf, n);
        recv_pos_ += n;

        // === 滑动窗口解析 ===
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

                    if (skip_crc_ || computed_crc == pkt.crc16) {
                        auto t = std::chrono::steady_clock::now();
                        float qw = pkt.q[0];
                        float qx = pkt.q[1];
                        float qy = pkt.q[2];
                        float qz = pkt.q[3];
                        Eigen::Quaterniond q(qw, qx, qy, qz);
                        tools::logger()->info("[Gimbal] Recv q: [{:.6f}, {:.6f}, {:.6f}, {:.6f}]", qw, qx, qy, qz);
                        queue_.push({q, t});

                        {
                            std::lock_guard<std::mutex> lock(mutex_);
                            // 假设 rx_data 中 yaw/pitch 为弧度；若为度，需转换
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
            i++;
        }

        // === 缓冲区滑动 ===
        if (found_any && processed < recv_pos_) {
            size_t remain = recv_pos_ - processed;
            std::memmove(recv_buffer_.data(), recv_buffer_.data() + processed, remain);
            recv_pos_ = remain;
        } else if (!found_any && recv_pos_ > 200) {
            size_t keep = 64;
            if (recv_pos_ > keep) {
                std::memmove(recv_buffer_.data(), recv_buffer_.data() + recv_pos_ - keep, keep);
                recv_pos_ = keep;
            }
        } else if (found_any) {
            recv_pos_ = 0;
        }
    }

    tools::logger()->info("[Gimbal] read_thread stopped.");
    if (std::chrono::steady_clock::now() - last_valid_packet_time_ > std::chrono::milliseconds(300)) {
    tools::logger()->warn("[Gimbal] No valid packet for 300ms. Triggering reconnect.");
    reconnect();
    recv_pos_ = 0;
    last_valid_packet_time_ = std::chrono::steady_clock::now(); // 防止连续触发
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
}
}

void Gimbal::reconnect() {
    tools::logger()->warn("[Gimbal] Attempting to reconnect...");

    if (serial_.isOpen()) {
        try { serial_.close(); } catch (...) {}
    }

    for (int i = 0; i < 10 && !quit_; ++i) {
        try {
            serial_.open();
            tools::logger()->info("[Gimbal] Reconnected successfully.");
            recv_pos_ = 0;
            return;
        } catch (const std::exception& e) {
            tools::logger()->warn("[Gimbal] Reconnect attempt {} failed: {}", i + 1, e.what());
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    tools::logger()->error("[Gimbal] Reconnect failed after 10 attempts.");
}

} // namespace io