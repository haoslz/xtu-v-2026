#include <fcntl.h>
#include <unistd.h>
#include <cstdint>
#include <iostream>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>

#include "io/gimbal/gimbal.hpp"
#include "tools/logger.hpp"
#include "tools/crc.hpp"
#include "tools/exiter.hpp"
#include "tools/plotter.hpp"
#include "tools/math_tools.hpp"

#include <opencv2/opencv.hpp>

// ====== 定义 TxPacket（发送给云台的控制包）======
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

// 全局退出标志
static std::atomic<bool> g_keep_running{true};

void signal_handler(int sig) {
    g_keep_running = false;
}

const std::string keys =
    "{help h usage ? | | 输出命令行参数说明}"
    "{fake f         | false | 是否发送 fake 数据}"
    "{@config-path   | | yaml配置文件路径 }";

using namespace std::chrono_literals;

int main(int argc, char* argv[]) {
    cv::CommandLineParser cli(argc, argv, keys);

    // ====== 模式 1: 持续发送 TxPacket 假数据 ======
    if (cli.get<bool>("fake")) {
        tools::logger()->info("[Fake] Starting continuous TxPacket sender...");

        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        std::string port = "/dev/gimbal";
        int fd = open(port.c_str(), O_WRONLY | O_NOCTTY);
        if (fd < 0) {
            tools::logger()->error("[Fake] Failed to open {}", port);
            return 1;
        }

        uint32_t frame = 0;

        while (g_keep_running.load()) {
            TxPacket pkt{};
            pkt.head[0] = 'S';
            pkt.head[1] = 'P';

            double t = static_cast<double>(frame) * 0.01; // 10ms per frame

            // 填充控制指令（示例：正弦轨迹）
            pkt.mode = 2; // control + fire
            pkt.yaw = static_cast<float>(0.3 * sin(t));
            pkt.yaw_vel = static_cast<float>(0.2 * cos(t));
            pkt.yaw_acc = 0.0f;

            pkt.pitch = static_cast<float>(0.15 * cos(1.2 * t));
            pkt.pitch_vel = static_cast<float>(-0.18 * sin(1.2 * t));
            pkt.pitch_acc = 0.0f;

            // 计算 CRC（不包含 crc16 字段）
            pkt.crc16 = tools::get_crc16(
                reinterpret_cast<const uint8_t*>(&pkt),
                sizeof(TxPacket) - sizeof(pkt.crc16)
            );

            // 打印原始字节
            {
                const uint8_t* raw = reinterpret_cast<const uint8_t*>(&pkt);
                std::string hex_str;
                for (size_t i = 0; i < sizeof(pkt); ++i) {
                    char buf[4];
                    std::snprintf(buf, sizeof(buf), "%02X ", raw[i]);
                    hex_str += buf;
                }
                if (!hex_str.empty()) hex_str.pop_back();
                tools::logger()->debug("[Fake] Raw TxPacket ({} bytes): {}", sizeof(pkt), hex_str);
            }

            // 安全打印字段（避免 packed 引用错误）
            char h0 = pkt.head[0], h1 = pkt.head[1];
            uint8_t mode = pkt.mode;
            float yaw = pkt.yaw, yaw_vel = pkt.yaw_vel, yaw_acc = pkt.yaw_acc;
            float pitch = pkt.pitch, pitch_vel = pkt.pitch_vel, pitch_acc = pkt.pitch_acc;
            uint16_t crc = pkt.crc16;

            tools::logger()->debug(
                "[Fake] TxPacket:\n"
                "  head: '{:c}{:c}'\n"
                "  mode: {}\n"
                "  yaw: {:.3f}, vel: {:.3f}, acc: {:.3f}\n"
                "  pitch: {:.3f}, vel: {:.3f}, acc: {:.3f}\n"
                "  crc16: 0x{:04X}",
                h0, h1, mode,
                yaw, yaw_vel, yaw_acc,
                pitch, pitch_vel, pitch_acc,
                crc
            );

            // 发送
            ssize_t written = write(fd, &pkt, sizeof(pkt));
            if (written != static_cast<ssize_t>(sizeof(pkt))) {
                tools::logger()->warn("[Fake] Partial write");
            }

            frame++;
            std::this_thread::sleep_for(10ms);
        }

        close(fd);
        tools::logger()->info("[Fake] Stopped by user.");
        return 0;
    }

    // ====== 模式 2: 正常运行 gimbal 测试 ======
    auto config_path = cli.get<std::string>("@config-path");
    if (cli.has("help") || config_path.empty()) {
        cli.printMessage();
        return 0;
    }

    tools::Exiter exiter;
    tools::Plotter plotter;
    io::Gimbal gimbal(config_path);

    auto t0 = std::chrono::steady_clock::now();
    auto last_mode = gimbal.mode();
    uint16_t last_bullet_count = 0;

    bool fire = false;
    int fire_count = 0;
    auto fire_stamp = std::chrono::steady_clock::now();
    bool first_fired = false;

    auto test_fire = cli.get<bool>("f");

    while (!exiter.exit()) {
        auto mode = gimbal.mode();
        if (mode != last_mode) {
            tools::logger()->info("Gimbal mode changed: {}", gimbal.str(mode));
            last_mode = mode;
        }

        auto t = std::chrono::steady_clock::now();
        auto state = gimbal.state();
        auto q = gimbal.q(t);
        auto ypr = tools::eulers(q, 2, 1, 0); // YPR: yaw-pitch-roll

        bool fired = (state.bullet_count > last_bullet_count);
        last_bullet_count = state.bullet_count;

        if (!first_fired && fired) {
            first_fired = true;
            tools::logger()->info("Gimbal first fired after: {:.3f}s", tools::delta_time(t, fire_stamp));
        }

        if (fire && fire_count > 20) {
            fire = false;
            fire_count = 0;
        } else if (!fire && fire_count > 100) {
            fire = true;
            fire_count = 0;
            fire_stamp = t;
            first_fired = false;
        }
        fire_count++;

        gimbal.send(true, test_fire && fire, 1.0f, 0, 0, 0, 0, 0);

        nlohmann::json data;
        data["q_yaw"] = ypr[0];
        data["q_pitch"] = ypr[1];
        data["yaw"] = state.yaw;
        data["vyaw"] = state.yaw_vel;
        data["pitch"] = state.pitch;
        data["vpitch"] = state.pitch_vel;
        data["bullet_speed"] = state.bullet_speed;
        data["bullet_count"] = state.bullet_count;
        data["fired"] = fired ? 1 : 0;
        data["fire"] = (test_fire && fire) ? 1 : 0;
        data["t"] = tools::delta_time(t, t0);
        plotter.plot(data);

        std::this_thread::sleep_for(9ms);
    }

    gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
    return 0;
}