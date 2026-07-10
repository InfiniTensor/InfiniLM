#pragma once

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"
#include "infinicore/tensor.hpp"

#include "../global_state/piecewise_prefill_state.hpp"
#include "../global_state/global_state.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>

namespace infinilm::agent_debug {

inline bool skip_tensor_peek() {
    if (infinicore::context::isGraphRecording()) {
        return true;
    }
    if (infinilm::global_state::get_forward_context().piecewise.compile_capture_active) {
        return true;
    }
    return false;
}

inline uint16_t first_elem_bits(const infinicore::Tensor &t) {
    if (!t || t->numel() == 0 || skip_tensor_peek()) {
        return 0;
    }
    auto on_cpu = t->contiguous()->to(infinicore::Device::cpu());
    infinicore::context::syncStream();
    const auto dt = t->dtype();
    if (dt == infinicore::DataType::F16 || dt == infinicore::DataType::BF16) {
        return *reinterpret_cast<const uint16_t *>(on_cpu->data());
    }
    return 0;
}

inline int64_t first_int64(const infinicore::Tensor &t) {
    if (!t || t->numel() == 0 || skip_tensor_peek()) {
        return -1;
    }
    auto on_cpu = t->contiguous()->to(infinicore::Device::cpu());
    infinicore::context::syncStream();
    return *reinterpret_cast<const int64_t *>(on_cpu->data());
}

inline int32_t first_int32(const infinicore::Tensor &t) {
    if (!t || t->numel() == 0 || skip_tensor_peek()) {
        return -1;
    }
    auto on_cpu = t->contiguous()->to(infinicore::Device::cpu());
    infinicore::context::syncStream();
    return *reinterpret_cast<const int32_t *>(on_cpu->data());
}

inline int32_t last_int32(const infinicore::Tensor &t) {
    if (!t || t->numel() == 0 || skip_tensor_peek()) {
        return -1;
    }
    auto on_cpu = t->contiguous()->to(infinicore::Device::cpu());
    infinicore::context::syncStream();
    const size_t n = on_cpu->numel();
    return reinterpret_cast<const int32_t *>(on_cpu->data())[n - 1];
}

inline uint32_t tensor_checksum_bf16(const infinicore::Tensor &t, size_t max_elems = 8) {
    if (!t || t->numel() == 0 || skip_tensor_peek()) {
        return 0;
    }
    auto on_cpu = t->contiguous()->to(infinicore::Device::cpu());
    infinicore::context::syncStream();
    const size_t n = std::min(max_elems, on_cpu->numel());
    const auto *data = reinterpret_cast<const uint16_t *>(on_cpu->data());
    uint32_t xor_sum = 0;
    for (size_t i = 0; i < n; ++i) {
        xor_sum ^= static_cast<uint32_t>(data[i]) << ((i % 2) * 16);
    }
    return xor_sum;
}

inline bool debug_enabled() {
    const char *v = std::getenv("INFINI_AGENT_DEBUG");
    return v != nullptr && v[0] != '\0' && std::string(v) != "0";
}

inline void log(const char *location, const char *message, const char *hypothesis_id,
                const std::string &data_json, const char *run_id = "pre-fix") {
    if (!debug_enabled()) {
        return;
    }
    const char *path = std::getenv("INFINI_AGENT_DEBUG_LOG");
    const std::string log_path =
        (path != nullptr && path[0] != '\0')
            ? path
            : "/workspace/.cursor/debug-52bf26.log";
    std::ofstream f(log_path, std::ios::app);
    if (!f) {
        return;
    }
    const auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
    f << "{\"sessionId\":\"52bf26\",\"runId\":\"" << run_id << "\",\"hypothesisId\":\"" << hypothesis_id
      << "\",\"location\":\"" << location << "\",\"message\":\"" << message << "\",\"data\":"
      << data_json << ",\"timestamp\":" << ts << "}\n";
}

inline void session_log(const char *location, const char *message, const char *hypothesis_id,
                        const std::string &data_json, const char *run_id = "cg-capture") {
    const char *path = std::getenv("INFINI_DEBUG_SESSION_LOG");
    const std::string log_path = (path != nullptr && path[0] != '\0')
                                     ? path
                                     : "/workspace/.cursor/debug-52bf26.log";
    std::ofstream f(log_path, std::ios::app);
    if (!f) {
        return;
    }
    const auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
    f << "{\"sessionId\":\"52bf26\",\"runId\":\"" << run_id << "\",\"hypothesisId\":\""
      << hypothesis_id << "\",\"location\":\"" << location << "\",\"message\":\"" << message
      << "\",\"data\":" << data_json << ",\"timestamp\":" << ts << "}\n";
}

} // namespace infinilm::agent_debug
