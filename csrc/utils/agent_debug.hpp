#pragma once

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"
#include "infinicore/tensor.hpp"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <string>

namespace infinilm::agent_debug {

inline bool skip_tensor_peek() {
    return infinicore::context::isGraphRecording();
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

inline void log(const char *location, const char *message, const char *hypothesis_id,
                const std::string &data_json, const char *run_id = "pre-fix") {
    // #region agent log
    std::ofstream f("/workspace/.cursor/debug-073e37.log", std::ios::app);
    if (!f) {
        return;
    }
    const auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
    f << "{\"sessionId\":\"073e37\",\"runId\":\"" << run_id << "\",\"hypothesisId\":\""
      << hypothesis_id << "\",\"location\":\"" << location << "\",\"message\":\"" << message
      << "\",\"data\":" << data_json << ",\"timestamp\":" << ts << "}\n";
    // #endregion
}

} // namespace infinilm::agent_debug
