#pragma once

#include "infinicore/tensor.hpp"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include <string>

namespace infinilm::models::llama {

// Helper function to log tensor statistics and sample values
// This is useful for debugging intermediate values in model forward passes
inline void log_tensor_stats(const infinicore::Tensor &tensor, const std::string &name,
                             bool log_samples = true, size_t max_samples = 10) {
    auto shape = tensor->shape();
    auto dtype = tensor->dtype();
    auto device = tensor->device();

    // Log basic info
    std::string shape_str = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) shape_str += ", ";
        shape_str += std::to_string(shape[i]);
    }
    shape_str += "]";

    SPDLOG_INFO("  {}: shape={}, dtype={}, device={}", name, shape_str, static_cast<int>(dtype), device.toString());

    // For F32 tensors, compute and log statistics
    if (dtype == infinicore::DataType::F32) {
        // Copy to CPU if needed and compute stats
        auto cpu_tensor = tensor->to(infinicore::Device(infinicore::Device::Type::CPU, 0));
        std::byte *raw_data = cpu_tensor->data();
        float *data = reinterpret_cast<float*>(raw_data);
        size_t numel = cpu_tensor->numel();

        if (numel > 0) {
            float min_val = *std::min_element(data, data + numel);
            float max_val = *std::max_element(data, data + numel);
            float sum = std::accumulate(data, data + numel, 0.0f);
            float mean_val = sum / static_cast<float>(numel);

            SPDLOG_INFO("    Stats: min={:.6e}, max={:.6e}, mean={:.6e}, numel={}",
                       min_val, max_val, mean_val, numel);

            // Log sample values at specific positions
            if (log_samples && numel > 0) {
                size_t sample_count = std::min(max_samples, numel);
                SPDLOG_INFO("    Sample values (first {}):", sample_count);
                for (size_t i = 0; i < sample_count; ++i) {
                    SPDLOG_INFO("      [{}] = {:.6e}", i, data[i]);
                }
            }
        }
    } else {
        SPDLOG_INFO("  {} (Stats computation skipped for non-F32 tensor)", name);
    }
}

// Helper function to log specific tensor positions (for debugging)
inline void log_tensor_positions(const infinicore::Tensor &tensor, const std::string &name,
                                 const std::vector<std::vector<size_t>> &positions) {
    auto shape = tensor->shape();
    auto dtype = tensor->dtype();

    // Only log for F32 tensors (or copy to CPU)
    if (dtype == infinicore::DataType::F32) {
        auto cpu_tensor = tensor->to(infinicore::Device(infinicore::Device::Type::CPU, 0));
        std::byte *raw_data = cpu_tensor->data();
        float *data = reinterpret_cast<float*>(raw_data);

        SPDLOG_INFO("  {}: Logging specific positions:", name);
        for (const auto &pos : positions) {
            if (pos.size() != shape.size()) {
                SPDLOG_INFO("    Position {}: dimension mismatch (expected {} dims, got {})",
                           pos.size(), shape.size());
                continue;
            }

            // Calculate linear index
            size_t idx = 0;
            size_t stride = 1;
            bool valid = true;
            for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
                if (pos[i] >= shape[i]) {
                    valid = false;
                    break;
                }
                idx += pos[i] * stride;
                stride *= shape[i];
            }

            if (valid && idx < cpu_tensor->numel()) {
                std::string pos_str = "[";
                for (size_t i = 0; i < pos.size(); ++i) {
                    if (i > 0) pos_str += ", ";
                    pos_str += std::to_string(pos[i]);
                }
                pos_str += "]";
                SPDLOG_INFO("    Position {}: value = {:.6e}", pos_str, data[idx]);
            } else {
                std::string pos_str = "[";
                for (size_t i = 0; i < pos.size(); ++i) {
                    if (i > 0) pos_str += ", ";
                    pos_str += std::to_string(pos[i]);
                }
                pos_str += "]";
                SPDLOG_INFO("    Position {}: invalid (out of bounds)", pos_str);
            }
        }
    }
}

} // namespace infinilm::models::llama
