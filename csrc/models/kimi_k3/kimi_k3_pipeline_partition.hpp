#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace infinilm::models::kimi_k3 {

inline std::pair<size_t, size_t> kimi_k3_pipeline_layer_range(
    size_t num_layers,
    size_t pp_size,
    size_t pp_stage) {
    if (pp_size == 0 || pp_stage >= pp_size) {
        throw std::runtime_error("Kimi K3: invalid pipeline parallel rank");
    }
    if (pp_size == 1) {
        return {0, num_layers};
    }

    // Treat the first-stage embedding/vision stack and the last-stage LM head
    // as one decoder layer each when balancing memory across pipeline stages.
    constexpr size_t endpoint_overhead = 1;
    const size_t virtual_layers = num_layers + 2 * endpoint_overhead;
    const auto to_model_layer = [num_layers](size_t virtual_boundary) {
        if (virtual_boundary <= endpoint_overhead) {
            return size_t{0};
        }
        return std::min(num_layers, virtual_boundary - endpoint_overhead);
    };

    return {
        to_model_layer(virtual_layers * pp_stage / pp_size),
        to_model_layer(virtual_layers * (pp_stage + 1) / pp_size),
    };
}

} // namespace infinilm::models::kimi_k3
