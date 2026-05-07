#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace infinilm::models::llama_legacy {

inline size_t get_rotary_dim(size_t head_dim, double partial_rotary_factor) {
    if (partial_rotary_factor <= 0.0 || partial_rotary_factor >= 1.0) {
        return head_dim;
    }

    size_t rotary_dim = static_cast<size_t>(std::llround(
        static_cast<double>(head_dim) * partial_rotary_factor));
    rotary_dim = std::clamp(rotary_dim, static_cast<size_t>(2), head_dim);
    if (rotary_dim % 2 != 0) {
        rotary_dim -= 1;
    }
    return std::max(rotary_dim, static_cast<size_t>(2));
}

} // namespace infinilm::models::llama_legacy
