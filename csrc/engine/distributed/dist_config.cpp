#include "dist_config.hpp"

#include <stdexcept>

namespace infinilm::engine::distributed {
DistConfig::DistConfig()
    : tp_device_ids{0} {}

DistConfig::DistConfig(int tp_size, int pp_size)
    : tp_device_ids(tp_size * pp_size, 0),
      tensor_parallel_size(tp_size),
      pipeline_parallel_size(pp_size) {
    if (tp_size < 1 || pp_size < 1) {
        throw std::invalid_argument("TP and PP sizes must be positive");
    }
    for (int i = 0; i < tp_size * pp_size; ++i) {
        tp_device_ids[i] = i;
    }
}

DistConfig::DistConfig(const std::vector<int> &tp_device_ids_)
    : tp_device_ids(tp_device_ids_),
      tensor_parallel_size(static_cast<int>(tp_device_ids_.size())) {}

int DistConfig::world_size() const {
    return tensor_parallel_size * pipeline_parallel_size;
}

DistConfig::operator std::string() const {
    std::string repr = "DistConfig(tp_device_ids=[";
    for (size_t i = 0; i < tp_device_ids.size(); ++i) {
        repr += std::to_string(tp_device_ids[i]);
        if (i != tp_device_ids.size() - 1) {
            repr += ", ";
        }
    }
    repr += "], tp_size=" + std::to_string(tensor_parallel_size)
          + ", pp_size=" + std::to_string(pipeline_parallel_size)
          + ", moe_ep_backend=" + moe_ep_backend
          + ", moe_ep_size=" + std::to_string(moe_ep_size) + ")";
    return repr;
}

} // namespace infinilm::engine::distributed
