#include "dist_config.hpp"
#include <stdexcept>

namespace infinilm::engine::distributed {
DistConfig::DistConfig()
    : tp_device_ids{0} {}

DistConfig::DistConfig(int tp_size)
    : tp_device_ids(tp_size > 0 ? static_cast<size_t>(tp_size) : 0, 0) {
    if (tp_size < 1) {
        throw std::invalid_argument("tensor parallel size must be >= 1");
    }
    for (int i = 0; i < tp_size; ++i) {
        tp_device_ids[i] = i;
    }
}

DistConfig::DistConfig(int tp_size, int pp_size)
    : tp_device_ids(tp_size > 0 ? static_cast<size_t>(tp_size) : 0, 0),
      pp_device_ids(pp_size > 0 ? static_cast<size_t>(pp_size) : 0, 0) {
    if (tp_size < 1 || pp_size < 1) {
        throw std::invalid_argument("tensor and pipeline parallel sizes must be >= 1");
    }
    for (int i = 0; i < tp_size; ++i) {
        tp_device_ids[i] = i;
    }
    for (int i = 0; i < pp_size; ++i) {
        pp_device_ids[i] = i;
    }
}

DistConfig::DistConfig(const std::vector<int> &tp_device_ids_)
    : tp_device_ids(tp_device_ids_) {}

DistConfig::operator std::string() const {
    std::string repr = "DistConfig(tp_device_ids=[";
    for (size_t i = 0; i < tp_device_ids.size(); ++i) {
        repr += std::to_string(tp_device_ids[i]);
        if (i != tp_device_ids.size() - 1) {
            repr += ", ";
        }
    }
    repr += "], pp_device_ids=[";
    for (size_t i = 0; i < pp_device_ids.size(); ++i) {
        repr += std::to_string(pp_device_ids[i]);
        if (i != pp_device_ids.size() - 1) {
            repr += ", ";
        }
    }
    repr += "], moe_ep_backend=" + moe_ep_backend + ", moe_ep_size=" + std::to_string(moe_ep_size) + ")";
    return repr;
}

} // namespace infinilm::engine::distributed
