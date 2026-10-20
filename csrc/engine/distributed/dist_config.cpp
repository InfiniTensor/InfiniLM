#include "dist_config.hpp"
#include <stdexcept>

namespace infinilm::engine::distributed {
namespace {
size_t checked_tp_size(int size) {
    if (size < 1) {
        throw std::invalid_argument("tensor parallel size must be at least 1");
    }
    return static_cast<size_t>(size);
}
} // namespace
DistConfig::DistConfig()
    : tp_device_ids{0} {}

DistConfig::DistConfig(int tp_size)
    : tp_device_ids(checked_tp_size(tp_size), 0) {
    for (int i = 0; i < tp_size; ++i) {
        tp_device_ids[i] = i;
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
    repr += "], moe_ep_backend=" + moe_ep_backend + ", moe_ep_size=" + std::to_string(moe_ep_size) + ", pp_size=" + std::to_string(pp_size) + ", pp_stage=" + std::to_string(pp_stage) + ", master_addr=" + master_addr + ", master_port=" + std::to_string(master_port) + ")";
    return repr;
}

} // namespace infinilm::engine::distributed
