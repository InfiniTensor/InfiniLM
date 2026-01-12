#include "dist_config.hpp"

namespace infinilm::engine::distributed {
DistConfig::DistConfig()
    : tp_device_ids{0} {}

DistConfig::DistConfig(int tp_size)
    : tp_device_ids(tp_size, 0) {
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
    repr += "])";
    return repr;
}

} // namespace infinilm::engine::distributed
