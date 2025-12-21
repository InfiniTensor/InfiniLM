#pragma once

#include <string>
#include <vector>

namespace infinilm::engine::distributed {

struct DistConfig {
    // Device IDs for each rank in tensor parallelism
    std::vector<int> tp_device_ids;

    DistConfig();
    explicit DistConfig(int tp_size);
    explicit DistConfig(const std::vector<int> &tp_device_ids_);

    explicit operator std::string() const;
};

} // namespace infinilm::engine::distributed
