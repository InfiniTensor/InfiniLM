#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace infinilm::engine::distributed {

struct DistConfig {
    // Device IDs for each rank in tensor parallelism
    std::vector<int> tp_device_ids;
    // Device IDs for pipeline stages. A single entry keeps the legacy TP-only path.
    std::vector<int> pp_device_ids{0};
    std::string moe_ep_backend{"disabled"};
    size_t moe_ep_size{1};

    DistConfig();
    explicit DistConfig(int tp_size);
    DistConfig(int tp_size, int pp_size);
    explicit DistConfig(const std::vector<int> &tp_device_ids_);

    explicit operator std::string() const;
};

} // namespace infinilm::engine::distributed
