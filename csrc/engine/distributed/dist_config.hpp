#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace infinilm::engine::distributed {

struct DistConfig {
    // Device IDs for all ranks, ordered as contiguous TP groups per PP stage.
    std::vector<int> tp_device_ids;
    int tensor_parallel_size{1};
    int pipeline_parallel_size{1};
    std::string moe_ep_backend{"disabled"};
    size_t moe_ep_size{1};

    DistConfig();
    explicit DistConfig(int tp_size, int pp_size = 1);
    explicit DistConfig(const std::vector<int> &tp_device_ids_);

    int world_size() const;

    explicit operator std::string() const;
};

} // namespace infinilm::engine::distributed
