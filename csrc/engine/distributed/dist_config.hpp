#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace infinilm::engine::distributed {

struct DistConfig {
    // Device IDs for each local rank in tensor parallelism
    std::vector<int> tp_device_ids;
    std::string moe_ep_backend{"disabled"};
    size_t moe_ep_size{1};

    // Inter-node pipeline parallelism size
    int pp_size{1};
    // Inter-node pipeline parallelism stage number of current node
    int pp_stage{0};
    // Address used to bootstrap distributed communication.
    std::string master_addr{"127.0.0.1"};
    int master_port{29500};

    DistConfig();
    explicit DistConfig(int tp_size);
    explicit DistConfig(const std::vector<int> &tp_device_ids_);

    explicit operator std::string() const;
};

} // namespace infinilm::engine::distributed
