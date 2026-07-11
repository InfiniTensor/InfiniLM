#pragma once

#include <cstddef>
#include <infiniccl.h>

#include <string>
#include <vector>

namespace infinilm::engine::distributed {

struct DistConfig {
    // Device IDs for each rank in tensor parallelism
    std::vector<int> tp_device_ids;
    std::string moe_ep_backend{"disabled"};
    size_t moe_ep_size{1};
    infinicclAllReduceBackend_t allreduce_backend;

    DistConfig();
    explicit DistConfig(int tp_size, infinicclAllReduceBackend_t allreduce_backend_ = INFINICCL_ALLREDUCE_BACKEND_NCCL);
    explicit DistConfig(const std::vector<int> &tp_device_ids_, infinicclAllReduceBackend_t allreduce_backend_ = INFINICCL_ALLREDUCE_BACKEND_NCCL);

    explicit operator std::string() const;
};

} // namespace infinilm::engine::distributed
