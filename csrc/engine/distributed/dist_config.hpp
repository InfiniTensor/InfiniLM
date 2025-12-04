#pragma once

#include <vector>

namespace infinilm::engine::distributed {

struct RankInfo {
    // Tensor parallelism size
    int tp_size;
    // Tensor parallelism rank number of this rank
    int tp_rank;
    // Device ID assigned to this rank
    int device_id;

    RankInfo();
    RankInfo(int tp_size_, int tp_rank_, int device_id_);
    RankInfo(int tp_size_, int tp_rank_);
};

struct DistConfig {
    // Device IDs for each rank in tensor parallelism
    std::vector<int> tp_device_ids;

    DistConfig();
    explicit DistConfig(int tp_size);
    explicit DistConfig(const std::vector<int> &tp_device_ids_);

    RankInfo getRankInfo(int rank) const;
};

} // namespace infinilm::engine::distributed
