#include "dist_config.hpp"

namespace infinilm::engine::distributed {

// ---------------- RankInfo ----------------

RankInfo::RankInfo()
    : tp_size(1), tp_rank(0), device_id(0) {}

RankInfo::RankInfo(int tp_size_, int tp_rank_, int device_id_)
    : tp_size(tp_size_), tp_rank(tp_rank_), device_id(device_id_) {}

RankInfo::RankInfo(int tp_size_, int tp_rank_)
    : RankInfo(tp_size_, tp_rank_, tp_rank_) {}

// ---------------- DistConfig ----------------

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

RankInfo DistConfig::getRankInfo(int rank) const {
    return RankInfo(tp_device_ids.size(), rank, tp_device_ids[rank]);
}

} // namespace infinilm::engine::distributed
