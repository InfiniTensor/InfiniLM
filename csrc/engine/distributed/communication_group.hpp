#pragma once

#include "dist_config.hpp"

#include <infiniccl.h>
#include <infinicore/context/context.hpp>

#include <sstream>
#include <vector>

namespace infinilm::engine::distributed {

// Communicator each rank will hold
struct RankInfo {
    // Device Type and ID assigned to this rank
    infinicore::Device deivce;
    // Tensor parallelism size
    int tp_size;
    // Tensor parallelism rank number of this rank
    int tp_rank;
    // Communicator handle
    infinicclComm_t comm;

    RankInfo(infinicore::Device _deivce = infinicore::context::getDevice())
        : tp_size(1), tp_rank(0), deivce(_deivce), comm(nullptr){};

    std::string to_string() const {
        std::stringstream ss;
        ss << "RankInfo: device=" << deivce.toString() << ", tp_size=" << tp_size << ", tp_rank=" << tp_rank;
        return ss.str();
    }
};

// The communication group managed by model infer engine
class CommunicationGroup {
public:
    explicit CommunicationGroup(const DistConfig &dist_config, infinicore::Device::Type device_type);

    const DistConfig &get_dist_config() const;

    RankInfo get_rank_info(int rank) const;

    int get_world_size() const;

    ~CommunicationGroup();

protected:
    DistConfig dist_config_;
    infinicore::Device::Type device_type_;
    std::vector<infinicclComm_t> communicators_;
};

} // namespace infinilm::engine::distributed
