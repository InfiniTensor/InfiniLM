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
    infinicore::Device device;
    // Local tensor parallelism size
    int tp_size;
    // Local tensor parallelism rank number of this rank
    int tp_rank;
    // Local communicator for current node
    infinicclComm_t comm;
    // Inter-node pipeline parallelism size
    int pp_size;
    // Inter-node pipeline parallelism stage number of this node
    int pp_stage;
    // The total number of ranks in the communication world
    int world_size;
    // The global rank id of this rank
    int world_rank;
    // Inter-node communicator
    infinicclComm_t world_comm;

    RankInfo(infinicore::Device _device = infinicore::context::getDevice())
        : device(_device),
          tp_size(1),
          tp_rank(0),
          comm(nullptr),
          pp_size(1),
          pp_stage(0),
          world_size(1),
          world_rank(0),
          world_comm(nullptr){};

    std::string to_string() const {
        std::stringstream ss;
        ss << "RankInfo: device=" << device.toString()
           << ", tp_size=" << tp_size
           << ", tp_rank=" << tp_rank
           << ", pp_size=" << pp_size
           << ", pp_stage=" << pp_stage
           << ", world_size=" << world_size
           << ", world_rank=" << world_rank;
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
    std::vector<infinicclComm_t> world_communicators_;
};

} // namespace infinilm::engine::distributed
