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
    // Tensor parallelism size
    int tp_size;
    // Tensor parallelism rank number of this rank
    int tp_rank;
    int pp_size;
    int pp_rank;
    int world_size;
    int global_rank;
    // Tensor-, pipeline-, and world-parallel communicator handles.
    infinicclComm_t comm;
    infinicclComm_t pp_comm;
    infinicclComm_t world_comm;

    RankInfo(infinicore::Device _device = infinicore::context::getDevice())
        : tp_size(1), tp_rank(0), pp_size(1), pp_rank(0), world_size(1),
          global_rank(0), device(_device), comm(nullptr), pp_comm(nullptr),
          world_comm(nullptr){};

    bool is_pipeline_first_stage() const { return pp_rank == 0; }
    bool is_pipeline_last_stage() const { return pp_rank + 1 == pp_size; }
    bool is_output_rank() const {
        return is_pipeline_last_stage() && tp_rank == 0;
    }

    std::string to_string() const {
        std::stringstream ss;
        ss << "RankInfo: device=" << device.toString()
           << ", global_rank=" << global_rank << "/" << world_size
           << ", tp_rank=" << tp_rank << "/" << tp_size
           << ", pp_rank=" << pp_rank << "/" << pp_size;
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

    int get_output_rank() const;

    ~CommunicationGroup();

protected:
    DistConfig dist_config_;
    infinicore::Device::Type device_type_;
    std::vector<infinicclComm_t> tp_communicators_;
    std::vector<infinicclComm_t> pp_communicators_;
    std::vector<infinicclComm_t> world_communicators_;
};

} // namespace infinilm::engine::distributed
