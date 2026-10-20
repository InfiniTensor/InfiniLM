#pragma once

#include "../../parallel/topology.hpp"
#include "../../parallel/global_tp_layout_r1261.hpp"
#include "dist_config.hpp"

#include <infiniccl/infiniccl.h>
#include <runtime/device.hpp>
#include <runtime/tensor.hpp>

#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace infinilm::engine::distributed {

// LM submission boundary: caller allocates the output and owns the topology.
// Communicators are borrowed from CommunicationGroup. Models and all recorded
// calls must be drained/destroyed before their group (InferEngine enforces this).
void allreduce_(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor output,
                const infinilm::runtime::Tensor &input, infinicclRedOp_t op,
                infinicclComm_t communicator);
void allgather_(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor output,
                const infinilm::runtime::Tensor &input, infinicclComm_t communicator);
void reduce_scatter_(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor output,
                     const infinilm::runtime::Tensor &input, infinicclRedOp_t op,
                     infinicclComm_t communicator);
void send(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &input,
          int peer, infinicclComm_t communicator);
void recv_(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor output,
           int peer, infinicclComm_t communicator);

// EP split metadata stays in LM. Collectives require uniform counts; unequal
// splits fail before any submission and never trigger a CPU fallback.
void allgatherv_(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor output,
                 const infinilm::runtime::Tensor &input,
                 const std::vector<size_t> &split_sizes,
                 infinicclComm_t communicator);
void allgatherv_many_(infinilm::runtime::Runtime &runtime,
                      const std::vector<infinilm::runtime::Tensor> &outputs,
                      const std::vector<infinilm::runtime::Tensor> &inputs,
                      const std::vector<size_t> &split_sizes,
                      infinicclComm_t communicator);
void reduce_scatterv_(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor output,
                      const infinilm::runtime::Tensor &input,
                      const std::vector<size_t> &split_sizes,
                      infinicclRedOp_t op, infinicclComm_t communicator);
void reduce_scatterv_many_(infinilm::runtime::Runtime &runtime,
                           const std::vector<infinilm::runtime::Tensor> &outputs,
                           const std::vector<infinilm::runtime::Tensor> &inputs,
                           const std::vector<size_t> &split_sizes,
                           infinicclRedOp_t op, infinicclComm_t communicator);

// Communicator each rank will hold
struct RankInfo {
    // Device Type and ID assigned to this rank
    infinilm::runtime::Device device;
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

    explicit RankInfo(infinilm::runtime::Device device)
        : device(std::move(device)),
          tp_size(1),
          tp_rank(0),
          comm(nullptr),
          pp_size(1),
          pp_stage(0),
          world_size(1),
          world_rank(0),
          world_comm(nullptr) {}

    std::string to_string() const {
        std::stringstream ss;
        ss << "RankInfo: device=" << device.ToString()
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
    explicit CommunicationGroup(const DistConfig &dist_config, infinilm::runtime::Device::Type device_type);
    CommunicationGroup(const CommunicationGroup &) = delete;
    CommunicationGroup &operator=(const CommunicationGroup &) = delete;
    CommunicationGroup(CommunicationGroup &&) = delete;
    CommunicationGroup &operator=(CommunicationGroup &&) = delete;

    const DistConfig &get_dist_config() const;

    RankInfo get_rank_info(int rank) const;

    int tp_size() const;

    ~CommunicationGroup();

protected:
    DistConfig dist_config_;
    infinilm::runtime::Device::Type device_type_;
    parallel::CrossNodeTpLayoutR1261 cross_node_tp_;
    parallel::Topology topology_;
    std::vector<infinicclComm_t> communicators_;
    std::vector<infinicclComm_t> world_communicators_;
};

} // namespace infinilm::engine::distributed
