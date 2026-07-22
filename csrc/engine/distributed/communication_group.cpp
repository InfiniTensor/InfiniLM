#include "communication_group.hpp"
#include "../../utils.hpp"
#include "tcp_rendezvous.hpp"

#include <exception>
#include <stdexcept>
#include <thread>

namespace infinilm::engine::distributed {

CommunicationGroup::CommunicationGroup(const DistConfig &dist_config, infinicore::Device::Type device_type)
    : dist_config_(dist_config), device_type_(device_type),
      communicators_(std::vector<infinicclComm_t>(dist_config.tp_device_ids.size(), nullptr)),
      world_communicators_(std::vector<infinicclComm_t>(dist_config.tp_device_ids.size(), nullptr)) {

    if (dist_config_.pp_size < 1) {
        throw std::runtime_error("DistConfig.pp_size must be at least 1");
    }
    if (dist_config_.pp_stage < 0 || dist_config_.pp_stage >= dist_config_.pp_size) {
        throw std::runtime_error("DistConfig.pp_stage must be in [0, pp_size)");
    }

    size_t world_size = dist_config_.tp_device_ids.size();
    size_t device_count = infinicore::context::getDeviceCount(device_type);
    if (device_count < world_size) {
        throw std::runtime_error("infinilm::engine::distributed::CommunicationGroup error, world size is larger than the number of available GPUs. world size: " + std::to_string(world_size) + ", device count: " + std::to_string(device_count));
    }

    if (infinicore::context::getDevice().getType() != device_type_) {
        infinicore::context::setDevice(infinicore::Device(device_type_, 0));
    }
    if (world_size > 1) {
        RUN_INFINI(infinicclCommInitAll(
            (infiniDevice_t)infinicore::context::getDevice().getType(),
            communicators_.data(),
            dist_config.tp_device_ids.size(),
            dist_config.tp_device_ids.data()));
        spdlog::info(
            "Intra-node TP communicator established: node_rank={}, local_ranks={}",
            dist_config_.pp_stage,
            world_size);
    }
    if (dist_config_.pp_size > 1) {
        // Bootstrap one InfiniCCL unique ID across nodes over a short-lived TCP
        // rendezvous. The resulting world communicator is independent of TCP
        // and is shared by PP activation transfers and final-token delivery.
        infinicclUniqueId_t unique_id;
        if (dist_config_.pp_stage == 0) {
            RUN_INFINI(infinicclGetUniqueId(&unique_id));
        }
        broadcast_rendezvous_payload(
            TcpRendezvousConfig{
                dist_config_.master_addr,
                dist_config_.master_port,
                dist_config_.pp_size,
                dist_config_.pp_stage,
            },
            &unique_id,
            sizeof(unique_id));

        const int tp_size = static_cast<int>(dist_config_.tp_device_ids.size());
        const int pp_world_size = dist_config_.pp_size * tp_size;
        std::vector<std::thread> init_threads;
        std::vector<std::exception_ptr> exceptions(tp_size);
        init_threads.reserve(tp_size);
        for (int local_rank = 0; local_rank < tp_size; ++local_rank) {
            init_threads.emplace_back([&, local_rank] {
                try {
                    infinicore::context::setDevice(infinicore::Device(device_type_, dist_config_.tp_device_ids[local_rank]));
                    const int global_rank = dist_config_.pp_stage * tp_size + local_rank;
                    RUN_INFINI(infinicclCommInitRank(&world_communicators_[local_rank],
                                                     pp_world_size,
                                                     unique_id,
                                                     global_rank));
                } catch (...) {
                    exceptions[local_rank] = std::current_exception();
                }
            });
        }
        for (auto &thread : init_threads) {
            thread.join();
        }
        for (auto &exception : exceptions) {
            if (exception) {
                std::rethrow_exception(exception);
            }
        }
        spdlog::info(
            "Global InfiniCCL communicator established: role={}, node_rank={}, nodes={}, local_tp_ranks={}, world_size={}",
            dist_config_.pp_stage == 0 ? "coordinator" : "participant",
            dist_config_.pp_stage,
            dist_config_.pp_size,
            tp_size,
            pp_world_size);
    }
}

const DistConfig &CommunicationGroup::get_dist_config() const {
    return dist_config_;
}

RankInfo CommunicationGroup::get_rank_info(int rank) const {
    RankInfo info;
    info.tp_size = dist_config_.tp_device_ids.size();
    info.tp_rank = rank;
    info.device = infinicore::Device(device_type_, dist_config_.tp_device_ids[rank]);
    info.comm = communicators_[rank];
    info.pp_size = dist_config_.pp_size;
    info.pp_stage = dist_config_.pp_stage;
    info.world_size = dist_config_.pp_size * info.tp_size;
    info.world_rank = dist_config_.pp_stage * info.tp_size + info.tp_rank;
    info.world_comm = world_communicators_[rank];
    return info;
}

int CommunicationGroup::get_world_size() const {
    return dist_config_.tp_device_ids.size();
}

CommunicationGroup::~CommunicationGroup() {
    if (communicators_.size() > 1) {
        for (auto &comm : communicators_) {
            infinicclCommDestroy(comm);
        }
    }
    for (auto &comm : world_communicators_) {
        if (comm != nullptr) {
            infinicclCommDestroy(comm);
        }
    }
}

} // namespace infinilm::engine::distributed
