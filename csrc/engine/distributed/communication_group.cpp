#include "communication_group.hpp"
#include "../../utils.hpp"

namespace infinilm::engine::distributed {

CommunicationGroup::CommunicationGroup(const DistConfig &dist_config, infinicore::Device::Type device_type)
    : dist_config_(dist_config), device_type_(device_type),
      tp_communicators_(dist_config.world_size(), nullptr),
      pp_communicators_(dist_config.world_size(), nullptr),
      world_communicators_(dist_config.pipeline_parallel_size > 1
                               ? dist_config.world_size()
                               : 0,
                           nullptr) {

    const int tp_size = dist_config_.tensor_parallel_size;
    const int pp_size = dist_config_.pipeline_parallel_size;
    const int world_size = dist_config_.world_size();
    if (tp_size < 1 || pp_size < 1
        || static_cast<int>(dist_config_.tp_device_ids.size()) != world_size) {
        throw std::runtime_error(
            "DistConfig device count must equal tensor_parallel_size * pipeline_parallel_size");
    }
    size_t device_count = infinicore::context::getDeviceCount(device_type);
    if (device_count < static_cast<size_t>(world_size)) {
        throw std::runtime_error("infinilm::engine::distributed::CommunicationGroup error, world size is larger than the number of available GPUs. world size: " + std::to_string(world_size) + ", device count: " + std::to_string(device_count));
    }

    if (infinicore::context::getDevice().getType() != device_type_) {
        infinicore::context::setDevice(infinicore::Device(device_type_, 0));
    }
    if (pp_size == 1 && world_size > 1) {
        RUN_INFINI(infinicclCommInitAll(
            (infiniDevice_t)infinicore::context::getDevice().getType(),
            tp_communicators_.data(), world_size,
            dist_config.tp_device_ids.data()));
        return;
    }

    if (pp_size > 1) {
        RUN_INFINI(infinicclCommInitAll(
            (infiniDevice_t)infinicore::context::getDevice().getType(),
            world_communicators_.data(), world_size,
            dist_config.tp_device_ids.data()));

        for (int stage = 0; stage < pp_size; ++stage) {
            std::vector<infinicclComm_t> stage_comms(tp_size, nullptr);
            const int offset = stage * tp_size;
            RUN_INFINI(infinicclCommInitAll(
                (infiniDevice_t)infinicore::context::getDevice().getType(),
                stage_comms.data(), tp_size,
                dist_config.tp_device_ids.data() + offset));
            for (int lane = 0; lane < tp_size; ++lane) {
                tp_communicators_[offset + lane] = stage_comms[lane];
            }
        }

        for (int lane = 0; lane < tp_size; ++lane) {
            std::vector<int> lane_devices(pp_size);
            std::vector<infinicclComm_t> lane_comms(pp_size, nullptr);
            for (int stage = 0; stage < pp_size; ++stage) {
                lane_devices[stage] = dist_config.tp_device_ids[stage * tp_size + lane];
            }
            RUN_INFINI(infinicclCommInitAll(
                (infiniDevice_t)infinicore::context::getDevice().getType(),
                lane_comms.data(), pp_size, lane_devices.data()));
            for (int stage = 0; stage < pp_size; ++stage) {
                pp_communicators_[stage * tp_size + lane] = lane_comms[stage];
            }
        }
    }
}

const DistConfig &CommunicationGroup::get_dist_config() const {
    return dist_config_;
}

RankInfo CommunicationGroup::get_rank_info(int rank) const {
    RankInfo info;
    info.tp_size = dist_config_.tensor_parallel_size;
    info.tp_rank = rank % info.tp_size;
    info.pp_size = dist_config_.pipeline_parallel_size;
    info.pp_rank = rank / info.tp_size;
    info.world_size = dist_config_.world_size();
    info.global_rank = rank;
    info.device = infinicore::Device(device_type_, dist_config_.tp_device_ids[rank]);
    info.comm = tp_communicators_[rank];
    info.pp_comm = pp_communicators_[rank];
    info.world_comm = world_communicators_.empty()
                        ? tp_communicators_[rank]
                        : world_communicators_[rank];
    return info;
}

int CommunicationGroup::get_world_size() const {
    return dist_config_.world_size();
}

int CommunicationGroup::get_output_rank() const {
    return (dist_config_.pipeline_parallel_size - 1)
         * dist_config_.tensor_parallel_size;
}

CommunicationGroup::~CommunicationGroup() {
    for (auto *group : {&pp_communicators_, &tp_communicators_,
                        &world_communicators_}) {
        for (auto &comm : *group) {
            if (comm != nullptr) {
                infinicclCommDestroy(comm);
            }
        }
    }
}

} // namespace infinilm::engine::distributed
