#include "communication_group.hpp"
#include "../../utils.hpp"
#include <stdexcept>
#include <unordered_set>

namespace infinilm::engine::distributed {

CommunicationGroup::CommunicationGroup(const DistConfig &dist_config, infinicore::Device::Type device_type)
    : dist_config_(dist_config), device_type_(device_type),
      communicators_(std::vector<infinicclComm_t>(dist_config.tp_device_ids.size(), nullptr)) {

    const size_t tp_size = dist_config_.tp_device_ids.size();
    const size_t pp_size = dist_config_.pp_device_ids.size();
    if (tp_size == 0 || pp_size == 0) {
        throw std::invalid_argument("DistConfig device lists must not be empty");
    }
    if (pp_size > 1 && tp_size != 1) {
        throw std::invalid_argument("Pipeline parallel MVP requires tensor parallel size == 1");
    }

    if (pp_size > 1) {
        std::unordered_set<int> unique_device_ids;
        for (int device_id : dist_config_.pp_device_ids) {
            if (!unique_device_ids.emplace(device_id).second) {
                throw std::invalid_argument(
                    "Pipeline stage device IDs must be unique");
            }
        }
    }

    const bool pipeline_parallel = pp_size > 1;
    const auto &worker_device_ids = pipeline_parallel ? dist_config_.pp_device_ids : dist_config_.tp_device_ids;
    size_t device_count = infinicore::context::getDeviceCount(device_type);
    for (int device_id : worker_device_ids) {
        if (device_id < 0 || static_cast<size_t>(device_id) >= device_count) {
            throw std::runtime_error("infinilm::engine::distributed::CommunicationGroup error, invalid device id " + std::to_string(device_id) + ", device count: " + std::to_string(device_count));
        }
    }

    if (infinicore::context::getDevice().getType() != device_type_) {
        infinicore::context::setDevice(infinicore::Device(device_type_, 0));
    }
    // Pipeline stages do not form a TP collective group in the MVP.
    if (!pipeline_parallel && tp_size > 1) {
        RUN_INFINI(infinicclCommInitAll(
            (infiniDevice_t)infinicore::context::getDevice().getType(),
            communicators_.data(),
            dist_config.tp_device_ids.size(),
            dist_config.tp_device_ids.data()));
    }
}

const DistConfig &CommunicationGroup::get_dist_config() const {
    return dist_config_;
}

RankInfo CommunicationGroup::get_rank_info(int rank) const {
    const bool pipeline_parallel = dist_config_.pp_device_ids.size() > 1;
    const int world_size = get_world_size();
    if (rank < 0 || rank >= world_size) {
        throw std::out_of_range("CommunicationGroup rank is out of range");
    }

    RankInfo info;
    info.global_rank = rank;
    info.world_size = world_size;
    if (pipeline_parallel) {
        info.tp_size = 1;
        info.tp_rank = 0;
        info.pp_size = static_cast<int>(dist_config_.pp_device_ids.size());
        info.pp_rank = rank;
        info.device = infinicore::Device(device_type_, dist_config_.pp_device_ids[rank]);
        info.comm = nullptr;
    } else {
        info.tp_size = static_cast<int>(dist_config_.tp_device_ids.size());
        info.tp_rank = rank;
        info.pp_size = 1;
        info.pp_rank = 0;
        info.device = infinicore::Device(device_type_, dist_config_.tp_device_ids[rank]);
        info.comm = communicators_[rank];
    }
    return info;
}

int CommunicationGroup::get_world_size() const {
    if (dist_config_.pp_device_ids.size() > 1) {
        return static_cast<int>(dist_config_.pp_device_ids.size());
    }
    return static_cast<int>(dist_config_.tp_device_ids.size());
}

CommunicationGroup::~CommunicationGroup() {
    for (auto &comm : communicators_) {
        if (comm != nullptr) {
            infinicclCommDestroy(comm);
        }
    }
}

} // namespace infinilm::engine::distributed
