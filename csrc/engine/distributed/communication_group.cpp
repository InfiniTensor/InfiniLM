#include "communication_group.hpp"
#include "../../utils.hpp"

namespace infinilm::engine::distributed {

CommunicationGroup::CommunicationGroup(const DistConfig &dist_config, infinicore::Device::Type device_type)
    : dist_config_(dist_config), device_type_(device_type),
      communicators_(std::vector<infinicclComm_t>(dist_config.tp_device_ids.size(), nullptr)) {
    if (infinicore::context::getDevice().getType() != device_type_) {
        infinicore::context::setDevice(infinicore::Device(device_type_, 0));
    }
    if (dist_config_.tp_device_ids.size() > 1) {
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
    RankInfo info;
    info.tp_size = dist_config_.tp_device_ids.size();
    info.tp_rank = rank;
    info.deivce = infinicore::Device(device_type_, dist_config_.tp_device_ids[rank]);
    info.comm = communicators_[rank];
    return info;
}

int CommunicationGroup::get_world_size() const {
    return dist_config_.tp_device_ids.size();
}

CommunicationGroup::~CommunicationGroup() {
    if (communicators_.size() > 1) {
        for (auto &comm : communicators_) {
            RUN_INFINI(infinicclCommDestroy(comm));
        }
    }
}

} // namespace infinilm::engine::distributed
