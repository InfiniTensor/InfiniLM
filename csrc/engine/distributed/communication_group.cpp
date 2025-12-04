#include "communication_group.hpp"
#include "../../utils.hpp"

namespace infinilm::engine::distributed {

CommunicationGroup::CommunicationGroup(const DistConfig &dist_config)
    : dist_config_(dist_config),
      communicators_(std::vector<infinicclComm_t>(dist_config.tp_device_ids.size(), nullptr)) {
    if (dist_config_.tp_device_ids.size() > 1) {
        RUN_INFINI(infinicclCommInitAll(
            (infiniDevice_t)infinicore::context::getDevice().getType(),
            communicators_.data(),
            dist_config.tp_device_ids.size(),
            dist_config.tp_device_ids.data()));
    }
}

const DistConfig &CommunicationGroup::getDistConfig() const {
    return dist_config_;
}

RankCommunicator CommunicationGroup::getRankCommunicator(int rank) const {
    RankCommunicator rc;
    rc.info = dist_config_.getRankInfo(rank);
    rc.comm = communicators_[rank];
    return rc;
}

CommunicationGroup::~CommunicationGroup() {
    if (communicators_.size() > 1) {
        for (auto &comm : communicators_) {
            RUN_INFINI(infinicclCommDestroy(comm));
        }
    }
}

} // namespace infinilm::engine::distributed
