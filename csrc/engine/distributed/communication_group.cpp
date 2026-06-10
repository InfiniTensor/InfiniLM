#include "communication_group.hpp"
#include "../../utils.hpp"

#include <spdlog/spdlog.h>

// Mirrors InfiniCore ``InfinicclComm`` (infiniccl_impl.h) — only ``comm`` is needed for abort.
struct InfinicclCommLayout {
    infiniDevice_t device_type;
    int device_id;
    void *comm;
};

#if defined(__has_include)
#if __has_include(<nccl.h>)
#include <nccl.h>
#define INFINILM_HAS_NCCL_ABORT 1
#endif
#endif

namespace infinilm::engine::distributed {

CommunicationGroup::CommunicationGroup(const DistConfig &dist_config, infinicore::Device::Type device_type)
    : dist_config_(dist_config), device_type_(device_type),
      communicators_(std::vector<infinicclComm_t>(dist_config.tp_device_ids.size(), nullptr)) {

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
    return info;
}

int CommunicationGroup::get_world_size() const {
    return dist_config_.tp_device_ids.size();
}

void CommunicationGroup::abort_all() {
    if (communicators_.size() <= 1) {
        return;
    }
    for (auto comm_handle : communicators_) {
        if (comm_handle == nullptr) {
            continue;
        }
        auto *layout = reinterpret_cast<InfinicclCommLayout *>(comm_handle);
        if (layout->comm == nullptr) {
            continue;
        }
#ifdef INFINILM_HAS_NCCL_ABORT
        auto result = ncclCommAbort(static_cast<ncclComm_t>(layout->comm));
        if (result != ncclSuccess) {
            spdlog::warn("CommunicationGroup::abort_all: ncclCommAbort failed (code={})", static_cast<int>(result));
        }
#else
        spdlog::warn("CommunicationGroup::abort_all: nccl.h unavailable; skipping communicator abort");
#endif
    }
}

CommunicationGroup::~CommunicationGroup() {
    if (communicators_.size() > 1) {
        for (auto &comm : communicators_) {
            infinicclCommDestroy(comm);
        }
    }
}

} // namespace infinilm::engine::distributed
