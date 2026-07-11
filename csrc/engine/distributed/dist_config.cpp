#include "dist_config.hpp"

namespace infinilm::engine::distributed {
namespace {

const char *allreduce_backend_name(infinicclAllReduceBackend_t backend) {
    switch (backend) {
    case INFINICCL_ALLREDUCE_BACKEND_AUTO:
        return "auto";
    case INFINICCL_ALLREDUCE_BACKEND_NCCL:
        return "nccl";
    case INFINICCL_ALLREDUCE_BACKEND_CUSTOM:
        return "custom";
    default:
        return "unknown";
    }
}

} // namespace

DistConfig::DistConfig()
    : tp_device_ids{0}, allreduce_backend(INFINICCL_ALLREDUCE_BACKEND_NCCL) {}

DistConfig::DistConfig(int tp_size, infinicclAllReduceBackend_t allreduce_backend_)
    : tp_device_ids(tp_size, 0), allreduce_backend(allreduce_backend_) {
    for (int i = 0; i < tp_size; ++i) {
        tp_device_ids[i] = i;
    }
}

DistConfig::DistConfig(const std::vector<int> &tp_device_ids_, infinicclAllReduceBackend_t allreduce_backend_)
    : tp_device_ids(tp_device_ids_), allreduce_backend(allreduce_backend_) {}

DistConfig::operator std::string() const {
    std::string repr = "DistConfig(tp_device_ids=[";
    for (size_t i = 0; i < tp_device_ids.size(); ++i) {
        repr += std::to_string(tp_device_ids[i]);
        if (i != tp_device_ids.size() - 1) {
            repr += ", ";
        }
    }
    repr += "], moe_ep_backend=" + moe_ep_backend;
    repr += ", moe_ep_size=" + std::to_string(moe_ep_size);
    repr += ", allreduce_backend=";
    repr += allreduce_backend_name(allreduce_backend);
    repr += ")";
    return repr;
}

} // namespace infinilm::engine::distributed
