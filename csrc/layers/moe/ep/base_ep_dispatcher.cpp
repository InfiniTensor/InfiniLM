#include "base_ep_dispatcher.hpp"

#include "runtime/runtime.hpp"

#include <stdexcept>
#include <utility>

namespace infinilm::layers::moe {

BaseEPDispatcher::BaseEPDispatcher(EPConfig ep_config, size_t num_experts,
                                   const engine::distributed::RankInfo &rank_info)
    : config_(std::move(ep_config)),
      num_experts_(num_experts), device_(rank_info.device) {
    if (static_cast<int>(device_.type()) < 0
        || static_cast<int>(device_.type()) >= static_cast<int>(infinilm::runtime::Device::Type::kCount)
        || device_.index() < 0
        || rank_info.tp_size <= 0 || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size
        || config_.ep_size == 0 || config_.ep_rank >= config_.ep_size || num_experts == 0) {
        throw std::invalid_argument("MoE EP requires valid rank topology and experts");
    }
    if (config_.ep_size != static_cast<size_t>(rank_info.tp_size) || config_.ep_rank != static_cast<size_t>(rank_info.tp_rank)) {
        throw std::runtime_error(
            "MoE EP currently reuses the tensor parallel communication group, "
            "so EP size/rank must match TP size/rank");
    }
    if (config_.ep_size > 1) {
        communicator_ = rank_info.comm;
        if (communicator_ == nullptr) {
            throw std::runtime_error("MoE EP requires a valid communication group when ep_size > 1");
        }
    }
}

void BaseEPDispatcher::initialize(infinilm::runtime::Runtime &runtime,
                                  MoeWorkspace &workspace) {
    (void)workspace;
    runtime.assertOwnerThread();
    if (runtime.device() != device_) {
        throw std::invalid_argument("MoE EP initialization device differs from rank topology");
    }
    if (runtime_owner_ && runtime_owner_.get() != &runtime) {
        throw std::invalid_argument("MoE EP initialization Runtime differs from resource owner");
    }
    runtime_owner_ = runtime.shared_from_this();
    if (config_.ep_size > 1) {
        try {
            (void)expert_map(runtime);
        } catch (...) {
            expert_map_ = infinilm::runtime::Tensor();
            runtime_owner_.reset();
            throw;
        }
    }
}

void BaseEPDispatcher::assertInitializedRuntime(infinilm::runtime::Runtime &runtime) const {
    runtime.assertOwnerThread();
    if (runtime_owner_.get() != &runtime || runtime.device() != device_) {
        throw std::invalid_argument("MoE EP Runtime differs from initialized resource owner");
    }
}

std::vector<size_t> BaseEPDispatcher::equal_split_sizes(size_t local_dim0) const {
    return std::vector<size_t>(config_.ep_size, local_dim0);
}

infinilm::runtime::Tensor BaseEPDispatcher::expert_map(infinilm::runtime::Runtime &runtime) const {
    assertInitializedRuntime(runtime);
    if (config_.ep_size == 1) {
        return infinilm::runtime::Tensor();
    }
    const auto device = runtime.device();
    if (expert_map_ && expert_map_->device().type() == device.type() && expert_map_->device().index() == device.index()) {
        return expert_map_;
    }

    const ExpertPlacement placement = make_expert_placement(config_, num_experts_);
    std::vector<int32_t> map(num_experts_, -1);
    for (size_t global_expert = placement.local_expert_start;
         global_expert < placement.local_expert_end;
         ++global_expert) {
        map[global_expert] = static_cast<int32_t>(placement.global_to_local(global_expert));
    }

    if (runtime.isGraphRecording()) {
        throw std::runtime_error("MoE EP expert_map was not initialized before graph capture");
    }
    auto cpu = infinilm::runtime::Tensor::from_blob(
        map.data(),
        {num_experts_},
        infinilm::runtime::DataType::kInt32,
        infinilm::runtime::Device(infinilm::runtime::Device::Type::kCpu, 0));
    expert_map_ = infinilm::runtime::Tensor::empty(runtime, {num_experts_}, infinilm::runtime::DataType::kInt32);
    expert_map_->copy_from(runtime, cpu);
    return expert_map_;
}

} // namespace infinilm::layers::moe
