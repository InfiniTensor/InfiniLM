#include "base_ep_dispatcher.hpp"

#include "../../../global_state/parallel_state.hpp"
#include "infinicore/context/context.hpp"

#include <stdexcept>
#include <utility>

namespace infinilm::layers::moe {

BaseEPDispatcher::BaseEPDispatcher(EPConfig ep_config, size_t num_experts)
    : config_(std::move(ep_config)),
      num_experts_(num_experts) {
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
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

void BaseEPDispatcher::initialize(const infinicore::Device &device,
                                  MoeWorkspace &workspace) {
    (void)workspace;
    if (config_.ep_size > 1) {
        (void)expert_map(device);
    }
}

std::vector<size_t> BaseEPDispatcher::equal_split_sizes(size_t local_dim0) const {
    return std::vector<size_t>(config_.ep_size, local_dim0);
}

infinicore::Tensor BaseEPDispatcher::expert_map(const infinicore::Device &device) const {
    if (config_.ep_size == 1) {
        return infinicore::Tensor();
    }
    if (expert_map_ && expert_map_->device().getType() == device.getType() && expert_map_->device().getIndex() == device.getIndex()) {
        return expert_map_;
    }

    const ExpertPlacement placement = make_expert_placement(config_, num_experts_);
    std::vector<int32_t> map(num_experts_, -1);
    for (size_t global_expert = placement.local_expert_start;
         global_expert < placement.local_expert_end;
         ++global_expert) {
        map[global_expert] = static_cast<int32_t>(placement.global_to_local(global_expert));
    }

    if (infinicore::context::isGraphRecording()) {
        throw std::runtime_error("MoE EP expert_map was not initialized before graph capture");
    }
    auto cpu = infinicore::Tensor::from_blob(
        map.data(),
        {num_experts_},
        infinicore::DataType::I32,
        infinicore::Device(infinicore::Device::Type::CPU, 0));
    expert_map_ = infinicore::Tensor::empty({num_experts_}, infinicore::DataType::I32, device);
    expert_map_->copy_from(cpu);
    return expert_map_;
}

} // namespace infinilm::layers::moe
