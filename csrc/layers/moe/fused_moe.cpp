#include "fused_moe.hpp"

#include "dispatcher/dispatcher_factory.hpp"
#include "ep/ep_config.hpp"
#include "runner/cuda_fused_moe_runner.hpp"

#include "../../global_state/parallel_state.hpp"

#include <stdexcept>

namespace infinilm::layers::moe {

FusedMoE::FusedMoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                   const infinicore::Device &device,
                   size_t layer_id) {
    (void)layer_id;

    const EPConfig ep_config = make_ep_config();
    const size_t num_experts = model_config->get<size_t>("num_experts");
    const ExpertPlacement expert_placement = make_expert_placement(ep_config, num_experts);
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t tp_size = infinilm::global_state::get_tensor_model_parallel_world_size();
    const size_t intermediate_size = model_config->get<size_t>("moe_intermediate_size");
    size_t intermediate_size_per_partition = intermediate_size;
    if (ep_config.backend == EPBackend::Disabled) {
        if (intermediate_size % tp_size != 0) {
            throw std::runtime_error("moe_intermediate_size must be divisible by tensor parallel world size");
        }
        intermediate_size_per_partition = intermediate_size / tp_size;
    }

    dispatcher_ = make_dispatcher(ep_config, num_experts);
    runner_ = std::make_shared<CudaFusedMoeRunner>(
        expert_placement.local_num_experts,
        hidden_size,
        intermediate_size_per_partition,
        model_config->get_or<size_t>("moe_align_block_size", 16));
    dispatcher_->initialize(device, workspace_);
}

infinicore::Tensor FusedMoE::forward(const infinicore::Tensor &hidden_states,
                                     const TopKOutput &topk_output,
                                     const MoeWeights &weights) const {
    auto dispatch_output = dispatcher_->dispatch(hidden_states, topk_output, workspace_);
    auto combine_input = runner_->run(dispatch_output, weights, workspace_);
    return dispatcher_->combine(combine_input, workspace_);
}

} // namespace infinilm::layers::moe
