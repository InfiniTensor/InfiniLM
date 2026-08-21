#include "fused_moe.hpp"

#include "../moe/kt_moe_callback.hpp"

#include "dispatcher/dispatcher_factory.hpp"
#include "ep/ep_config.hpp"
#include "runner/cuda_fused_moe_runner.hpp"

#include "../../global_state/parallel_state.hpp"

#include <stdexcept>

namespace infinilm::layers::moe {

FusedMoE::FusedMoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                   const infinicore::Device &device,
                   size_t layer_id)
    : layer_id_(layer_id),
      skip_experts_(model_config->get_or<bool>("use_kt_moe", false)) {
    if (skip_experts_) {
        // KT offload: routed experts live on CPU (kt-kernel); skip building
        // the GPU dispatcher/runner entirely. forward() consults the KT
        // callback registry.
        return;
    }

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
    // KT (KTransformers) branch: delegate routed-expert compute to CPU-GPU
    // heterogeneous offload before touching GPU weights.
    {
        auto kt_cb = infinilm::layers::moe::KTMoECallbackRegistry::instance().get(
            static_cast<int>(layer_id_));
        if (kt_cb) {
            return (*kt_cb)(hidden_states, topk_output.topk_weights, topk_output.topk_ids,
                            static_cast<int>(layer_id_));
        }
        if (skip_experts_) {
            throw std::runtime_error(
                "FusedMoE: use_kt_moe is enabled but no KT callback is registered for layer "
                + std::to_string(layer_id_));
        }
    }

    auto dispatch_output = dispatcher_->dispatch(hidden_states, topk_output, workspace_);
    auto combine_input = runner_->run(dispatch_output, weights, workspace_);
    return dispatcher_->combine(combine_input, workspace_);
}

} // namespace infinilm::layers::moe
