#include "minimax_text_01_sparse_moe_block.hpp"

#include "../../global_state/global_state.hpp"

#include <infinicore/ops/distributed/allreduce.hpp>
#include <infinicore/ops/fused_moe_mxfp4.hpp>

namespace infinilm::models::minimax_text_01 {

MiniMaxText01SparseMoeBlock::MiniMaxText01SparseMoeBlock(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device,
    size_t layer_id) {
    INFINICORE_NN_MODULE_INIT(gate, model_config, device);
    INFINICORE_NN_MODULE_INIT(experts, model_config, device);
    INFINICORE_NN_MODULE_INIT(fused_moe, model_config, device, layer_id);
}

infinicore::Tensor MiniMaxText01SparseMoeBlock::forward(
    const infinicore::Tensor &hidden_states) const {
    ASSERT(hidden_states->ndim() == 3);

    auto shape = hidden_states->shape();
    auto hidden_states_reshaped = hidden_states->view({shape[0] * shape[1], shape[2]});

    auto [routing_weights, selected_experts] = gate_->forward(hidden_states_reshaped);

    if (experts_->use_mxfp4()) {
        // MXFP4: the per-expert packed weights go through the
        // fused_moe_mxfp4 kernel (activation = swiglu). The kernel does not
        // combine across ranks, so an explicit allreduce is required when
        // TP > 1 (same as kimi_k3).
        const auto &w = experts_->mxfp4_weights();
        auto final_hidden_states = infinicore::op::fused_moe_mxfp4(
            hidden_states_reshaped,
            selected_experts,
            routing_weights,
            w.packed_w13,
            w.w13_scale,
            w.packed_w2,
            w.w2_scale,
            infinicore::op::FusedMoeActivation::Swiglu);
        const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
        if (rank_info.tp_size > 1 && rank_info.comm != nullptr) {
            infinicore::op::distributed::allreduce_(
                final_hidden_states, final_hidden_states, INFINICCL_SUM, rank_info.comm);
        }
        return final_hidden_states->view({shape[0], shape[1], shape[2]});
    }

    infinilm::layers::moe::TopKOutput topk_output{
        routing_weights,
        selected_experts,
        infinicore::Tensor(),
    };

    auto final_hidden_states = fused_moe_->forward(
        hidden_states_reshaped,
        topk_output,
        experts_->moe_weights());

    return final_hidden_states->view({shape[0], shape[1], shape[2]});
}

} // namespace infinilm::models::minimax_text_01
