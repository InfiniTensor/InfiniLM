#include "../models_registry.hpp"
#include "qwen3_moe_sparse_moe_block.hpp"

#include <string>
#include <exception>
#include <stdexcept>
#include "../../layers/moe/ep/ep_config.hpp"
#include "runtime/runtime.hpp"

namespace infinilm::models::qwen3_moe {

Qwen3MoeSparseMoeBlock::Qwen3MoeSparseMoeBlock(infinilm::runtime::Runtime &runtime,
    std::shared_ptr<config::ModelConfig> model_config, size_t layer_idx,
    const engine::distributed::RankInfo &rank_info, backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config) {
    (void)attention_backend;
    models::validate_selected_construction(model_config, runtime, rank_info, ep_config);
    try {
        gate_ = this->bind_child("gate", std::make_shared<Qwen3MoeTopKRouter>(runtime, model_config, rank_info, ep_config));
        experts_ = this->bind_child("experts", std::make_shared<Qwen3MoeExperts>(runtime, model_config, rank_info, ep_config));
        fused_moe_ = this->bind_child("fused_moe", std::make_shared<layers::moe::FusedMoE>(runtime, model_config, rank_info, ep_config, layer_idx));
        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) runtime.syncStream();
    } catch (...) {
        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) {
            try { runtime.syncStream(); } catch (...) { std::terminate(); }
}
        throw;
    }
}

infinilm::runtime::Tensor Qwen3MoeSparseMoeBlock::forward(
    infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const {
    gate_->validate_runtime(runtime, hidden_states); // Before any view/allocation/submit.
    const auto weights = experts_->moe_weights(runtime);
    ASSERT(hidden_states->ndim() == 3);
    const auto shape = hidden_states->shape();
    infinilm::runtime::Tensor flattened, routing_weights, selected_experts, borrowed, independent, result;
    infinilm::layers::moe::TopKOutput topk;
    try {
        flattened = hidden_states->view({shape[0] * shape[1], shape[2]});
        std::tie(routing_weights, selected_experts) = gate_->forward(runtime, flattened);
        topk = {routing_weights, selected_experts, infinilm::runtime::Tensor()};
        borrowed = fused_moe_->forward(runtime, flattened, topk, weights);
        independent = infinilm::runtime::Tensor::empty(runtime, borrowed->shape(), borrowed->dtype());
        independent->copy_from(runtime, borrowed);
        result = independent->view({shape[0], shape[1], shape[2]});
        runtime.syncStream();
        return result;
    } catch (...) {
        if (runtime.isGraphRecording()) { runtime.cancelGraphRecording(); if (runtime.isGraphRecording()) std::terminate(); }
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

} // namespace infinilm::models::qwen3_moe
