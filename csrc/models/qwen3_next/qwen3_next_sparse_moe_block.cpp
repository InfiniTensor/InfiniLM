#include "execution/submit.hpp"
#include "../../weights/weights.hpp"
#include "../../layers/mlp/fused_activation.hpp"
#include "qwen3_next_sparse_moe_block.hpp"

#include "runtime/runtime.hpp"

#include "../../engine/distributed/communication_group.hpp"

#include <string>
#include <exception>
#include <stdexcept>

namespace infinilm::models::qwen3_next {

Qwen3NextSharedExpert::Qwen3NextSharedExpert(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording() || !model_config || rank_info.device != runtime.device()
        || rank_info.tp_size <= 0 || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size
        || (rank_info.tp_size > 1 && !rank_info.comm))
        throw std::invalid_argument("Next construction requires matching eager Runtime/rank");
    (void)attention_backend; (void)ep_config;
    device_ = runtime.device();
    const auto dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t intermediate_size = model_config->get<size_t>("shared_expert_intermediate_size");
    auto quantization_method = model_config->get_quantization_method();
    try {
        auto register_fn = [this](const std::string &n, infinilm::weights::Weight p) { this->bind_weight(n, std::move(p)); };
        gate_up_proj_ = std::make_shared<layers::linear::GateUpParallelLinear>(
            runtime, hidden_size, intermediate_size, "gate_proj", "up_proj", register_fn,
            quantization_method, false, dtype, rank_info);
        down_proj_ = std::make_shared<layers::linear::RowParallelLinear>(
            runtime, intermediate_size, hidden_size, quantization_method, false, dtype,
            rank_info.tp_rank, rank_info.tp_size, rank_info.comm);
        this->bind_child("down_proj", down_proj_);
    } catch (...) {
        const auto failure = std::current_exception();
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}



infinilm::runtime::Tensor Qwen3NextSharedExpert::forward(infinilm::runtime::Runtime &runtime,
                                                 const infinilm::runtime::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    auto gate_up = gate_up_proj_->forward(runtime, hidden_states_mutable);
    auto activation_shape = gate_up->shape();
    if (activation_shape.empty() || activation_shape.back() % 2 != 0)
        throw std::runtime_error("SiluAndMul input last dim must be even.");
    activation_shape.back() /= 2;
    auto intermediate = infinilm::runtime::Tensor::empty(runtime, activation_shape, gate_up->dtype());
    infinilm::layers::mlp::packed_activation_into(runtime, intermediate, gate_up);
    return down_proj_->forward(runtime, intermediate);
}

Qwen3NextSparseMoeBlock::Qwen3NextSparseMoeBlock(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx, const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording() || !model_config || rank_info.device != runtime.device()
        || rank_info.tp_size <= 0 || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size
        || (rank_info.tp_size > 1 && !rank_info.comm))
        throw std::invalid_argument("Next construction requires matching eager Runtime/rank");
    try {
        gate_ = std::make_shared<infinilm::layers::moe::TopKRouter>(runtime, model_config);
        this->bind_child("gate", gate_);
        experts_ = std::make_shared<infinilm::layers::moe::FusedMoeExperts>(runtime, model_config, rank_info, ep_config);
        this->bind_child("experts", experts_);
        fused_moe_ = std::make_shared<infinilm::layers::moe::FusedMoE>(runtime, model_config, rank_info, ep_config, layer_idx);
        this->bind_child("fused_moe", fused_moe_);
        shared_expert_ = std::make_shared<Qwen3NextSharedExpert>(runtime, model_config, rank_info, attention_backend, ep_config);
        this->bind_child("shared_expert", shared_expert_);
        shared_expert_gate_ = std::make_shared<infinilm::layers::linear::ReplicatedLinear>(
            runtime, model_config->get<size_t>("hidden_size"), 1, false, model_config->get_dtype());
        this->bind_child("shared_expert_gate", shared_expert_gate_);
    } catch (...) {
        const auto failure = std::current_exception();
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}

infinilm::runtime::Tensor Qwen3NextSparseMoeBlock::forward(infinilm::runtime::Runtime &runtime,
                                                   const infinilm::runtime::Tensor &hidden_states) const {
    ASSERT(hidden_states->ndim() == 3);

    auto shape = hidden_states->shape();
    auto hidden_states_reshaped = hidden_states->view({shape[0] * shape[1], shape[2]});

    auto [routing_weights, selected_experts] = gate_->forward(runtime, hidden_states_reshaped);
    infinilm::layers::moe::TopKOutput topk_output{
        routing_weights,
        selected_experts,
        infinilm::runtime::Tensor(),
    };
    auto routed_states = fused_moe_->forward(
        runtime, hidden_states_reshaped,
        topk_output,
        experts_->moe_weights());

    auto shared_states = shared_expert_->forward(runtime, hidden_states);
    auto hidden_states_for_gate = hidden_states;
    const auto ops_input = shared_expert_gate_->forward(runtime, hidden_states_for_gate);
    auto shared_gate = infinilm::runtime::Tensor::empty(runtime, ops_input->shape(), ops_input->dtype());
    infinilm::execution::submit<infini::ops::Sigmoid>(
        runtime, infini::ops::Sigmoid::DefaultConfig(runtime.device().type()), ops_input, shared_gate);
    INFINILM_TRACE_OP(Sigmoid, shared_gate);
    shared_gate = shared_gate->as_strided(shared_states->shape(), {shared_gate->stride(0), shared_gate->stride(1), 0});
    {
        auto ops_output = infinilm::runtime::Tensor::empty(runtime, shared_states->shape(), shared_states->dtype());
        infinilm::execution::submit<infini::ops::Mul>(
            runtime, infini::ops::Mul::DefaultConfig(runtime.device().type()), shared_states, shared_gate, ops_output);
        INFINILM_TRACE_OP(Mul, ops_output);
        shared_states = std::move(ops_output);
    }

    auto routed_states_3d = routed_states->as_strided(
        {shape[0], shape[1], shape[2]},
        {static_cast<infinilm::runtime::Stride>(shape[1] * shape[2]), static_cast<infinilm::runtime::Stride>(shape[2]), 1});
    auto ops_output = infinilm::runtime::Tensor::empty(runtime, routed_states_3d->shape(), routed_states_3d->dtype());
    infinilm::execution::submit<infini::ops::Add>(
        runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), routed_states_3d, shared_states, ops_output);
    INFINILM_TRACE_OP(Add, ops_output);
    return ops_output;
}

} // namespace infinilm::models::qwen3_next
