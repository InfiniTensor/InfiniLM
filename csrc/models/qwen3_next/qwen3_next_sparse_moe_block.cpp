#include "qwen3_next_sparse_moe_block.hpp"

#include "../../global_state/global_state.hpp"

#include <infinicore/ops/add.hpp>
#include <infinicore/ops/distributed/allreduce.hpp>
#include <infinicore/ops/linear.hpp>
#include <infinicore/ops/mul.hpp>
#include <infinicore/ops/sigmoid.hpp>
#include <infinicore/ops/swiglu.hpp>

#include <string>

namespace infinilm::models::qwen3_next {

Qwen3NextSharedExpert::Qwen3NextSharedExpert(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                             const infinicore::Device &device) {
    const auto &dtype{model_config->get_dtype()};
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t intermediate_size = model_config->get<size_t>("shared_expert_intermediate_size");

    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    auto quantization_method = model_config->get_quantization_method();
    auto register_fn = [this](const std::string &n, infinicore::nn::Parameter p) { this->register_parameter(n, std::move(p)); };
    gate_up_proj_ = std::make_shared<infinilm::layers::linear::GateUpParallelLinear>(
        hidden_size,
        intermediate_size,
        "gate_proj",
        "up_proj",
        register_fn,
        quantization_method,
        false,
        dtype,
        device,
        rank_info);
    down_proj_ = this->register_module<infinilm::layers::linear::RowParallelLinear>(
        "down_proj",
        intermediate_size,
        hidden_size,
        quantization_method,
        false,
        dtype,
        device,
        rank_info.tp_rank,
        rank_info.tp_size,
        rank_info.comm);
}

infinicore::Tensor Qwen3NextSharedExpert::forward(const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    auto [gate, up] = gate_up_proj_->forward_split(hidden_states_mutable);
    auto intermediate = infinicore::op::swiglu(up, gate);
    return down_proj_->forward(intermediate);
}

Qwen3NextSparseMoeBlock::Qwen3NextSparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                 const infinicore::Device &device)
    : Qwen3NextSparseMoeBlock(model_config, 0, device) {
}

Qwen3NextSparseMoeBlock::Qwen3NextSparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                 size_t layer_idx,
                                                 const infinicore::Device &device) {
    gate_ = this->register_module<infinilm::layers::moe::TopKRouter>("gate", model_config, device);
    experts_ = this->register_module<infinilm::layers::moe::FusedMoeExperts>("experts", model_config, device);
    fused_moe_ = this->register_module<infinilm::layers::moe::FusedMoE>("fused_moe", model_config, device, layer_idx);
    (void)layer_idx;
    shared_expert_ = this->register_module<Qwen3NextSharedExpert>("shared_expert", model_config, device);
    shared_expert_gate_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>(
        "shared_expert_gate",
        model_config->get<size_t>("hidden_size"),
        1,
        false,
        model_config->get_dtype(),
        device);
}

infinicore::Tensor Qwen3NextSparseMoeBlock::forward(const infinicore::Tensor &hidden_states) const {
    ASSERT(hidden_states->ndim() == 3);

    auto shape = hidden_states->shape();
    auto hidden_states_reshaped = hidden_states->view({shape[0] * shape[1], shape[2]});

    auto [routing_weights, selected_experts] = gate_->forward(hidden_states_reshaped);
    infinilm::layers::moe::TopKOutput topk_output{
        routing_weights,
        selected_experts,
        infinicore::Tensor(),
    };
    // KT (KTransformers) offload is handled inside FusedMoE::forward when
    // a KT callback is registered for this layer.
    auto routed_states = fused_moe_->forward(
        hidden_states_reshaped,
        topk_output,
        experts_->moe_weights());

    auto shared_states = shared_expert_->forward(hidden_states);
    auto hidden_states_for_gate = hidden_states;
    auto shared_gate = infinicore::op::sigmoid(shared_expert_gate_->forward(hidden_states_for_gate));
    shared_gate = shared_gate->as_strided(shared_states->shape(), {shared_gate->stride(0), shared_gate->stride(1), 0});
    shared_states = infinicore::op::mul(shared_states, shared_gate);

    auto routed_states_3d = routed_states->as_strided(
        {shape[0], shape[1], shape[2]},
        {static_cast<infinicore::Stride>(shape[1] * shape[2]), static_cast<infinicore::Stride>(shape[2]), 1});
    return infinicore::op::add(routed_states_3d, shared_states);
}

} // namespace infinilm::models::qwen3_next
