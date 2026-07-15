#include "deepseek_v2_moe.hpp"

#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include "infinicore/ops/moe_argsort_bincount.hpp"
#include "infinicore/ops/moe_expand_input.hpp"
#include "infinicore/ops/moe_silu_and_mul_quant.hpp"
#include "infinicore/ops/moe_sum_vllm.hpp"
#include "infinicore/ops/w16a16_group_gemm.hpp"

#include <stdexcept>

namespace infinilm::models::deepseek_v2 {

DeepseekV2Experts::DeepseekV2Experts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                     const infinicore::Device &device) {
    const auto &dtype = model_config->get_dtype();
    if (dtype != infinicore::DataType::F16 && dtype != infinicore::DataType::BF16) {
        throw std::runtime_error("DeepseekV2Experts requires fp16 or bfloat16 weights");
    }
    if (model_config->get_or<std::string>("hidden_act", "silu") != "silu") {
        throw std::runtime_error("DeepseekV2Experts supports only SiLU activation");
    }

    hidden_size_ = model_config->get<size_t>("hidden_size");
    num_experts_ = model_config->get<size_t>("num_experts");
    num_experts_per_tok_ = model_config->get<size_t>("num_experts_per_tok");
    const size_t moe_intermediate_size = model_config->get<size_t>("moe_intermediate_size");
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    tp_size_ = static_cast<size_t>(rank_info.tp_size);
    communicator_ = rank_info.comm;
    if (moe_intermediate_size % tp_size_ != 0) {
        throw std::runtime_error("DeepseekV2Experts: moe_intermediate_size must be divisible by tp_size");
    }
    local_moe_intermediate_size_ = moe_intermediate_size / tp_size_;

    w1_ = infinicore::Tensor::empty(
        {num_experts_, local_moe_intermediate_size_ * 2, hidden_size_}, dtype, device);
    w2_ = infinicore::Tensor::empty(
        {num_experts_, hidden_size_, local_moe_intermediate_size_}, dtype, device);

    for (size_t expert_id = 0; expert_id < num_experts_; ++expert_id) {
        const auto prefix = std::to_string(expert_id);
        auto gate_weight = w1_->narrow({{0, expert_id, 1}, {1, 0, local_moe_intermediate_size_}})
                               ->view({local_moe_intermediate_size_, hidden_size_});
        auto up_weight = w1_->narrow({{0, expert_id, 1}, {1, local_moe_intermediate_size_, local_moe_intermediate_size_}})
                             ->view({local_moe_intermediate_size_, hidden_size_});
        auto down_weight = w2_->narrow({{0, expert_id, 1}})
                               ->view({hidden_size_, local_moe_intermediate_size_});
        register_parameter(prefix + ".gate_proj.weight",
                           infinicore::nn::Parameter(gate_weight, 0, rank_info.tp_rank, rank_info.tp_size));
        register_parameter(prefix + ".up_proj.weight",
                           infinicore::nn::Parameter(up_weight, 0, rank_info.tp_rank, rank_info.tp_size));
        register_parameter(prefix + ".down_proj.weight",
                           infinicore::nn::Parameter(down_weight, 1, rank_info.tp_rank, rank_info.tp_size));
    }
}

infinicore::Tensor DeepseekV2Experts::forward(const infinicore::Tensor &hidden_states,
                                              const infinicore::Tensor &top_k_index,
                                              const infinicore::Tensor &top_k_weights,
                                              std::optional<infinicore::Tensor> shared_output) const {
    ASSERT(hidden_states->ndim() == 2);
    const size_t num_tokens = hidden_states->size(0);
    const size_t expanded_tokens = num_tokens * num_experts_per_tok_;
    const auto &attn_metadata = infinilm::global_state::get_forward_context().attn_metadata;
    const bool is_decode = attn_metadata.total_sequence_lengths.has_value()
                        && num_tokens == attn_metadata.total_sequence_lengths.value()->size(0);

    auto tokens_per_experts_gpu = infinicore::Tensor::empty(
        {num_experts_}, infinicore::DataType::I32, hidden_states->device());
    auto sorted_indices = infinicore::Tensor::empty(
        {expanded_tokens}, infinicore::DataType::I32, hidden_states->device());
    auto inv_pos = infinicore::Tensor::empty(
        {expanded_tokens}, infinicore::DataType::I32, hidden_states->device());
    infinicore::op::moe_argsort_bincount_with_inv_pos_(
        tokens_per_experts_gpu, sorted_indices, inv_pos, top_k_index, num_experts_);
    auto tokens_per_experts = is_decode
                                ? tokens_per_experts_gpu
                                : tokens_per_experts_gpu->to(infinicore::Device::Type::CPU);

    auto expanded_input = infinicore::Tensor::empty(
        {expanded_tokens, hidden_size_}, hidden_states->dtype(), hidden_states->device());
    infinicore::op::moe_expand_input_with_inv_pos_(
        expanded_input, std::nullopt, hidden_states, inv_pos, num_experts_per_tok_, 128, 0);

    auto gate_up = infinicore::Tensor::empty(
        {expanded_tokens, local_moe_intermediate_size_ * 2}, hidden_states->dtype(), hidden_states->device());
    infinicore::op::w16a16_group_gemm_(gate_up,
                                       expanded_input,
                                       w1_,
                                       tokens_per_experts,
                                       std::nullopt,
                                       std::nullopt,
                                       true,
                                       is_decode);

    auto activated = infinicore::Tensor::empty(
        {expanded_tokens, local_moe_intermediate_size_}, hidden_states->dtype(), hidden_states->device());
    infinicore::op::moe_silu_and_mul_quant_(activated, std::nullopt, gate_up, 0);

    auto expert_output = infinicore::Tensor::empty(
        {expanded_tokens, hidden_size_}, hidden_states->dtype(), hidden_states->device());
    infinicore::op::w16a16_group_gemm_(expert_output,
                                       activated,
                                       w2_,
                                       tokens_per_experts,
                                       sorted_indices,
                                       std::nullopt,
                                       true,
                                       is_decode);

    auto output = infinicore::Tensor::empty(
        {num_tokens, hidden_size_}, hidden_states->dtype(), hidden_states->device());
    infinicore::op::moe_sum_vllm_(output,
                                  expert_output->view({num_tokens, num_experts_per_tok_, hidden_size_}),
                                  top_k_weights,
                                  shared_output);
    if (tp_size_ > 1 && communicator_ != nullptr) {
        infinicore::op::distributed::allreduce_(output, output, INFINICCL_SUM, communicator_);
    }
    return output;
}

DeepseekV2MoE::DeepseekV2MoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                             const infinicore::Device &device) {
    INFINICORE_NN_MODULE_INIT(gate, model_config, device);
    INFINICORE_NN_MODULE_INIT(experts, model_config, device);

    const size_t n_shared_experts = model_config->get_or<size_t>("n_shared_experts", 0);
    has_shared_experts_ = n_shared_experts > 0;
    if (has_shared_experts_) {
        auto shared_config_json = model_config->get_config_json();
        shared_config_json["intermediate_size"] = model_config->get<size_t>("moe_intermediate_size") * n_shared_experts;
        shared_config_json["reduce_results"] = false;
        auto shared_config = std::make_shared<infinilm::config::ModelConfig>(shared_config_json);
        INFINICORE_NN_MODULE_INIT(shared_experts, shared_config, device);
    }
}

infinicore::Tensor DeepseekV2MoE::forward(const infinicore::Tensor &hidden_states) const {
    ASSERT(hidden_states->ndim() == 3);
    const auto shape = hidden_states->shape();
    auto flat_hidden_states = hidden_states->view({shape[0] * shape[1], shape[2]});
    auto [routing_weights, selected_experts] = gate_->forward(flat_hidden_states);

    std::optional<infinicore::Tensor> shared_output;
    if (has_shared_experts_) {
        shared_output = shared_experts_->forward(hidden_states)->view({shape[0] * shape[1], shape[2]});
    }
    return experts_->forward(flat_hidden_states, selected_experts, routing_weights, shared_output)->view(shape);
}

} // namespace infinilm::models::deepseek_v2
