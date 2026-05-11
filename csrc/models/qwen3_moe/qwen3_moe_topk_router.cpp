#include "qwen3_moe_topk_router.hpp"

#include "infinicore/ops.hpp"

namespace infinilm::models::qwen3_moe {

Qwen3MoeTopKRouter::Qwen3MoeTopKRouter(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                       const infinicore::Device &device) {
    const auto &dtype{model_config->get_dtype()};

    size_t hidden_size = model_config->get<size_t>("hidden_size");
    size_t num_experts = model_config->get<size_t>("num_experts");
    num_experts_per_tok_ = model_config->get<size_t>("num_experts_per_tok");
    norm_topk_prob_ = model_config->get<bool>("norm_topk_prob");

    ASSERT((num_experts > 0) && (num_experts_per_tok_ > 0) && (num_experts_per_tok_ <= num_experts));

    INFINICORE_NN_PARAMETER_INIT(weight, ({num_experts, hidden_size}, dtype, device));
}

std::tuple<infinicore::Tensor, infinicore::Tensor> Qwen3MoeTopKRouter::forward(const infinicore::Tensor &hidden_states) const {

    ASSERT(hidden_states->ndim() == 2);

    size_t ntoken = hidden_states->shape()[0];
    auto router_logits = infinicore::op::linear(hidden_states, weight_, std::nullopt, 1.0f);

    auto router_scores = infinicore::Tensor::empty({ntoken, num_experts_per_tok_}, infinicore::DataType::F32, hidden_states->device());
    auto router_indices = infinicore::Tensor::empty({ntoken, num_experts_per_tok_}, infinicore::DataType::I32, hidden_states->device());

    infinicore::op::topksoftmax(router_scores, router_indices, router_logits, num_experts_per_tok_, norm_topk_prob_);

    return std::make_tuple(router_scores, router_indices);
}

} // namespace infinilm::models::qwen3_moe
