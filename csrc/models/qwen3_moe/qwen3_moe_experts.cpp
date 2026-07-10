#include "qwen3_moe_experts.hpp"

#include "../../config/model_config.hpp"
#include "../../global_state/global_state.hpp"
#include "infinicore/ops.hpp"

#include <optional>
#include <string>

namespace infinilm::models::qwen3_moe {

Qwen3MoeExperts::Qwen3MoeExperts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                 const infinicore::Device &device) {
    const auto &dtype = model_config->get_dtype();
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const int tp_rank = rank_info.tp_rank;
    const int tp_size = rank_info.tp_size;

    num_experts_ = model_config->get<size_t>("num_experts");
    num_experts_per_tok_ = model_config->get<size_t>("num_experts_per_tok");
    hidden_size_ = model_config->get<size_t>("hidden_size");
    const size_t intermediate_size = model_config->get<size_t>("moe_intermediate_size");

    ASSERT((num_experts_ > 0) && (num_experts_per_tok_ > 0) && (num_experts_per_tok_ <= num_experts_));
    ASSERT(intermediate_size % static_cast<size_t>(tp_size) == 0);
    intermediate_size_per_partition_ = intermediate_size / static_cast<size_t>(tp_size);

    INFINICORE_NN_PARAMETER_INIT(w1, ({num_experts_, 2 * intermediate_size_per_partition_, hidden_size_}, dtype, device));
    INFINICORE_NN_PARAMETER_INIT(w2, ({num_experts_, hidden_size_, intermediate_size_per_partition_}, dtype, device));

    for (size_t expert = 0; expert < num_experts_; ++expert) {
        auto gate_weight = w1_
                               ->narrow({{0, expert, 1}, {1, 0, intermediate_size_per_partition_}})
                               ->squeeze(0);
        auto up_weight = w1_
                             ->narrow({{0, expert, 1}, {1, intermediate_size_per_partition_, intermediate_size_per_partition_}})
                             ->squeeze(0);
        auto down_weight = w2_
                               ->narrow({{0, expert, 1}})
                               ->squeeze(0);

        const std::string prefix = std::to_string(expert) + ".";
        this->register_parameter(
            prefix + "gate_proj.weight",
            infinicore::nn::Parameter(gate_weight, 0, tp_rank, tp_size));
        this->register_parameter(
            prefix + "up_proj.weight",
            infinicore::nn::Parameter(up_weight, 0, tp_rank, tp_size));
        this->register_parameter(
            prefix + "down_proj.weight",
            infinicore::nn::Parameter(down_weight, 1, tp_rank, tp_size));
    }
}

infinicore::Tensor Qwen3MoeExperts::forward(const infinicore::Tensor &hidden_states,
                                            const infinicore::Tensor &top_k_index,
                                            const infinicore::Tensor &top_k_weights) const {
    ASSERT(hidden_states->ndim() == 2);
    ASSERT(top_k_index->ndim() == 2 && top_k_weights->ndim() == 2);

    return infinicore::op::fused_moe(hidden_states, top_k_index, top_k_weights, w1_, w2_, std::nullopt, std::nullopt,
                                     infinicore::op::FusedMoeActivation::Swiglu);
}

} // namespace infinilm::models::qwen3_moe
