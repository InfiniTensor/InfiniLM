#include "deepseek_v4_moe.hpp"
#include "deepseek_v4_linear.hpp"

#include "../../global_state/global_state.hpp"
#include "../../layers/moe/ep/ep_config.hpp"
#include "../../utils.hpp"
#include "deepseek_v4_utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/deepseek_v4_router.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include "infinicore/ops/linear.hpp"

#include "spdlog/spdlog.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace infinilm::models::deepseek_v4 {
namespace {

bool env_flag(const char *name) {
    const char *value = std::getenv(name);
    return value != nullptr && std::string(value) == "1";
}

bool disable_fused_hash_topk_env() {
    static const bool value = env_flag("DSV4_DISABLE_FUSED_HASH_TOPK");
    return value;
}

} // namespace

DeepseekV4TopK::DeepseekV4TopK(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                               size_t layer_idx,
                               const infinicore::Device &device)
    : hidden_size_(model_config->get<size_t>("hidden_size")),
      num_experts_(model_config->get<size_t>("n_routed_experts")),
      num_experts_per_tok_(model_config->get<size_t>("num_experts_per_tok")),
      routed_scaling_(static_cast<float>(model_config->get_or<double>("routed_scaling_factor", 1.0))),
      scoring_func_(model_config->get_or<std::string>("scoring_func", "sqrtsoftplus")),
      norm_topk_prob_(model_config->get_or<bool>("norm_topk_prob", true)),
      layer_idx_(layer_idx) {
    ASSERT(scoring_func_ == "sqrtsoftplus");
    const auto &dtype = model_config->get_dtype();
    INFINICORE_NN_PARAMETER_INIT(weight, ({num_experts_, hidden_size_}, dtype, device));

    const std::string topk_method = model_config->get_or<std::string>("topk_method", "");
    if (topk_method == "noaux_tc") {
        INFINICORE_NN_PARAMETER_INIT(bias, ({num_experts_}, infinicore::DataType::F32, device));
    }
}

std::tuple<infinicore::Tensor, infinicore::Tensor>
DeepseekV4TopK::forward(const infinicore::Tensor &hidden_states) const {
    if (hidden_states->ndim() != 2 || hidden_states->shape()[1] != hidden_size_) {
        throw std::runtime_error("DeepseekV4MoE router: expected hidden_states shape [N,D]");
    }

    infinicore::Tensor router_bias;
    if (has_bias()) {
        router_bias = bias_;
    }

    auto gate_logits = infinicore::op::linear(hidden_states, weight_, std::nullopt, 1.0f);

    return infinicore::op::deepseek_v4_topk_router(gate_logits,
                                                   num_experts_per_tok_,
                                                   norm_topk_prob_,
                                                   router_bias);
}

DeepseekV4HashTopK::DeepseekV4HashTopK(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                       size_t layer_idx,
                                       const infinicore::Device &device)
    : hidden_size_(model_config->get<size_t>("hidden_size")),
      num_experts_(model_config->get<size_t>("n_routed_experts")),
      num_experts_per_tok_(model_config->get<size_t>("num_experts_per_tok")),
      routed_scaling_(static_cast<float>(model_config->get_or<double>("routed_scaling_factor", 1.0))),
      scoring_func_(model_config->get_or<std::string>("scoring_func", "sqrtsoftplus")),
      norm_topk_prob_(model_config->get_or<bool>("norm_topk_prob", true)),
      layer_idx_(layer_idx) {
    ASSERT(scoring_func_ == "sqrtsoftplus");
    const auto &dtype = model_config->get_dtype();
    INFINICORE_NN_PARAMETER_INIT(weight, ({num_experts_, hidden_size_}, dtype, device));

    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    INFINICORE_NN_PARAMETER_INIT(tid2eid, ({vocab_size, num_experts_per_tok_}, infinicore::DataType::I64, device));
}

std::tuple<infinicore::Tensor, infinicore::Tensor>
DeepseekV4HashTopK::forward(const infinicore::Tensor &hidden_states,
                            const infinicore::Tensor &input_ids) const {

    if (hidden_states->ndim() != 2 || hidden_states->shape()[1] != hidden_size_) {
        throw std::runtime_error("DeepseekV4MoE router: expected hidden_states shape [N,D]");
    }
    if (input_ids.empty()) {
        throw std::runtime_error("DeepseekV4HashTopK: hash-routed layer requires input_ids");
    }

    if (input_ids->device() != hidden_states->device()) {
        throw std::runtime_error("DeepseekV4HashTopK: input_ids must be on the same device as hidden_states");
    }

    auto input_ids_view = input_ids->is_contiguous() ? input_ids : input_ids->contiguous();
    if (!disable_fused_hash_topk_env()) {
        try {
            return infinicore::op::deepseek_v4_hash_topk_router(
                hidden_states,
                weight_,
                input_ids_view,
                tid2eid_,
                norm_topk_prob_);
        } catch (const std::exception &e) {
            spdlog::warn("DeepseekV4HashTopK: fused hash topk unavailable, falling back to linear + hash router: {}", e.what());
        }
    }

    auto gate_logits = infinicore::op::linear(hidden_states, weight_, std::nullopt, 1.0f);

    return infinicore::op::deepseek_v4_hash_router(
        gate_logits,
        input_ids_view,
        tid2eid_,
        norm_topk_prob_);
}

DeepseekV4Experts::DeepseekV4Experts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                     const infinicore::Device &device)
    : hidden_size_(model_config->get<size_t>("hidden_size")),
      num_experts_(model_config->get<size_t>("n_routed_experts")),
      num_experts_per_tok_(model_config->get<size_t>("num_experts_per_tok")) {
    const size_t intermediate_size = model_config->get<size_t>("moe_intermediate_size");
    const auto dtype = model_config->get_dtype();
    const bool use_w8a8 = use_deepseek_v4_w8a8_linear(model_config);

    const auto ep_config = infinilm::layers::moe::make_ep_config();
    const auto placement = infinilm::layers::moe::make_expert_placement(ep_config, num_experts_);
    const size_t num_local_experts = placement.local_num_experts;

    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const size_t tp_rank = static_cast<size_t>(rank_info.tp_rank);
    const size_t tp_size = static_cast<size_t>(rank_info.tp_size);
    const bool ep_enabled = ep_config.backend != infinilm::layers::moe::EPBackend::Disabled;
    if (ep_enabled) {
        intermediate_size_per_partition_ = intermediate_size;
    } else {
        ASSERT(intermediate_size % tp_size == 0);
        intermediate_size_per_partition_ = intermediate_size / tp_size;
    }
    const size_t expert_tp_rank = ep_enabled ? 0 : tp_rank;
    const size_t expert_tp_size = ep_enabled ? 1 : tp_size;

    const auto weight_dtype = use_w8a8 ? infinicore::DataType::I8 : dtype;
    w13_weight_ = infinicore::nn::Parameter(
        {num_local_experts, intermediate_size_per_partition_ * 2, hidden_size_},
        weight_dtype,
        device);
    w2_weight_ = infinicore::nn::Parameter(
        {num_local_experts, hidden_size_, intermediate_size_per_partition_},
        weight_dtype,
        device);
    if (use_w8a8) {
        w13_weight_scale_ = infinicore::nn::Parameter(
            {num_local_experts, intermediate_size_per_partition_ * 2, 1},
            infinicore::DataType::F32,
            device);
        w2_weight_scale_ = infinicore::nn::Parameter(
            {num_local_experts, hidden_size_, 1},
            infinicore::DataType::F32,
            device);
    }

    for (size_t local_expert = 0; local_expert < num_local_experts; ++local_expert) {
        const size_t global_expert = placement.local_expert_start + local_expert;
        const std::string prefix = std::to_string(global_expert) + ".";
        auto w1_weight = w13_weight_
                             ->narrow({{0, local_expert, 1}, {1, 0, intermediate_size_per_partition_}})
                             ->squeeze(0);
        auto w3_weight = w13_weight_
                             ->narrow({{0, local_expert, 1}, {1, intermediate_size_per_partition_, intermediate_size_per_partition_}})
                             ->squeeze(0);
        auto w2_weight = w2_weight_
                             ->narrow({{0, local_expert, 1}})
                             ->squeeze(0);
        this->register_parameter(
            prefix + "w1.weight",
            infinicore::nn::Parameter(w1_weight, 0, expert_tp_rank, expert_tp_size));
        this->register_parameter(
            prefix + "w3.weight",
            infinicore::nn::Parameter(w3_weight, 0, expert_tp_rank, expert_tp_size));
        this->register_parameter(
            prefix + "w2.weight",
            infinicore::nn::Parameter(w2_weight, 1, expert_tp_rank, expert_tp_size));
        if (use_w8a8) {
            auto w1_scale = w13_weight_scale_
                                ->narrow({{0, local_expert, 1}, {1, 0, intermediate_size_per_partition_}})
                                ->squeeze(0);
            auto w3_scale = w13_weight_scale_
                                ->narrow({{0, local_expert, 1}, {1, intermediate_size_per_partition_, intermediate_size_per_partition_}})
                                ->squeeze(0);
            auto w2_scale = w2_weight_scale_
                                ->narrow({{0, local_expert, 1}})
                                ->squeeze(0);
            this->register_parameter(
                prefix + "w1.weight_scale",
                infinicore::nn::Parameter(w1_scale, 0, expert_tp_rank, expert_tp_size));
            this->register_parameter(
                prefix + "w3.weight_scale",
                infinicore::nn::Parameter(w3_scale, 0, expert_tp_rank, expert_tp_size));
            this->register_parameter(
                prefix + "w2.weight_scale",
                infinicore::nn::Parameter(w2_scale, -1, 0, 1));
        }
    }

    moe_weights_.packed_w13 = w13_weight_;
    moe_weights_.packed_w2 = w2_weight_;
    if (use_w8a8) {
        moe_weights_.packed_w13_scale = w13_weight_scale_;
        moe_weights_.packed_w2_scale = w2_weight_scale_;
    }
    INFINICORE_NN_MODULE_INIT(fused_moe, model_config, device, 0);
}

infinicore::Tensor DeepseekV4Experts::forward(const infinicore::Tensor &hidden_states,
                                              const infinicore::Tensor &top_k_index,
                                              const infinicore::Tensor &top_k_weights) const {
    if (hidden_states->ndim() != 2 || hidden_states->shape()[1] != hidden_size_) {
        throw std::runtime_error("DeepseekV4Experts: expected hidden_states shape [N,D]");
    }
    infinilm::layers::moe::TopKOutput topk_output{
        top_k_weights,
        top_k_index,
        infinicore::Tensor(),
    };
    return fused_moe_->forward(hidden_states, topk_output, moe_weights_);
}

DeepseekV4MoE::DeepseekV4MoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                             const infinicore::Device &device)
    : DeepseekV4MoE(std::move(model_config), 0, device) {
}

DeepseekV4MoE::DeepseekV4MoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                             size_t layer_idx,
                             const infinicore::Device &device)
    : hidden_size_(model_config->get<size_t>("hidden_size")),
      layer_idx_(layer_idx) {
    const size_t moe_intermediate_size = model_config->get<size_t>("moe_intermediate_size");
    const size_t num_shared_experts = model_config->get_or<size_t>("n_shared_experts", 0);
    const size_t num_hash_layers = model_config->get_or<size_t>("num_hash_layers", 0);
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    tp_size_ = static_cast<size_t>(rank_info.tp_size);
    communicator_ = rank_info.comm;

    if (layer_idx < num_hash_layers) {
        topk_ = this->register_module<DeepseekV4HashTopK>("gate", model_config, layer_idx, device);
    } else {
        topk_ = this->register_module<DeepseekV4TopK>("gate", model_config, layer_idx, device);
    }
    INFINICORE_NN_MODULE_INIT(experts, model_config, device);
    has_shared_experts_ = num_shared_experts > 0;
    if (num_shared_experts > 0) {
        INFINICORE_NN_MODULE_INIT(shared_experts, model_config, moe_intermediate_size * num_shared_experts, device);
    }

    f32_allreduce_ = env_flag("DSV4_FFN_F32_ALLREDUCE");
}

infinicore::Tensor DeepseekV4MoE::forward(const infinicore::Tensor &hidden_states,
                                          const infinicore::Tensor &input_ids) const {
    const auto shape = hidden_states->shape();
    if (shape.size() != 3 || shape[2] != hidden_size_) {
        throw std::runtime_error("DeepseekV4MoE: expected hidden_states shape [B,S,D]");
    }
    auto hidden_flat = hidden_states->view({shape[0] * shape[1], hidden_size_});
    infinicore::Tensor topk_weights;
    infinicore::Tensor topk_ids;
    if (std::holds_alternative<std::shared_ptr<DeepseekV4HashTopK>>(topk_)) {
        std::tie(topk_weights, topk_ids) = std::get<std::shared_ptr<DeepseekV4HashTopK>>(topk_)->forward(hidden_flat, input_ids);
    } else {
        std::tie(topk_weights, topk_ids) = std::get<std::shared_ptr<DeepseekV4TopK>>(topk_)->forward(hidden_flat);
    }
    auto final_hidden_states = experts_->forward(hidden_flat, topk_ids, topk_weights);
    if (has_shared_experts_) {
        final_hidden_states = infinicore::op::add(final_hidden_states, shared_experts_->forward(hidden_flat));
    }
    return final_hidden_states->view(shape);
}

} // namespace infinilm::models::deepseek_v4
