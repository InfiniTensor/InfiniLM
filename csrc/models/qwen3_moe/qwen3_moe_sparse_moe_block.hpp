#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/common_modules.hpp"
#include "infinicore/device.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include "../../engine/distributed/distributed.hpp"

namespace infinilm::models::qwen3_moe {

using Qwen3MoeMLP = infinilm::models::layers::MoeMLP;

/**
 * @brief Sparse MoE (Mixture of Experts) block for Qwen3 MoE
 *
 * Implements the sparse MoE block with multiple expert MLPs and a gating mechanism.
 * This is a placeholder implementation that throws an error when used.
 */
class Qwen3MoeSparseMoeBlock : public infinicore::nn::Module {
public:
    /**
     * @brief Construct Qwen3MoeSparseMoeBlock module
     *
     * Signature matches TemplateDecoderLayer's MLP interface (hidden_size/intermediate_size
     * are unused; MoE reads moe_intermediate_size from config).
     *
     * @param model_config Model configuration
     * @param hidden_size Hidden size (unused, for interface compatibility)
     * @param intermediate_size Intermediate size (unused, for interface compatibility)
     * @param device Device to create tensors on
     * @param rank_info Rank information for distributed training
     */
    Qwen3MoeSparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           size_t hidden_size,
                           size_t intermediate_size,
                           const infinicore::Device &device,
                           engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    /**
     * @brief Forward pass: compute MoE output
     *
     * @param hidden_states Input tensor of shape [batch, seq_len, hidden_size]
     * @return Output tensor of shape [batch, seq_len, hidden_size]
     */
    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

    // Module information
    size_t hidden_size() const { return hidden_size_; }

protected:
    INFINICORE_NN_MODULE(infinicore::nn::Linear, gate);
    INFINICORE_NN_MODULE_VEC(Qwen3MoeMLP, experts);

    engine::distributed::RankInfo rank_info_;
    size_t hidden_size_;
    size_t moe_intermediate_size_;
    size_t num_experts_;
    size_t num_experts_per_tok_;
    bool norm_topk_prob_;
    bool use_bias_;

    std::shared_ptr<infinilm::config::ModelConfig> model_config_;
};

} // namespace infinilm::models::qwen3_moe
