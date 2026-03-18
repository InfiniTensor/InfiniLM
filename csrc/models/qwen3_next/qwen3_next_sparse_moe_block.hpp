#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/common_modules.hpp"
#include "infinicore/device.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include "../../engine/distributed/distributed.hpp"

namespace infinilm::models::qwen3_next {

using Qwen3NextMLP = infinilm::models::layers::MoeMLP;

/**
 * @brief Sparse MoE (Mixture of Experts) block for Qwen3 MoE
 *
 * Implements the sparse MoE block with multiple expert MLPs and a gating mechanism.
 * This is a placeholder implementation that throws an error when used.
 */
class Qwen3NextSparseMoeBlock : public infinicore::nn::Module {
public:
    /**
     * @brief Construct Qwen3NextSparseMoeBlock module
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
     Qwen3NextSparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           const infinicore::Device &device,
                           engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    /**
     * @brief Forward pass: compute MoE output
     *
     * @param hidden_states Input tensor of shape [batch, seq_len, hidden_size]
     * @return Output tensor of shape [batch, seq_len, hidden_size]
     */
    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;


protected:
    INFINICORE_NN_MODULE(infinicore::nn::Linear, gate);
    INFINICORE_NN_MODULE_VEC(Qwen3NextMLP, experts);
    INFINICORE_NN_MODULE(Qwen3NextMLP, shared_expert);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, shared_expert_gate);
};

} // namespace infinilm::models::qwen3_next
