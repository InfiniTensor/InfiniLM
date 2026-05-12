#pragma once

#include "../qwen3_moe/qwen3_moe_sparse_moe_block.hpp"
#include "qwen3_next_attention.hpp"
#include "qwen3_next_gated_deltanet.hpp"
#include <string>
#include <tuple>

namespace infinilm::models::qwen3_next {
using Qwen3NextSparseMoeBlock = qwen3_moe::Qwen3MoeSparseMoeBlock;

class Qwen3NextDecoderLayer : public infinicore::nn::Module {
public:
    Qwen3NextDecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                          size_t layer_idx,
                          const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(const infinicore::Tensor &positions,
                                                               infinicore::Tensor &hidden_states,
                                                               infinicore::Tensor &residual);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               infinicore::Tensor &hidden_states);

    size_t layer_idx() const { return layer_idx_; }

protected:
    std::shared_ptr<infinicore::nn::RMSNorm> input_layernorm_;
    std::shared_ptr<infinicore::nn::RMSNorm> post_attention_layernorm_;
    std::shared_ptr<Qwen3NextAttention> self_attn_;
    std::shared_ptr<Qwen3NextGatedDeltaNet> linear_attn_;
    std::shared_ptr<Qwen3NextSparseMoeBlock> mlp_;

private:
    size_t layer_idx_;
    std::string layer_type_;
};

} // namespace infinilm::models::qwen3_next
