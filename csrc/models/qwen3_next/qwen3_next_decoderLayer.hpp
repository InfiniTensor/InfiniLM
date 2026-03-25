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

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(infinicore::Tensor &hidden_states, infinicore::Tensor &residual);

    infinicore::Tensor forward(infinicore::Tensor &hidden_states);

    size_t layer_idx() const { return layer_idx_; }

    void set_rotary_emb(const std::shared_ptr<infinicore::nn::RoPE> &rotary_emb);

protected:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);

    INFINICORE_NN_MODULE(Qwen3NextAttention, self_attn);
    INFINICORE_NN_MODULE(Qwen3NextGatedDeltaNet, linear_attn);
    INFINICORE_NN_MODULE(Qwen3NextSparseMoeBlock, mlp);

private:
    size_t layer_idx_;
    std::string layer_type_;
};

} // namespace infinilm::models::qwen3_next
