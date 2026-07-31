#pragma once

#include "../qwen3_5/qwen3_5_attention.hpp"
#include "../qwen3_next/qwen3_next_gated_deltanet.hpp"
#include "qwen3_5_moe_sparse_moe_block.hpp"

#include <string>
#include <tuple>

namespace infinilm::models::qwen3_5_moe {

class Qwen35MoeDecoderLayer : public infinicore::nn::Module {
public:
    Qwen35MoeDecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                          size_t layer_idx,
                          const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(
        const infinicore::Tensor &positions,
        infinicore::Tensor &hidden_states,
        infinicore::Tensor &residual);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               infinicore::Tensor &hidden_states);

    size_t layer_idx() const { return layer_idx_; }

protected:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(qwen3_5::Qwen35Attention, self_attn);
    INFINICORE_NN_MODULE(qwen3_next::Qwen3NextGatedDeltaNet, linear_attn);
    INFINICORE_NN_MODULE(Qwen35MoeSparseMoeBlock, mlp);

private:
    size_t layer_idx_;
    std::string layer_type_;
};

} // namespace infinilm::models::qwen3_5_moe
