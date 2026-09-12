#pragma once

#include "../../layers/common_modules.hpp"
#include "minimax_attention.hpp"
#include "minimax_lightning_attention.hpp"
#include "minimax_moe.hpp"
#include <string>

namespace infinilm::models::minimax {

class MiniMaxDecoderLayer : public infinicore::nn::Module {
public:
    MiniMaxDecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                        size_t layer_idx,
                        const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               infinicore::Tensor &hidden_states) const;

    size_t layer_idx() const { return layer_idx_; }

protected:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(MiniMaxLightningAttention, linear_attn);
    INFINICORE_NN_MODULE(MiniMaxAttention, self_attn);
    INFINICORE_NN_MODULE(infinilm::layers::mlp::MLP, mlp);
    INFINICORE_NN_MODULE(MiniMaxMoeBlock, moe);

private:
    size_t layer_idx_;
    std::string layer_type_;
    size_t num_experts_{1};
    double alpha_attn_{1.0};
    double beta_attn_{1.0};
    double alpha_mlp_{1.0};
    double beta_mlp_{1.0};
};

} // namespace infinilm::models::minimax
