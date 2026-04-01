#pragma once

#include "../../layers/mlp/mlp.hpp"
#include "minicpm_sala_attention.hpp"
#include <tuple>
#include <variant>

namespace infinilm::models::minicpm_sala {
using MiniCPMMLP = infinilm::layers::MLP;
using MiniCPMSALAAttention = std::variant<std::shared_ptr<InfLLMv2Attention>, std::shared_ptr<LightningAttention>>;

class MiniCPMSALADecoderLayer : public infinicore::nn::Module {
public:
    MiniCPMSALADecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                            size_t layer_idx,
                            const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(infinicore::Tensor &hidden_states, infinicore::Tensor &residual);

    infinicore::Tensor forward(infinicore::Tensor &hidden_states);

    void set_rotary_emb(const std::shared_ptr<infinicore::nn::RoPE> &rotary_emb);

protected:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(MiniCPMSALAAttention, self_attn);
    INFINICORE_NN_MODULE(MiniCPMMLP, mlp);

    size_t layer_idx_;
};

} // namespace infinilm::models::minicpm_sala
