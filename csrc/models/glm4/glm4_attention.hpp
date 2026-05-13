#pragma once

#include "../../layers/common_modules.hpp"

namespace infinilm::layers::attention {
class AttentionLayer;
}

namespace infinilm::models::glm4 {

class Glm4Attention : public infinicore::nn::Module {
public:
    Glm4Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  size_t layer_idx,
                  const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &positions, infinicore::Tensor &hidden_states);

    void set_rotary_emb(const std::shared_ptr<infinicore::nn::RoPE> &rotary_emb);

protected:
    // Operator layers
    INFINICORE_NN_MODULE(infinilm::layers::linear::QKVParallelLinear, qkv_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::RowParallelLinear, o_proj);
    std::shared_ptr<infinilm::layers::attention::AttentionLayer> attn_;
    ::infinilm::backends::AttentionBackend attention_backend_;
    std::shared_ptr<infinicore::nn::RoPE> rotary_emb_;
    std::shared_ptr<infinilm::config::ModelConfig> model_config_;

    // Model parameters
    size_t layer_idx_;
    size_t hidden_size_;
    size_t num_attention_heads_;
    size_t num_key_value_heads_;
    size_t head_dim_;
    size_t rotary_dim_;
    bool use_bias_;
    bool use_output_bias_;
    float scaling_;

    // KV Cache quantization
    INFINICORE_NN_PARAMETER(kv_cache_k_scale);
    INFINICORE_NN_PARAMETER(kv_cache_v_scale);

private:
    infinicore::Tensor forward_static_(const infinicore::Tensor &positions, infinicore::Tensor &hidden_states);
    infinicore::Tensor forward_paged_(const infinicore::Tensor &positions, infinicore::Tensor &hidden_states);
};

} // namespace infinilm::models::glm4
