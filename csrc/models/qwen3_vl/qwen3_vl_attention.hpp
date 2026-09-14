#pragma once

#include "../../layers/common_modules.hpp"

namespace infinilm::models::qwen3_vl {

class Qwen3VLAttention : public infinicore::nn::Module {
public:
    Qwen3VLAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     size_t layer_idx, const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               const infinicore::Tensor &hidden_states) const;

    void process_weights_after_loading() override {
        qkv_proj_->process_weights_after_loading();
    }

    void reset_runtime_state() const override {
        qkv_proj_->reset_runtime_state();
    }

private:
    infinicore::Tensor
    forward_static_(const infinicore::Tensor &positions,
                    const infinicore::Tensor &hidden_states) const;
    infinicore::Tensor
    forward_paged_(const infinicore::Tensor &positions,
                   const infinicore::Tensor &hidden_states) const;

    std::shared_ptr<infinilm::layers::linear::QKVParallelLinear> qkv_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> o_proj_;
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, q_norm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, k_norm);
    std::shared_ptr<infinicore::nn::RoPE> mrope_;
    std::shared_ptr<infinilm::layers::attention::AttentionLayer> attn_;

    ::infinilm::backends::AttentionBackend attention_backend_;
    size_t layer_idx_;
    size_t num_attention_heads_;
    size_t num_key_value_heads_;
    size_t hidden_size_;
    size_t head_dim_;

    INFINICORE_NN_PARAMETER(kv_cache_k_scale);
    INFINICORE_NN_PARAMETER(kv_cache_v_scale);
};

} // namespace infinilm::models::qwen3_vl
