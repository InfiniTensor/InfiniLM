#pragma once

#include "../../layers/common_modules.hpp"

namespace infinilm::models::ernie4_5_moe_vl {

class Ernie4_5Attention : public infinicore::nn::Module {
public:
    Ernie4_5Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                      size_t layer_idx,
                      const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               const infinicore::Tensor &hidden_states) const;

    void process_weights_after_loading() override {
        qkv_proj_->process_weights_after_loading();
    }

    void reset_runtime_state() const override {
        qkv_proj_->reset_runtime_state();
    }

private:
    infinicore::Tensor forward_static_(const infinicore::Tensor &positions,
                                       const infinicore::Tensor &hidden_states) const;

    infinicore::Tensor forward_paged_(const infinicore::Tensor &positions,
                                      const infinicore::Tensor &hidden_states) const;

    std::shared_ptr<infinilm::layers::linear::QKVParallelLinear> qkv_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> o_proj_;
    std::shared_ptr<infinicore::nn::RoPE> rotary_emb_;
    std::shared_ptr<infinilm::layers::attention::AttentionLayer> attn_;

    ::infinilm::backends::AttentionBackend attention_backend_;
    size_t layer_idx_{0};
    size_t num_attention_heads_{0};
    size_t num_key_value_heads_{0};
    size_t hidden_size_{0};
    size_t head_dim_{0};
    double rope_theta_{500000.0};
    size_t mrope_section_h_{22};
    size_t mrope_section_w_{22};
    size_t mrope_section_t_{20};

    INFINICORE_NN_PARAMETER(kv_cache_k_scale);
    INFINICORE_NN_PARAMETER(kv_cache_v_scale);
};

} // namespace infinilm::models::ernie4_5_moe_vl
