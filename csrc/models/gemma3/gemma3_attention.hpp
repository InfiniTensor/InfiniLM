#pragma once

#include "../../layers/common_modules.hpp"
#include <unordered_map>

namespace infinilm::models::gemma3 {

/**
 * @brief Gemma-3 attention with QK-norm, per-layer-type RoPE theta and
 * (for sliding layers) model-level sliding-window attention.
 *
 * Gemma-3 alternates "sliding_attention" layers (local RoPE theta, attention
 * restricted to the last `sliding_window` keys) with "full_attention" layers
 * (global RoPE theta, plain causal attention) in a
 * `sliding_window_pattern`-periodic layout (default 6 -> 5:1).
 *
 * Full layers delegate to the shared AttentionLayer. Sliding layers manage the
 * static KV cache directly: the cache layout and update semantics mirror
 * `StaticAttentionImpl`, the decode path reads only the trailing window, and
 * the prefill path applies a banded causal mask built on host and cached per
 * (seq_len, total_len). Requires the STATIC attention backend.
 */
class Gemma3Attention : public infinicore::nn::Module {
public:
    Gemma3Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
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

    size_t layer_idx() const { return layer_idx_; }
    size_t num_heads() const { return num_attention_heads_; }
    size_t num_kv_heads() const { return num_key_value_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t hidden_size() const { return hidden_size_; }

private:
    // Shared prologue: project, QK-normalize, apply this layer's RoPE.
    std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor>
    project_and_rotate_(const infinicore::Tensor &positions,
                        const infinicore::Tensor &hidden_states) const;

    infinicore::Tensor forward_full_(infinicore::Tensor &q_reshaped,
                                     infinicore::Tensor &k_reshaped,
                                     infinicore::Tensor &v_reshaped) const;

    infinicore::Tensor forward_sliding_(infinicore::Tensor &q_reshaped,
                                        infinicore::Tensor &k_reshaped,
                                        infinicore::Tensor &v_reshaped) const;

    // Banded causal mask [seq, total] on `device`: entry (i, j) is 0 when key j
    // is visible to query i (j <= past+i and past+i-j < window), else -1e9.
    infinicore::Tensor sliding_mask_(size_t seq, size_t past, size_t k_start,
                                     size_t k_len, const infinicore::DataType &dtype,
                                     const infinicore::Device &device) const;

protected:
    std::shared_ptr<infinilm::layers::linear::QKVParallelLinear> qkv_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> o_proj_;
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, q_norm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, k_norm);
    std::shared_ptr<infinicore::nn::RoPE> rotary_emb_;

    // Only used by full-attention layers.
    std::shared_ptr<infinilm::layers::attention::AttentionLayer> attn_;
    ::infinilm::backends::AttentionBackend attention_backend_;

    size_t layer_idx_;
    size_t num_attention_heads_;
    size_t num_key_value_heads_;
    size_t hidden_size_;
    size_t head_dim_;
    float scale_;
    bool is_sliding_{false};
    size_t sliding_window_{0};

    // Per-(seq, total) cache of built sliding masks (device tensors).
    mutable std::unordered_map<size_t, infinicore::Tensor> mask_cache_;

    // For off-line kv cache quantization
    INFINICORE_NN_PARAMETER(kv_cache_k_scale);
    INFINICORE_NN_PARAMETER(kv_cache_v_scale);
};

} // namespace infinilm::models::gemma3
