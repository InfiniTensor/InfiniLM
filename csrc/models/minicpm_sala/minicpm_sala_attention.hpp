#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../cache/kv_cache.hpp"
#include "../../config/model_config.hpp"
#include "../../engine/distributed/distributed.hpp"

#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <string>

namespace infinilm::models::minicpm_sala {

// Dense attention fallback implementation used for Milestone 1.
// Parameter names are aligned with HF MiniCPM-SALA safetensors keys:
//   model.layers.N.self_attn.{q_proj,k_proj,v_proj,o_proj,...}
// TODO(refactor): KV cache is currently per-layer dense; refactor to use engine paged KV pool
// and block_tables/slot_mapping to match SGLang minicpm-sala pattern (see minicpm_sala_attention.cpp).
class MiniCPMSALAAttention : public infinicore::nn::Module {
public:
    MiniCPMSALAAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         const infinicore::Device &device,
                         size_t layer_idx,
                         const std::string &mixer_type,
                         engine::distributed::RankInfo rank_info = engine::distributed::RankInfo(),
                         backends::AttentionBackend attention_backend = backends::AttentionBackend::Default);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &position_ids,
                               std::shared_ptr<infinilm::cache::Cache> kv_cache,
                               std::optional<infinicore::Tensor> past_sequence_lengths,
                               std::optional<infinicore::Tensor> total_sequence_lengths,
                               std::optional<infinicore::Tensor> input_offsets,
                               std::optional<infinicore::Tensor> cu_seqlens,
                               std::optional<infinicore::Tensor> block_tables,
                               std::optional<infinicore::Tensor> slot_mapping) const;

    void set_rotary_emb(const std::shared_ptr<infinicore::nn::RoPE> &rotary_emb);
    void reset_cache();

private:
    infinicore::Tensor forward_dense_(const infinicore::Tensor &hidden_states,
                                     const infinicore::Tensor &position_ids,
                                     std::shared_ptr<infinilm::cache::Cache> kv_cache,
                                     std::optional<infinicore::Tensor> past_sequence_lengths,
                                     std::optional<infinicore::Tensor> total_sequence_lengths,
                                     std::optional<infinicore::Tensor> cu_seqlens) const;

protected:
    // Projections (HF-aligned naming)
    INFINICORE_NN_MODULE(infinicore::nn::Linear, q_proj);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, k_proj);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, v_proj);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, o_proj);

    // Optional (Lightning layers): q_norm/k_norm/o_norm + z_proj
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, q_norm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, k_norm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, o_norm);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, z_proj);

    // Optional (Sparse layers): o_gate
    INFINICORE_NN_MODULE(infinicore::nn::Linear, o_gate);

    std::shared_ptr<infinilm::config::ModelConfig> model_config_;
    std::shared_ptr<infinicore::nn::RoPE> rotary_emb_;
    engine::distributed::RankInfo rank_info_;

    size_t layer_idx_;
    // Layer index remapped into the cache instance (minicpm4-cache vs lightning-cache).
    // StaticKVCache allocates a compact [num_layers, ...] slab per cache type.
    size_t cache_layer_idx_ = 0;
    size_t hidden_size_;
    size_t num_attention_heads_;
    size_t num_key_value_heads_;
    size_t head_dim_;
    float scaling_;

    bool use_qk_norm_ = false;
    bool use_output_gate_ = false;
    bool use_output_norm_ = false;
    bool use_rope_ = false;
    bool is_sparse_layer_ = false;

    // InfLLM-v2 local-window masking plumbing for `mixer_type=="minicpm4"`.
    // When enabled: causal=false + window_size_left=sparse_window_size + window_size_right=0.
    int infllmv2_window_left_ = -1;
    int infllmv2_window_right_ = -1;
    bool use_local_window_ = false;

    backends::AttentionBackend attention_backend_;

    // Lightning layers only: per-head log-decay for Simple GLA (HF _build_slope_tensor * -1).
    infinicore::Tensor g_gamma_;
};

} // namespace infinilm::models::minicpm_sala
