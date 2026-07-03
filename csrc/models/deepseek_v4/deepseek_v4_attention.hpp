#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/attention/attention.hpp"
#include "../../layers/linear/linear.hpp"
#include "deepseek_v4_compressor.hpp"
#include "deepseek_v4_indexer.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4Attention : public infinicore::nn::Module {
public:
    DeepseekV4Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         const infinicore::Device &device);
    DeepseekV4Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         size_t layer_idx,
                         const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               const infinicore::Tensor &hidden_states) const;

private:
    infinicore::Tensor forward_static_(const infinicore::Tensor &positions,
                                       const infinicore::Tensor &hidden_states) const;
    infinicore::Tensor forward_paged_(const infinicore::Tensor &positions,
                                      const infinicore::Tensor &hidden_states) const;
    infinicore::Tensor apply_grouped_output_projection_(const infinicore::Tensor &attn_output) const;
    infinicore::Tensor dense_attention_(const infinicore::Tensor &query_states,
                                        const infinicore::Tensor &key_states,
                                        const infinicore::Tensor &value_states) const;
    infinicore::Tensor dense_attention_reference_(const infinicore::Tensor &positions,
                                                  const infinicore::Tensor &query_states,
                                                  const infinicore::Tensor &key_states,
                                                  const infinicore::Tensor &hidden_states,
                                                  const infinicore::Tensor &q_residual) const;

    INFINICORE_NN_PARAMETER(attn_sink);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, q_norm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, kv_norm);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, wq_a);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, wq_b);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, wkv);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, wo_a);
    INFINICORE_NN_MODULE(infinilm::layers::linear::RowParallelLinear, wo_b);
    INFINICORE_NN_MODULE(DeepseekV4Compressor, compressor);
    INFINICORE_NN_MODULE(DeepseekV4Indexer, indexer);

    std::shared_ptr<infinicore::nn::RoPE> rotary_emb_;
    std::shared_ptr<infinilm::layers::attention::AttentionLayer> attn_;
    INFINICORE_NN_PARAMETER(kv_cache_k_scale);
    INFINICORE_NN_PARAMETER(kv_cache_v_scale);

    infinilm::backends::AttentionBackend attention_backend_;
    size_t layer_idx_{0};
    size_t hidden_size_{0};
    size_t global_num_attention_heads_{0};
    size_t num_attention_heads_{0};
    size_t num_key_value_heads_{1};
    size_t head_dim_{0};
    size_t q_lora_rank_{0};
    size_t o_lora_rank_{0};
    size_t global_o_groups_{0};
    size_t o_groups_{0};
    size_t o_a_input_size_{0};
    size_t o_a_output_size_{0};
    size_t qk_rope_head_dim_{0};
    size_t sliding_window_{0};
    size_t compress_ratio_{0};
    double rms_norm_eps_{1e-6};
    double rope_theta_{10000.0};
    double compress_rope_theta_{10000.0};
    double rope_factor_{1.0};
    double rope_beta_fast_{32.0};
    double rope_beta_slow_{1.0};
    int64_t rope_original_max_position_embeddings_{0};
    float softmax_scale_{1.0f};
};

} // namespace infinilm::models::deepseek_v4
