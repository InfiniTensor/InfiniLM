#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../config/model_config.hpp"
#include "../../layers/linear/fused_linear.hpp"
#include "deepseek_v2_indexer.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/tensor.hpp"

#include <memory>

namespace infinilm::models::deepseek_v2 {

class DeepseekV2MLAAttention final : public infinicore::nn::Module {
public:
    DeepseekV2MLAAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           size_t layer_idx,
                           const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               const infinicore::Tensor &hidden_states) const;

    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

private:
    infinicore::Tensor position_ids_for_rope_(const infinicore::Tensor &position_ids) const;
    infinicore::Tensor project_q_nope_to_latent_(const infinicore::Tensor &q_nope) const;
    infinicore::Tensor project_latent_to_value_(const infinicore::Tensor &attn_output,
                                                size_t batch_size,
                                                size_t seq_len) const;

    size_t layer_idx_{0};
    size_t hidden_size_{0};
    size_t num_attention_heads_{0};
    size_t qk_nope_head_dim_{0};
    size_t qk_rope_head_dim_{0};
    size_t q_head_dim_{0};
    size_t v_head_dim_{0};
    size_t q_lora_rank_{0};
    size_t kv_lora_rank_{0};
    size_t mla_head_dim_{0};
    float softmax_scale_{1.0f};
    infinilm::backends::AttentionBackend attention_backend_;
    bool use_sparse_{false};
    bool skip_topk_{false};
    infinicore::Tensor w_uk_;
    infinicore::Tensor w_uv_;

    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, q_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::MergedReplicatedLinear, fused_qkv_a_proj);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, q_a_layernorm);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, q_b_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, kv_a_proj_with_mqa);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, kv_a_layernorm);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, kv_b_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::RowParallelLinear, o_proj);
    INFINICORE_NN_MODULE(DeepseekV32Indexer, indexer);

    std::shared_ptr<infinicore::nn::RoPE> rotary_emb_;
    infinicore::Tensor rope_cos_sin_cache_;
    infinicore::nn::Parameter kv_cache_k_scale_;
    infinicore::nn::Parameter kv_cache_v_scale_;
};

} // namespace infinilm::models::deepseek_v2
