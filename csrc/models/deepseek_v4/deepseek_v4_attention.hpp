#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/attention/attention.hpp"
#include "../../layers/linear/linear.hpp"
#include "deepseek_v4_compressor.hpp"
#include "deepseek_v4_attention_state.hpp"
#include "deepseek_v4_indexer.hpp"
#include "deepseek_v4_rope.hpp"
#include "deepseek_v4_utils.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
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

    infinicore::Tensor attention_prefill_(const infinicore::Tensor &positions,
                                          const infinicore::Tensor &query_states,
                                          const infinicore::Tensor &key_states,
                                          const infinicore::Tensor &hidden_states,
                                          const infinicore::Tensor &q_residual) const;

    infinicore::Tensor sliding_attention_gpu_(const infinicore::Tensor &q_rope,
                                              const infinicore::Tensor &key_states,
                                              const std::vector<int64_t> &pos,
                                              size_t query_start,
                                              const infinicore::Tensor &raw_positions) const;

    infinicore::Tensor compressed_attention_gpu_(const infinicore::Tensor &query_states,
                                                 const infinicore::Tensor &key_states,
                                                 const infinicore::Tensor &hidden_states,
                                                 const infinicore::Tensor &q_residual,
                                                 const std::vector<int64_t> &positions,
                                                 size_t query_start,
                                                 const infinicore::Tensor &raw_positions) const;

    void reset_runtime_state() const override {
        runtime_state_.reset();
    }

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

    DeepseekV4RoPE rotary_emb_;

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
    size_t sliding_window_{0};
    double rms_norm_eps_{1e-6};
    float softmax_scale_{1.0f};

    infinicore::Tensor no_index_sentinel_;
    infinicore::Tensor block_position_table_;
    mutable DeepseekV4AttentionState runtime_state_;
};

} // namespace infinilm::models::deepseek_v4
