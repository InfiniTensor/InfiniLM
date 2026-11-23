#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/linear.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include "../../layers/attention/backends/attention_layer.hpp"
#include "../../layers/rotary_embedding/rope.hpp"
#include "../../weights/weights.hpp"

namespace infinilm::global_state { struct ForwardContext; }
namespace infinilm::engine::distributed { struct RankInfo; }
namespace infinilm::backends { enum class AttentionBackend; }
namespace infinilm::layers::moe { struct EPConfig; }

namespace infinilm::models::deepseek_v4 {

class DeepseekV4MLA final : public infinilm::weights::ModelWeights {
public:
    DeepseekV4MLA(infinilm::runtime::Runtime &runtime,
                  std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  size_t layer_idx,
                  const engine::distributed::RankInfo &rank_info,
                  backends::AttentionBackend attention_backend,
                  const layers::moe::EPConfig &ep_config,
                  std::shared_ptr<infinilm::layers::RoPE> rope);

    infinilm::runtime::Tensor forward(const infinilm::runtime::Tensor &positions,
                                      const infinilm::runtime::Tensor &hidden_states,
                                      infinilm::global_state::ForwardContext &forward_context,
                                      const infinilm::runtime::Tensor &hidden_packed = {},
                                      const infinilm::runtime::Tensor &hidden_scale = {}) const;

private:
    size_t layer_idx_{0};
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> wq_a_;
    std::shared_ptr<infinilm::layers::RMSNorm> q_norm_;
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> wq_b_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> wkv_;
    std::shared_ptr<infinilm::layers::RMSNorm> kv_norm_;
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> wo_a_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> wo_b_;
    infinilm::weights::Weight compressor_ape_;
    infinilm::weights::Weight compressor_wkv_;
    infinilm::weights::Weight compressor_wgate_;
    infinilm::weights::Weight compressor_norm_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> indexer_wq_b_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> indexer_weights_proj_;
    infinilm::weights::Weight indexer_ape_;
    infinilm::weights::Weight indexer_wkv_;
    infinilm::weights::Weight indexer_wgate_;
    infinilm::weights::Weight indexer_norm_;
    std::shared_ptr<infinilm::layers::attention::AttentionLayer> attn_;
    std::shared_ptr<infinilm::layers::RoPE> rope_;
    infinilm::runtime::Tensor q_head_norm_weight_;
    infinilm::runtime::DataType model_dtype_;
    size_t local_heads_{0};
    size_t head_dim_{0};
    size_t o_groups_{0};
    size_t o_lora_{0};
    infinilm::backends::AttentionBackend attention_backend_;
    infinilm::weights::Weight attn_sink_;
    size_t compress_ratio_{0};
    size_t indexer_heads_{0};
    size_t indexer_head_dim_{0};
    size_t index_topk_{0};
};

} // namespace infinilm::models::deepseek_v4
