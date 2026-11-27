#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../config/model_config.hpp"
#include "../../engine/distributed/communication_group.hpp"
#include "../../global_state/forward_context.hpp"
#include "../../layers/linear/linear.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include "../../weights/weights.hpp"
#include "gemma4_config.hpp"
#include "gemma4_rope.hpp"
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace infinilm::models::gemma4 {

using Gemma4VisionSpans = std::vector<std::pair<size_t, size_t>>;

class Gemma4TextAttention : public infinilm::weights::ModelWeights {
public:
    Gemma4TextAttention(infinilm::runtime::Runtime &runtime,
                        std::shared_ptr<infinilm::config::ModelConfig> model_config,
                        const Gemma4TextSpec &text_spec,
                        const Gemma4LayerSpec &layer_spec,
                        std::shared_ptr<Gemma4RoPE> rope,
                        const engine::distributed::RankInfo &rank_info,
                        backends::AttentionBackend attention_backend);

    infinilm::runtime::Tensor forward(const infinilm::runtime::Tensor &positions,
                                      const infinilm::runtime::Tensor &hidden_states,
                                      global_state::ForwardContext &forward_context,
                                      const Gemma4VisionSpans *vision_spans = nullptr) const;

    size_t layer_idx() const { return layer_idx_; }
    bool is_kv_shared() const { return is_kv_shared_; }

private:
    infinilm::runtime::Tensor forward_static_(const infinilm::runtime::Tensor &positions,
                                              const infinilm::runtime::Tensor &hidden_states,
                                              global_state::ForwardContext &forward_context,
                                              const Gemma4VisionSpans *vision_spans) const;

    infinilm::runtime::Tensor forward_paged_(const infinilm::runtime::Tensor &positions,
                                             const infinilm::runtime::Tensor &hidden_states,
                                             global_state::ForwardContext &forward_context) const;

    infinilm::runtime::Device device_;
    infinilm::runtime::Runtime *construction_owner_{nullptr};
    backends::AttentionBackend attention_backend_;

    size_t layer_idx_{0};
    size_t kv_source_layer_idx_{0};
    bool is_sliding_{false};
    size_t sliding_window_{512};
    bool is_kv_shared_{false};
    bool use_k_eq_v_{false};
    bool use_bidirectional_vision_attention_{false};

    size_t hidden_size_{0};
    size_t head_dim_{0};
    size_t num_attention_heads_{0};
    size_t num_key_value_heads_{0};
    double rms_norm_eps_{1e-6};

    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> q_proj_;
    std::shared_ptr<infinilm::layers::linear::BaseLinear> k_proj_;
    std::shared_ptr<infinilm::layers::linear::BaseLinear> v_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> o_proj_;

    std::shared_ptr<infinilm::layers::RMSNorm> q_norm_;
    std::shared_ptr<infinilm::layers::RMSNorm> k_norm_;
    infinilm::runtime::Tensor v_norm_ones_; // Non-checkpoint constant ones for unscaled v_norm

    std::shared_ptr<Gemma4RoPE> rope_;
};

} // namespace infinilm::models::gemma4
