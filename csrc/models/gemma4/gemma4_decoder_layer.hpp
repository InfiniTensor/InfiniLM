#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../config/model_config.hpp"
#include "../../engine/distributed/communication_group.hpp"
#include "../../global_state/forward_context.hpp"
#include "../../layers/linear/linear.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include "../../weights/weights.hpp"
#include "gemma4_attention.hpp"
#include "gemma4_config.hpp"
#include "gemma4_mlp.hpp"
#include "gemma4_moe.hpp"
#include "gemma4_rope.hpp"
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include <cstddef>
#include <memory>

namespace infinilm::models::gemma4 {

class Gemma4TextDecoderLayer : public infinilm::weights::ModelWeights {
public:
    Gemma4TextDecoderLayer(infinilm::runtime::Runtime &runtime,
                           std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           const Gemma4TextSpec &text_spec,
                           const Gemma4LayerSpec &layer_spec,
                           std::shared_ptr<Gemma4RoPE> rope,
                           const engine::distributed::RankInfo &rank_info,
                           backends::AttentionBackend attention_backend);

    // per_layer_input is optional ([B, S, hidden_size_per_layer_input] when PLE is enabled).
    infinilm::runtime::Tensor forward(const infinilm::runtime::Tensor &positions,
                                      const infinilm::runtime::Tensor &hidden_states,
                                      const infinilm::runtime::Tensor &per_layer_input,
                                      global_state::ForwardContext &forward_context,
                                      const Gemma4VisionSpans *vision_spans = nullptr) const;

private:
    size_t layer_idx_{0};
    size_t hidden_size_{0};
    size_t hidden_size_per_layer_input_{0};
    bool enable_moe_block_{false};

    std::shared_ptr<Gemma4TextAttention> self_attn_;
    std::shared_ptr<Gemma4TextMLP> mlp_;

    std::shared_ptr<infinilm::layers::RMSNorm> input_layernorm_;
    std::shared_ptr<infinilm::layers::RMSNorm> post_attention_layernorm_;
    std::shared_ptr<infinilm::layers::RMSNorm> pre_feedforward_layernorm_;
    std::shared_ptr<infinilm::layers::RMSNorm> post_feedforward_layernorm_;
    infinilm::weights::Weight layer_scalar_; // [1]

    // Optional PLE (E2B / E4B)
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> per_layer_input_gate_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> per_layer_projection_;
    std::shared_ptr<infinilm::layers::RMSNorm> post_per_layer_input_norm_;

    // Optional parallel MoE block (26B-A4B)
    std::shared_ptr<Gemma4TextRouter> router_;
    std::shared_ptr<Gemma4TextExperts> experts_;
    std::shared_ptr<infinilm::layers::RMSNorm> post_feedforward_layernorm_1_;
    std::shared_ptr<infinilm::layers::RMSNorm> pre_feedforward_layernorm_2_;
    std::shared_ptr<infinilm::layers::RMSNorm> post_feedforward_layernorm_2_;
};

} // namespace infinilm::models::gemma4
