#pragma once

#include "../qwen3/qwen3_attention.hpp"

namespace infinilm::models::qwen3_dspark {

class Qwen3DSparkAttention final : public qwen3::Qwen3Attention {
public:
    using qwen3::Qwen3Attention::Qwen3Attention;

    void inject_target_hidden(
        const infinilm::runtime::Tensor &positions,
        const infinilm::runtime::Tensor &hidden_states,
        const infinilm::runtime::Tensor &slot_mapping,
        infinilm::global_state::ForwardContext &forward_context) const;
};

} // namespace infinilm::models::qwen3_dspark
