#include "qwen3_dspark_attention.hpp"

#include <stdexcept>

namespace infinilm::models::qwen3_dspark {

void Qwen3DSparkAttention::inject_target_hidden(
    const infinilm::runtime::Tensor &positions,
    const infinilm::runtime::Tensor &hidden_states,
    const infinilm::runtime::Tensor &slot_mapping,
    infinilm::global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    if (hidden_states->ndim() != 3 || hidden_states->size(0) != 1
        || hidden_states->size(2) != hidden_size_) {
        throw std::invalid_argument(
            "Qwen3DSpark target hidden states must have shape [1, tokens, hidden]");
    }
    const size_t tokens = hidden_states->size(1);
    if (slot_mapping->ndim() != 1 || slot_mapping->size(0) != tokens) {
        throw std::invalid_argument(
            "Qwen3DSpark target slot mapping must contain one slot per token");
    }

    auto projected_input = hidden_states;
    auto [q, k, v] = qkv_proj_->forward_split(runtime, projected_input);
    (void)q;
    auto k_rows = k_norm_->forward(
        runtime, k->view({tokens, num_key_value_heads_, head_dim_}));
    auto v_rows = v->view({tokens, num_key_value_heads_, head_dim_});

    infinilm::runtime::Tensor rope_positions;
    if (positions->ndim() == 2) {
        if (positions->size(0) != 1 || positions->size(1) != tokens) {
            throw std::invalid_argument(
                "Qwen3DSpark target positions must match the packed token count");
        }
        rope_positions = positions->view({tokens});
    } else if (positions->ndim() == 1 && positions->size(0) == tokens) {
        rope_positions = positions;
    } else {
        throw std::invalid_argument(
            "Qwen3DSpark target positions must have shape [tokens] or [1, tokens]");
    }
    rotary_emb_->forward(runtime, k_rows, rope_positions, true);
    attn_->write_kv_cache(k_rows, v_rows, slot_mapping, forward_context);
}

} // namespace infinilm::models::qwen3_dspark
