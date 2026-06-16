#include "qwen3_attention.hpp"
#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"
#include "../../utils/layer_hidden_dump.hpp"

namespace infinilm::models::qwen3 {

Qwen3Attention::Qwen3Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                               size_t layer_idx,
                               const infinicore::Device &device)
    : Attention(model_config, layer_idx, device) {
    const auto &dtype{model_config->get_dtype()};
    double rms_norm_eps = model_config->get<double>("rms_norm_eps");
    INFINICORE_NN_MODULE_INIT(q_norm, head_dim_, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(k_norm, head_dim_, rms_norm_eps, dtype, device);
}

infinicore::Tensor Qwen3Attention::forward(const infinicore::Tensor &positions,
                                           const infinicore::Tensor &hidden_states) const {
    if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        return forward_static_(positions, hidden_states);
    }
    return forward_paged_(positions, hidden_states);
}

infinicore::Tensor Qwen3Attention::forward_static_(const infinicore::Tensor &position_ids,
                                                   const infinicore::Tensor &hidden_states) const {

    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    auto [q, k, v] = qkv_proj_->forward_split(hidden_states_mutable);

    q = q_norm_->forward(q->view({batch_size * seq_len, num_attention_heads_, head_dim_}));
    k = k_norm_->forward(k->view({batch_size * seq_len, num_key_value_heads_, head_dim_}));

    auto q_reshaped = q->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

    auto pos_shape = position_ids->shape();
    infinicore::Tensor pos_ids_for_rope = position_ids;
    if (pos_shape.size() == 2) {
        auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
        pos_ids_for_rope = pos_narrowed->contiguous()->view({pos_shape[1]});
    } else if (pos_shape.size() == 1) {
        pos_ids_for_rope = position_ids->contiguous();
    } else {
        throw std::runtime_error("infinilm::models::qwen3::Qwen3Attention: Unexpected position_ids shape");
    }

    auto q_rope = infinicore::Tensor::empty({batch_size, num_attention_heads_, seq_len, head_dim_}, q_reshaped->dtype(), q_reshaped->device())->permute({0, 2, 1, 3});
    rotary_emb_->forward(q_rope, q_reshaped, pos_ids_for_rope);
    rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);

    auto attn_output = attn_->forward(q_rope, k_reshaped, v_reshaped);
    return o_proj_->forward(attn_output);
}

infinicore::Tensor Qwen3Attention::forward_paged_(const infinicore::Tensor &position_ids,
                                                  const infinicore::Tensor &hidden_states) const {

    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    ASSERT_EQ(batch_size, 1);

    auto [q, k, v] = qkv_proj_->forward_split(hidden_states_mutable);

    auto q_reshaped = q->view({seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->view({seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({seq_len, num_key_value_heads_, head_dim_});
    q_reshaped = q_norm_->forward(q_reshaped);
    k_reshaped = k_norm_->forward(k_reshaped);

    auto pos_shape = position_ids->shape();
    infinicore::Tensor pos_ids_for_rope = position_ids;
    if (pos_shape.size() == 2) {
        auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
        pos_ids_for_rope = pos_narrowed->view({pos_shape[1]});
    } else if (pos_shape.size() == 1) {
        pos_ids_for_rope = position_ids;
    } else {
        throw std::runtime_error("Unexpected position_ids shape");
    }

    rotary_emb_->forward(q_reshaped, pos_ids_for_rope, true);
    rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);

    auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped);
    {
        size_t valid_len = 0;
        auto &piecewise = global_state::get_forward_context().piecewise;
        if (piecewise.valid_seq_len > 0) {
            valid_len = piecewise.valid_seq_len;
        }
        infinilm::utils::dump_layer_hidden(attn_output, layer_idx_, valid_len, "post_attn");
    }
    auto output = o_proj_->forward(attn_output);
    {
        size_t valid_len = 0;
        auto &piecewise = global_state::get_forward_context().piecewise;
        if (piecewise.valid_seq_len > 0) {
            valid_len = piecewise.valid_seq_len;
        }
        infinilm::utils::dump_layer_hidden(output, layer_idx_, valid_len, "post_o_proj");
    }
    return output;
}

void Qwen3Attention::forward_pre_attn_piecewise(const infinicore::Tensor &position_ids,
                                                const infinicore::Tensor &hidden_states,
                                                global_state::PiecewiseLayerStaging &staging) const {
    auto &piecewise = global_state::get_forward_context().piecewise;
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];
    ASSERT_EQ(batch_size, 1);

    const size_t valid_len = piecewise.valid_seq_len > 0 ? piecewise.valid_seq_len : seq_len;

    infinicore::Tensor hidden_for_qkv = hidden_states_mutable;
    size_t qkv_seq_len = seq_len;
    if (valid_len < seq_len) {
        hidden_for_qkv = hidden_states_mutable->narrow({{1, 0, valid_len}});
        qkv_seq_len = valid_len;
    }
    auto [q, k, v] = qkv_proj_->forward_split(hidden_for_qkv);

    auto q_heads = q->view({qkv_seq_len, num_attention_heads_, head_dim_});
    auto k_heads = k->view({qkv_seq_len, num_key_value_heads_, head_dim_});
    q_heads = q_norm_->forward(q_heads);
    k_heads = k_norm_->forward(k_heads);

    auto q_staged = q_heads->view({1, qkv_seq_len, num_attention_heads_, head_dim_});
    auto k_staged = k_heads->view({1, qkv_seq_len, num_key_value_heads_, head_dim_});
    auto v_heads = v->view({1, qkv_seq_len, num_key_value_heads_, head_dim_});

    if (valid_len < seq_len) {
        staging.q_rope->narrow({{1, 0, valid_len}})->copy_from(q_staged->narrow({{1, 0, valid_len}}));
        staging.k_rope->narrow({{1, 0, valid_len}})->copy_from(k_staged->narrow({{1, 0, valid_len}}));
        staging.v_rope->narrow({{1, 0, valid_len}})->copy_from(v_heads->narrow({{1, 0, valid_len}}));
        auto q_tail = staging.q_rope->narrow({{1, valid_len, seq_len - valid_len}});
        auto k_tail = staging.k_rope->narrow({{1, valid_len, seq_len - valid_len}});
        auto v_tail = staging.v_rope->narrow({{1, valid_len, seq_len - valid_len}});
        set_zeros(q_tail);
        set_zeros(k_tail);
        set_zeros(v_tail);
    } else {
        staging.q_rope->copy_from(q_staged);
        staging.k_rope->copy_from(k_staged);
        staging.v_rope->copy_from(v_heads);
    }

    auto pos_shape = position_ids->shape();
    infinicore::Tensor pos_ids_for_rope = position_ids;
    if (pos_shape.size() == 2) {
        auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
        pos_ids_for_rope = pos_narrowed->view({pos_shape[1]});
    } else if (pos_shape.size() == 1) {
        pos_ids_for_rope = position_ids;
    } else {
        throw std::runtime_error("Unexpected position_ids shape");
    }
    if (pos_ids_for_rope->size(0) > valid_len) {
        pos_ids_for_rope = pos_ids_for_rope->narrow({{0, 0, valid_len}});
    }

    auto q_rope = staging.q_rope->view({seq_len, num_attention_heads_, head_dim_})->narrow({{0, 0, valid_len}});
    auto k_rope = staging.k_rope->view({seq_len, num_key_value_heads_, head_dim_})->narrow({{0, 0, valid_len}});
    rotary_emb_->forward(q_rope, pos_ids_for_rope, true);
    rotary_emb_->forward(k_rope, pos_ids_for_rope, true);
}

} // namespace infinilm::models::qwen3
