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
        pos_ids_for_rope = position_ids->contiguous();
    } else {
        throw std::runtime_error("Unexpected position_ids shape");
    }

    infinilm::utils::eager_dump_barrier("eager_dump_post_pre_rope_q");
    infinilm::utils::dump_seq_heads_row(q_reshaped, layer_idx_, seq_len, "post_pre_rope_q");
    infinilm::utils::eager_dump_barrier("eager_dump_post_pre_rope_k");
    infinilm::utils::dump_seq_heads_row(k_reshaped, layer_idx_, seq_len, "post_pre_rope_k");
    rotary_emb_->forward(q_reshaped, pos_ids_for_rope, true);
    rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);
    infinilm::utils::eager_dump_barrier("eager_dump_post_rope_q");
    infinilm::utils::dump_seq_heads_row(q_reshaped, layer_idx_, seq_len, "post_rope_q");
    infinilm::utils::eager_dump_barrier("eager_dump_post_rope_k");
    infinilm::utils::dump_seq_heads_row(k_reshaped, layer_idx_, seq_len, "post_rope_k");

    auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped);
    {
        size_t valid_len = 0;
        auto &piecewise = global_state::get_forward_context().piecewise;
        if (piecewise.valid_seq_len > 0) {
            valid_len = piecewise.valid_seq_len;
        }
        infinilm::utils::eager_dump_barrier("eager_dump_post_attn");
        infinilm::utils::dump_layer_hidden(attn_output, layer_idx_, valid_len, "post_attn");
    }
    auto attn_mut = attn_output;
    auto projected = o_proj_->forward_matmul_only(attn_mut);
    {
        size_t valid_len = 0;
        auto &piecewise = global_state::get_forward_context().piecewise;
        if (piecewise.valid_seq_len > 0) {
            valid_len = piecewise.valid_seq_len;
        }
        infinilm::utils::eager_dump_barrier("eager_dump_post_o_proj_pre_ar");
        infinilm::utils::dump_layer_hidden(projected, layer_idx_, valid_len, "post_o_proj_pre_ar");
    }
    if (o_proj_->needs_allreduce()) {
        if (!projected->is_contiguous()) {
            projected = projected->contiguous();
        }
        o_proj_->allreduce_output(projected);
    }
    {
        size_t valid_len = 0;
        auto &piecewise = global_state::get_forward_context().piecewise;
        if (piecewise.valid_seq_len > 0) {
            valid_len = piecewise.valid_seq_len;
        }
        infinilm::utils::eager_dump_barrier("eager_dump_post_o_proj");
        infinilm::utils::dump_layer_hidden(projected, layer_idx_, valid_len, "post_o_proj");
    }
    return projected;
}

void Qwen3Attention::forward_pre_attn_piecewise(const infinicore::Tensor &position_ids,
                                                const infinicore::Tensor &hidden_states,
                                                global_state::PiecewiseLayerStaging &staging) const {
    (void)position_ids;
    forward_pre_attn_piecewise_fill_staging(hidden_states, staging);
}

void Qwen3Attention::forward_pre_attn_piecewise_fill_staging(const infinicore::Tensor &hidden_states,
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

    auto copy_to_staging = [&](infinicore::Tensor src, infinicore::Tensor dst) {
        if (valid_len < seq_len) {
            auto src_n = src->narrow({{1, 0, valid_len}});
            auto dst_n = dst->narrow({{1, 0, valid_len}});
            if (src_n->is_contiguous()) {
                dst_n->copy_from(src_n);
            } else {
                auto tmp = src_n->contiguous();
                dst_n->copy_from(tmp);
            }
        } else if (src->is_contiguous()) {
            dst->copy_from(src);
        } else {
            dst->copy_from(src->contiguous());
        }
    };
    copy_to_staging(q_staged, staging.q_rope);
    copy_to_staging(k_staged, staging.k_rope);
    copy_to_staging(v_heads, staging.v_rope);
    if (valid_len < seq_len) {
        auto q_tail = staging.q_rope->narrow({{1, valid_len, seq_len - valid_len}});
        auto k_tail = staging.k_rope->narrow({{1, valid_len, seq_len - valid_len}});
        auto v_tail = staging.v_rope->narrow({{1, valid_len, seq_len - valid_len}});
        set_zeros(q_tail);
        set_zeros(k_tail);
        set_zeros(v_tail);
    }
}

void Qwen3Attention::forward_pre_attn_piecewise_apply_rope(const infinicore::Tensor &position_ids,
                                                           global_state::PiecewiseLayerStaging &staging) const {
    auto &piecewise = global_state::get_forward_context().piecewise;
    const size_t seq_len = staging.q_rope->size(1);
    const size_t valid_len = piecewise.valid_seq_len > 0 ? piecewise.valid_seq_len : seq_len;

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
    if (pos_ids_for_rope->size(0) > valid_len) {
        pos_ids_for_rope = pos_ids_for_rope->narrow({{0, 0, valid_len}})->contiguous();
    }

    auto q_staging_narrow = staging.q_rope->narrow({{1, 0, valid_len}});
    auto k_staging_narrow = staging.k_rope->narrow({{1, 0, valid_len}});
    auto q_view = staging.q_rope->view({seq_len, num_attention_heads_, head_dim_})->narrow({{0, 0, valid_len}});
    auto k_view = staging.k_rope->view({seq_len, num_key_value_heads_, head_dim_})->narrow({{0, 0, valid_len}});

    // Match forward_paged_: ensure contiguous Q/K views before flash-attn RoPE.
    auto apply_rope_inplace_or_copyback = [&](infinicore::Tensor view, infinicore::Tensor staging_narrow) {
        infinicore::Tensor rope_in = view;
        if (!rope_in->is_contiguous()) {
            rope_in = rope_in->contiguous();
        }
        rotary_emb_->forward(rope_in, pos_ids_for_rope, true);
        if (!view->is_contiguous()) {
            staging_narrow->copy_from(rope_in->view(staging_narrow->shape()));
        }
    };
    apply_rope_inplace_or_copyback(q_view, q_staging_narrow);
    apply_rope_inplace_or_copyback(k_view, k_staging_narrow);

    infinilm::utils::dump_staging_heads_row(staging.q_rope, layer_idx_, valid_len, "post_rope_q");
    infinilm::utils::dump_staging_heads_row(staging.k_rope, layer_idx_, valid_len, "post_rope_k");
}

} // namespace infinilm::models::qwen3
