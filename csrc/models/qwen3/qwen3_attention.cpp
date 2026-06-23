#include "qwen3_attention.hpp"
#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"
#include "../../utils/agent_debug.hpp"
#include "infinicore/context/context.hpp"

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

    // #region agent log
    if (layer_idx_ == 0) {
        const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
        infinilm::agent_debug::log(
            "qwen3_attention.cpp:forward_paged_",
            "qwen3_paged_attn_entry",
            "C",
            std::string("{\"tp_rank\":") + std::to_string(tp_rank) +
                ",\"seq_len\":" + std::to_string(seq_len) +
                ",\"num_attn_heads\":" + std::to_string(num_attention_heads_) +
                ",\"num_kv_heads\":" + std::to_string(num_key_value_heads_) +
                ",\"pos_rank\":" + std::to_string(position_ids->shape().size()) +
                ",\"pos_dim0\":" + std::to_string(position_ids->size(0)) + "}");
    }
    // #endregion

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
        pos_ids_for_rope = pos_narrowed->contiguous()->view({pos_shape[1]});
    } else if (pos_shape.size() == 1) {
        pos_ids_for_rope = position_ids->contiguous();
    } else {
        throw std::runtime_error("Unexpected position_ids shape");
    }

    if (!q_reshaped->is_contiguous()) {
        q_reshaped = q_reshaped->contiguous();
    }
    if (!k_reshaped->is_contiguous()) {
        k_reshaped = k_reshaped->contiguous();
    }
    if (!v_reshaped->is_contiguous()) {
        v_reshaped = v_reshaped->contiguous();
    }

    // #region agent log
    if (layer_idx_ == 0) {
        infinilm::agent_debug::log(
            "qwen3_attention.cpp:forward_paged_",
            "rope_inputs",
            "R",
            std::string("{\"q_shape\":\"") + std::to_string(q_reshaped->size(0)) + "," +
                std::to_string(q_reshaped->size(1)) + "," + std::to_string(q_reshaped->size(2)) +
                "\",\"q_stride2\":" + std::to_string(q_reshaped->stride(2)) + ",\"pos_len\":" +
                std::to_string(pos_ids_for_rope->size(0)) + "}",
            "post-fix");
    }
    // #endregion

    auto q_rope_out = infinicore::Tensor::empty(
        {seq_len, num_attention_heads_, head_dim_}, q_reshaped->dtype(), q_reshaped->device());
    auto k_rope_out = infinicore::Tensor::empty(
        {seq_len, num_key_value_heads_, head_dim_}, k_reshaped->dtype(), k_reshaped->device());
    rotary_emb_->forward(q_rope_out, q_reshaped, pos_ids_for_rope);
    infinicore::context::syncStream();
    // #region agent log
    if (layer_idx_ <= 1) {
        const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
        infinilm::agent_debug::log(
            "qwen3_attention.cpp:forward_paged_",
            "q_rope_done",
            "R",
            std::string("{\"tp_rank\":") + std::to_string(tp_rank) +
                ",\"layer\":" + std::to_string(layer_idx_) + "}",
            "post-fix");
    }
    // #endregion
    rotary_emb_->forward(k_rope_out, k_reshaped, pos_ids_for_rope);
    infinicore::context::syncStream();
    // #region agent log
    if (layer_idx_ <= 1) {
        const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
        infinilm::agent_debug::log(
            "qwen3_attention.cpp:forward_paged_",
            "k_rope_done",
            "R",
            std::string("{\"tp_rank\":") + std::to_string(tp_rank) +
                ",\"layer\":" + std::to_string(layer_idx_) + "}",
            "post-fix");
    }
    // #endregion

    q_reshaped = q_rope_out;
    k_reshaped = k_rope_out;

    auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped);
    infinicore::context::syncStream();
    // #region agent log
    if (layer_idx_ <= 1) {
        const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
        infinilm::agent_debug::log(
            "qwen3_attention.cpp:forward_paged_",
            "eager_attn_out",
            "T3",
            std::string("{\"tp_rank\":") + std::to_string(tp_rank) + ",\"layer\":" +
                std::to_string(layer_idx_) + ",\"first_bits\":" +
                std::to_string(infinilm::agent_debug::first_elem_bits(attn_output)) + "}",
            "eager-baseline");
    }
    // #endregion
    // #region agent log
    if (layer_idx_ <= 1) {
        const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
        infinilm::agent_debug::log(
            "qwen3_attention.cpp:forward_paged_",
            "paged_attn_done",
            "P",
            std::string("{\"tp_rank\":") + std::to_string(tp_rank) +
                ",\"layer\":" + std::to_string(layer_idx_) + "}",
            "post-fix");
    }
    // #endregion

    auto output = o_proj_->forward(attn_output);
    infinicore::context::syncStream();

    // #region agent log
    if (layer_idx_ <= 1) {
        const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
        infinilm::agent_debug::log(
            "qwen3_attention.cpp:forward_paged_",
            "o_proj_done",
            "O",
            std::string("{\"tp_rank\":") + std::to_string(tp_rank) +
                ",\"layer\":" + std::to_string(layer_idx_) + "}",
            "post-fix");
    }
    // #endregion

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

    auto [q, k, v] = qkv_proj_->forward_split(hidden_states_mutable);
    const size_t valid_len = piecewise.valid_seq_len > 0 ? piecewise.valid_seq_len : seq_len;
    // #region agent log
    if (layer_idx_ <= 1) {
        const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
        infinilm::agent_debug::log(
            "qwen3_attention.cpp:forward_pre_attn_piecewise",
            "pw_qkv",
            "T0",
            std::string("{\"tp_rank\":") + std::to_string(tp_rank) + ",\"layer\":" +
                std::to_string(layer_idx_) + ",\"first_bits\":" +
                std::to_string(infinilm::agent_debug::first_elem_bits(q)) + "}",
            "piecewise-upstream");
    }
    // #endregion

    auto q_heads = q->view({seq_len, num_attention_heads_, head_dim_});
    auto k_heads = k->view({seq_len, num_key_value_heads_, head_dim_});
    q_heads = q_norm_->forward(q_heads);
    k_heads = k_norm_->forward(k_heads);
    // #region agent log
    if (layer_idx_ <= 1) {
        const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
        infinilm::agent_debug::log(
            "qwen3_attention.cpp:forward_pre_attn_piecewise",
            "pw_q_norm",
            "T2",
            std::string("{\"tp_rank\":") + std::to_string(tp_rank) + ",\"layer\":" +
                std::to_string(layer_idx_) + ",\"first_bits\":" +
                std::to_string(infinilm::agent_debug::first_elem_bits(q_heads)) + "}",
            "piecewise-upstream");
        infinilm::agent_debug::log(
            "qwen3_attention.cpp:forward_pre_attn_piecewise",
            "pw_k_norm",
            "T2",
            std::string("{\"tp_rank\":") + std::to_string(tp_rank) + ",\"layer\":" +
                std::to_string(layer_idx_) + ",\"first_bits\":" +
                std::to_string(infinilm::agent_debug::first_elem_bits(k_heads)) + "}",
            "piecewise-upstream");
    }
    // #endregion

    auto q_staged = q_heads->view({1, seq_len, num_attention_heads_, head_dim_});
    auto k_staged = k_heads->view({1, seq_len, num_key_value_heads_, head_dim_});
    auto v_heads = v->view({1, seq_len, num_key_value_heads_, head_dim_});

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
        pos_ids_for_rope = position_ids->contiguous();
    } else {
        throw std::runtime_error("Unexpected position_ids shape");
    }
    if (pos_ids_for_rope->size(0) > valid_len) {
        pos_ids_for_rope = pos_ids_for_rope->narrow({{0, 0, valid_len}})->contiguous();
    }

    auto q_rope = staging.q_rope->view({seq_len, num_attention_heads_, head_dim_})->narrow({{0, 0, valid_len}});
    auto k_rope = staging.k_rope->view({seq_len, num_key_value_heads_, head_dim_})->narrow({{0, 0, valid_len}});
    if (!q_rope->is_contiguous()) {
        q_rope = q_rope->contiguous();
    }
    if (!k_rope->is_contiguous()) {
        k_rope = k_rope->contiguous();
    }
    rotary_emb_->forward(q_rope, pos_ids_for_rope, true);
    rotary_emb_->forward(k_rope, pos_ids_for_rope, true);
    // #region agent log
    if (layer_idx_ <= 1) {
        const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
        infinilm::agent_debug::log(
            "qwen3_attention.cpp:forward_pre_attn_piecewise",
            "pw_q_rope",
            "T1",
            std::string("{\"tp_rank\":") + std::to_string(tp_rank) + ",\"layer\":" +
                std::to_string(layer_idx_) + ",\"first_bits\":" +
                std::to_string(infinilm::agent_debug::first_elem_bits(q_rope)) + "}",
            "piecewise-upstream");
        infinilm::agent_debug::log(
            "qwen3_attention.cpp:forward_pre_attn_piecewise",
            "pw_k_rope",
            "T1",
            std::string("{\"tp_rank\":") + std::to_string(tp_rank) + ",\"layer\":" +
                std::to_string(layer_idx_) + ",\"first_bits\":" +
                std::to_string(infinilm::agent_debug::first_elem_bits(k_rope)) + "}",
            "piecewise-upstream");
    }
    // #endregion
}

} // namespace infinilm::models::qwen3
