#include "attention.hpp"
#include "../../global_state/ar_profile.hpp"
#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"
#include "../../utils/agent_debug.hpp"
#include "../rotary_embedding/rotary_embedding.hpp"
#include "infinicore/context/context.hpp"

namespace infinilm::layers::attention {

namespace {

bool piecewise_upstream_active() {
    const auto &pw = infinilm::global_state::get_forward_context().piecewise;
    return pw.valid_seq_len > 0 || pw.phase != infinilm::global_state::PiecewiseCapturePhase::None;
}

void log_piecewise_stage(size_t layer_idx, const char *stage, const char *hypothesis_id,
                         const infinicore::Tensor &t, const char *run_id = "piecewise-upstream") {
    if (layer_idx > 1 || !piecewise_upstream_active()) {
        return;
    }
    const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
    infinilm::agent_debug::log(
        "attention.cpp:piecewise",
        stage,
        hypothesis_id,
        std::string("{\"tp_rank\":") + std::to_string(tp_rank) + ",\"layer\":" + std::to_string(layer_idx) +
            ",\"first_bits\":" + std::to_string(infinilm::agent_debug::first_elem_bits(t)) + "}",
        run_id);
}

} // namespace

Attention::Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     size_t layer_idx,
                     const infinicore::Device &device) {
    layer_idx_ = layer_idx;
    hidden_size_ = model_config->get<size_t>("hidden_size");
    head_dim_ = model_config->get<size_t>("head_dim");

    const auto &dtype{model_config->get_dtype()};
    size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
    size_t total_num_kv_heads = model_config->get<size_t>("num_key_value_heads");
    bool use_bias = model_config->get_or<bool>("attention_bias", true);
    bool use_output_bias = model_config->get_or<bool>("attention_output_bias", false);
    double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    attention_backend_ = infinilm::global_state::get_infinilm_config().attention_backend;
    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
    int tp_size = infinilm::global_state::get_tensor_model_parallel_world_size();

    num_attention_heads_ = total_num_heads / tp_size;
    num_key_value_heads_ = total_num_kv_heads < tp_size ? 1 : total_num_kv_heads / tp_size;

    auto quantization_method = model_config->get_quantization_method();
    auto register_fn = [this](const std::string &n, infinicore::nn::Parameter p) { this->register_parameter(n, std::move(p)); };
    qkv_proj_ = std::make_shared<layers::linear::QKVParallelLinear>(
        hidden_size_, head_dim_, total_num_heads, total_num_kv_heads,
        "q_proj", "k_proj", "v_proj", register_fn,
        quantization_method, use_bias, dtype, device, rank_info);
    o_proj_ = this->register_module<layers::linear::RowParallelLinear>(
        "o_proj", total_num_heads * head_dim_, hidden_size_, quantization_method,
        use_output_bias, dtype, device, tp_rank, tp_size, rank_info.comm);

    rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(model_config, device);

    float scaling = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    attn_ = std::make_shared<AttentionLayer>(num_attention_heads_, head_dim_, scaling, num_key_value_heads_, layer_idx_,
                                             kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_);

    init_kv_cache_quant_params(register_fn, device, kv_cache_k_scale_, kv_cache_v_scale_);
}

infinicore::Tensor Attention::forward(const infinicore::Tensor &positions,
                                      const infinicore::Tensor &hidden_states) const {
    if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        return forward_static_(positions, hidden_states);
    }
    return forward_paged_(positions, hidden_states);
}

infinicore::Tensor Attention::forward_static_(const infinicore::Tensor &position_ids,
                                              const infinicore::Tensor &hidden_states) const {
    // hidden_states shape: [batch, seq_len, hidden_size]
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    // 1. Project Q, K, V
    auto [q, k, v] = qkv_proj_->forward_split(hidden_states_mutable);

    // 2. Reshape for multi-head attention
    auto q_reshaped = q->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

    // 3. Prepare position_ids for RoPE
    auto pos_shape = position_ids->shape();
    infinicore::Tensor pos_ids_for_rope = position_ids;
    if (pos_shape.size() == 2) {
        auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
        pos_ids_for_rope = pos_narrowed->contiguous()->view({pos_shape[1]});
    } else if (pos_shape.size() == 1) {
        pos_ids_for_rope = position_ids->contiguous();
    } else {
        throw std::runtime_error("infinilm::layers::attention::Attention: Unexpected position_ids shape");
    }

    // 4. Apply RoPE to QK
    auto q_rope = infinicore::Tensor::empty({batch_size, num_attention_heads_, seq_len, head_dim_}, q_reshaped->dtype(), q_reshaped->device())->permute({0, 2, 1, 3});
    rotary_emb_->forward(q_rope, q_reshaped, pos_ids_for_rope);
    rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);

    // 5. Attn Backend calculate
    auto attn_output = attn_->forward(q_rope, k_reshaped, v_reshaped);

    // 7. Project output
    auto output = o_proj_->forward(attn_output);
    return output;
}

void Attention::forward_pre_attn_piecewise(const infinicore::Tensor &position_ids,
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

    auto q_heads = q->view({1, seq_len, num_attention_heads_, head_dim_});
    auto k_heads = k->view({1, seq_len, num_key_value_heads_, head_dim_});
    auto v_heads = v->view({1, seq_len, num_key_value_heads_, head_dim_});

    if (valid_len < seq_len) {
        staging.q_rope->narrow({{1, 0, valid_len}})->copy_from(q_heads->narrow({{1, 0, valid_len}}));
        staging.k_rope->narrow({{1, 0, valid_len}})->copy_from(k_heads->narrow({{1, 0, valid_len}}));
        staging.v_rope->narrow({{1, 0, valid_len}})->copy_from(v_heads->narrow({{1, 0, valid_len}}));
        auto q_tail = staging.q_rope->narrow({{1, valid_len, seq_len - valid_len}});
        auto k_tail = staging.k_rope->narrow({{1, valid_len, seq_len - valid_len}});
        auto v_tail = staging.v_rope->narrow({{1, valid_len, seq_len - valid_len}});
        set_zeros(q_tail);
        set_zeros(k_tail);
        set_zeros(v_tail);
    } else {
        staging.q_rope->copy_from(q_heads);
        staging.k_rope->copy_from(k_heads);
        staging.v_rope->copy_from(v_heads);
    }

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
    log_piecewise_stage(layer_idx_, "pw_q_rope", "T1", q_rope);
    log_piecewise_stage(layer_idx_, "pw_k_rope", "T1", k_rope);
    // #endregion
}

void Attention::forward_eager_attn_piecewise(const infinicore::Tensor &,
                                             global_state::PiecewiseLayerStaging &staging) const {
    auto &piecewise = global_state::get_forward_context().piecewise;
    const size_t seq_len = staging.q_rope->size(1);
    const size_t valid_len = piecewise.valid_seq_len > 0 ? piecewise.valid_seq_len : seq_len;

    auto q = staging.q_rope->view({seq_len, num_attention_heads_, head_dim_})->narrow({{0, 0, valid_len}});
    auto k = staging.k_rope->view({seq_len, num_key_value_heads_, head_dim_})->narrow({{0, 0, valid_len}});
    auto v = staging.v_rope->view({seq_len, num_key_value_heads_, head_dim_})->narrow({{0, 0, valid_len}});
    if (!q->is_contiguous()) {
        q = q->contiguous();
    }
    if (!k->is_contiguous()) {
        k = k->contiguous();
    }
    if (!v->is_contiguous()) {
        v = v->contiguous();
    }

    auto attn_output = attn_->forward(q, k, v);
    if (valid_len < seq_len) {
        staging.attn_output->narrow({{1, 0, valid_len}})->copy_from(attn_output->narrow({{1, 0, valid_len}}));
        auto attn_tail = staging.attn_output->narrow({{1, valid_len, seq_len - valid_len}});
        set_zeros(attn_tail);
    } else {
        staging.attn_output->copy_from(attn_output);
    }
    // #region agent log
    log_piecewise_stage(
        layer_idx_,
        "pw_attn_out",
        "T3",
        staging.attn_output->narrow({{1, 0, valid_len}}));
    // #endregion
}

void Attention::forward_post_attn_piecewise_into(infinicore::Tensor &hidden_states,
                                               global_state::PiecewiseLayerStaging &staging) const {
    forward_post_attn_piecewise_graph_into(hidden_states, staging);
    forward_post_attn_piecewise_allreduce_into(hidden_states, staging);
}

void Attention::forward_post_attn_piecewise_graph_into(infinicore::Tensor &hidden_states,
                                                       global_state::PiecewiseLayerStaging &staging) const {
    auto &piecewise = global_state::get_forward_context().piecewise;
    const size_t seq_len = staging.attn_output->size(1);
    const size_t valid_len = piecewise.valid_seq_len > 0 ? piecewise.valid_seq_len : seq_len;

    auto hidden_narrow = hidden_states->narrow({{1, 0, valid_len}});
    auto attn_for_proj = staging.attn_output->narrow({{1, 0, valid_len}});
    auto projected = o_proj_->forward_matmul_only(attn_for_proj);
    hidden_narrow->copy_from(projected);
    // #region agent log
    log_piecewise_stage(layer_idx_, "pw_o_proj_pre_ar", "T4", hidden_narrow);
    // #endregion
    if (valid_len < seq_len) {
        auto hidden_tail = hidden_states->narrow({{1, valid_len, seq_len - valid_len}});
        set_zeros(hidden_tail);
    }
}

void Attention::forward_post_attn_piecewise_cg_into(infinicore::Tensor &hidden_states,
                                                  global_state::PiecewiseLayerStaging &staging) const {
    auto &piecewise = global_state::get_forward_context().piecewise;
    const size_t seq_len = staging.attn_output->size(1);
    const size_t valid_len = piecewise.valid_seq_len > 0 ? piecewise.valid_seq_len : seq_len;

    auto hidden_narrow = hidden_states->narrow({{1, 0, valid_len}});
    auto attn_for_proj = staging.attn_output->narrow({{1, 0, valid_len}});
    auto projected = o_proj_->forward(attn_for_proj);
    hidden_narrow->copy_from(projected);
    // #region agent log
    log_piecewise_stage(layer_idx_, "pw_o_proj_post_ar", "T4", hidden_narrow);
    // #endregion
    if (valid_len < seq_len) {
        auto hidden_tail = hidden_states->narrow({{1, valid_len, seq_len - valid_len}});
        set_zeros(hidden_tail);
    }
}

void Attention::forward_post_attn_piecewise_allreduce_into(infinicore::Tensor &hidden_states,
                                                           global_state::PiecewiseLayerStaging &staging) const {
    if (!o_proj_->needs_allreduce()) {
        return;
    }
    auto &piecewise = global_state::get_forward_context().piecewise;
    const size_t seq_len = staging.attn_output->size(1);
    const size_t valid_len = piecewise.valid_seq_len > 0 ? piecewise.valid_seq_len : seq_len;
    global_state::ar_profile::allreduce_hidden_valid_contiguous(
        hidden_states,
        valid_len,
        piecewise.ar_staging,
        [&](infinicore::Tensor &t) { o_proj_->allreduce_output(t); });
}

infinicore::Tensor Attention::forward_paged_(const infinicore::Tensor &position_ids,
                                             const infinicore::Tensor &hidden_states) const {
    // hidden_states shape: [batch, seq_len, hidden_size]
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    // Only support batchsize==1, all requests should be flattened along seqlen dimension
    ASSERT_EQ(batch_size, 1);

    // 1. Project Q, K, V
    auto [q, k, v] = qkv_proj_->forward_split(hidden_states_mutable);

    // 2. Reshape for multi-head attention
    auto q_reshaped = q->view({seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->view({seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({seq_len, num_key_value_heads_, head_dim_});

    // 3. Prepare position_ids for RoPE
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

    // 4. Apply RoPE to QK (HPCC flash-attn / paged_caching require contiguous Q/K/V)
    if (!q_reshaped->is_contiguous()) {
        q_reshaped = q_reshaped->contiguous();
    }
    if (!k_reshaped->is_contiguous()) {
        k_reshaped = k_reshaped->contiguous();
    }
    if (!v_reshaped->is_contiguous()) {
        v_reshaped = v_reshaped->contiguous();
    }
    rotary_emb_->forward(q_reshaped, pos_ids_for_rope, true);
    rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);

    // 5. Attn Backend calculate
    auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped);

    // 6. Project output
    auto output = o_proj_->forward(attn_output);
    return output;
}

void init_kv_cache_quant_params(std::function<void(const std::string &, infinicore::nn::Parameter)> register_fn,
                                const infinicore::Device &device,
                                infinicore::nn::Parameter &kv_cache_k_scale,
                                infinicore::nn::Parameter &kv_cache_v_scale) {
    auto kv_quant_scheme = infinilm::global_state::get_infinilm_config().model_config->get_kv_quant_scheme();
    switch (kv_quant_scheme) {
    case infinilm::quantization::KVQuantAlgo::NONE:
        break;
    case infinilm::quantization::KVQuantAlgo::INT8:
        kv_cache_k_scale = infinicore::nn::Parameter({1}, infinicore::DataType::F32, device, 0, 0, 1);
        register_fn("kv_cache_k_scale", kv_cache_k_scale);
        kv_cache_v_scale = infinicore::nn::Parameter({1}, infinicore::DataType::F32, device, 0, 0, 1);
        register_fn("kv_cache_v_scale", kv_cache_v_scale);
        break;
    default:
        throw std::runtime_error("unsupported kv_quant_scheme");
    }
}

} // namespace infinilm::layers::attention
