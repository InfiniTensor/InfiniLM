#include "minicpm5_moe_attention.hpp"

#include "../../global_state/ar_profile.hpp"
#include "../../global_state/global_state.hpp"
#include "../../layers/rotary_embedding/rotary_embedding.hpp"
#include "../../utils.hpp"
#include "../../layers/attention/attention.hpp"
#include "minicpm5_moe_router_cpu_detail.hpp"

#include "infinicore/ops/mul.hpp"
#include "infinicore/ops/sigmoid.hpp"

#include <stdexcept>

namespace infinilm::models::minicpm5_moe {

namespace {

void ensure_same_shape_buf(infinicore::Tensor &buf, const infinicore::Tensor &like) {
    if (!buf || buf->shape() != like->shape() || buf->dtype() != like->dtype()
        || buf->device() != like->device()) {
        buf = infinicore::Tensor::empty(like->shape(), like->dtype(), like->device());
    }
}

void store_gate_score(infinicore::Tensor &gate_buf, const infinicore::Tensor &gate) {
    auto g = gate->contiguous();
    ensure_same_shape_buf(gate_buf, g);
    gate_buf->copy_from(g);
}

infinicore::Tensor apply_gate_sigmoid_mul(const infinicore::Tensor &attn_output,
                                          const infinicore::Tensor &gate_score,
                                          infinicore::Tensor &gate_sigmoid_buf) {
    // Device sigmoid×mul (MetaX via InfiniCore). CPU path kept as last-resort fallback.
    auto gate_view = gate_score;
    if (attn_output->shape().size() == 3 && gate_score->shape().size() == 2 && attn_output->shape()[0] == 1) {
        gate_view = gate_score->view({1, gate_score->shape()[0], gate_score->shape()[1]});
    }
    try {
        auto gate_in = gate_view->contiguous();
        ensure_same_shape_buf(gate_sigmoid_buf, gate_in);
        infinicore::op::sigmoid_(gate_sigmoid_buf, gate_in);
        return infinicore::op::mul(attn_output->contiguous(), gate_sigmoid_buf);
    } catch (const std::exception &) {
        auto a_cpu = attn_output->to(infinicore::Device::cpu())->contiguous();
        auto g_cpu = gate_view->to(infinicore::Device::cpu())->contiguous();
        const size_t n = a_cpu->numel();
        auto out_cpu = infinicore::Tensor::empty(a_cpu->shape(), a_cpu->dtype(), infinicore::Device::cpu());
        for (size_t i = 0; i < n; ++i) {
            float a = router_cpu_detail::scalar_to_f32(a_cpu, i);
            float g = router_cpu_detail::sigmoid_f32(router_cpu_detail::scalar_to_f32(g_cpu, i));
            router_cpu_detail::write_f32_as_element(out_cpu, i, a * g);
        }
        return out_cpu->to(attn_output->device());
    }
}

} // namespace

MiniCPM5MoeAttention::MiniCPM5MoeAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                           size_t layer_idx,
                                           const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    const auto &dtype{model_config->get_dtype()};
    hidden_size_ = model_config->get<size_t>("hidden_size");
    head_dim_ = model_config->get<size_t>("head_dim");
    size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
    size_t total_num_kv_heads = model_config->get<size_t>("num_key_value_heads");
    use_gated_attention_ = model_config->get_or<bool>("use_gated_attention", false);

    const engine::distributed::RankInfo &rank_info =
        infinilm::global_state::get_tensor_model_parallel_rank_info();
    const int tp_rank = rank_info.tp_rank;
    const int tp_size = rank_info.tp_size;

    if ((total_num_kv_heads < static_cast<size_t>(tp_size)) ||
        (0 != (total_num_kv_heads % static_cast<size_t>(tp_size)))) {
        throw std::runtime_error("MiniCPM5MoeAttention: num_key_value_heads must be divisible by tp_size");
    }

    num_attention_heads_ = total_num_heads / static_cast<size_t>(tp_size);
    num_key_value_heads_ = total_num_kv_heads / static_cast<size_t>(tp_size);

    const bool use_bias = model_config->get_or<bool>("attention_bias", false);
    const bool use_output_bias = model_config->get_or<bool>("attention_output_bias", false);
    auto quantization_method = model_config->get_quantization_method();

    auto register_fn = [this](const std::string &n, infinicore::nn::Parameter p) {
        this->register_parameter(n, std::move(p));
    };

    if (use_gated_attention_) {
        qkv_proj_ = std::make_shared<infinilm::layers::linear::QKVParallelLinear>(
            hidden_size_,
            /*q_dim=*/2 * head_dim_, head_dim_, head_dim_,
            total_num_heads, total_num_kv_heads, total_num_kv_heads,
            "q_proj", "k_proj", "v_proj", register_fn, quantization_method,
            use_bias, use_bias, use_bias, dtype, device, rank_info);
    } else {
        qkv_proj_ = std::make_shared<infinilm::layers::linear::QKVParallelLinear>(
            hidden_size_, head_dim_, total_num_heads, total_num_kv_heads,
            "q_proj", "k_proj", "v_proj", register_fn, quantization_method,
            use_bias, dtype, device, rank_info);
    }

    o_proj_ = this->register_module<infinilm::layers::linear::RowParallelLinear>(
        "o_proj", total_num_heads * head_dim_, hidden_size_, quantization_method,
        use_output_bias, dtype, device, tp_rank, tp_size, rank_info.comm);

    attention_backend_ = infinilm::global_state::get_infinilm_config().attention_backend;
    rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(model_config, device);
    float scaling = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(
        num_attention_heads_, head_dim_, scaling, num_key_value_heads_, layer_idx_,
        kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_);

    infinilm::layers::attention::init_kv_cache_quant_params(
        register_fn, device, kv_cache_k_scale_, kv_cache_v_scale_);
}

infinicore::Tensor MiniCPM5MoeAttention::forward(const infinicore::Tensor &position_ids,
                                                 const infinicore::Tensor &hidden_states) const {
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];
    auto hidden_states_mutable = hidden_states;
    auto [qg, k, v] = qkv_proj_->forward_split(hidden_states_mutable);

    infinicore::Tensor q;
    if (use_gated_attention_) {
        if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
            auto qg_view = qg->view({batch_size, seq_len, num_attention_heads_, 2 * head_dim_});
            q = qg_view->narrow({{3, 0, head_dim_}})->contiguous();
            auto gate = qg_view->narrow({{3, head_dim_, head_dim_}})->contiguous();
            store_gate_score(gate_score_cache_,
                             gate->view({batch_size, seq_len, num_attention_heads_ * head_dim_}));
        } else {
            ASSERT_EQ(batch_size, 1);
            auto qg_view = qg->view({seq_len, num_attention_heads_, 2 * head_dim_});
            q = qg_view->narrow({{2, 0, head_dim_}})->contiguous();
            auto gate = qg_view->narrow({{2, head_dim_, head_dim_}})->contiguous();
            store_gate_score(gate_score_cache_,
                             gate->view({seq_len, num_attention_heads_ * head_dim_}));
        }
    } else if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        q = qg->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    } else {
        ASSERT_EQ(batch_size, 1);
        q = qg->view({seq_len, num_attention_heads_, head_dim_});
    }

    auto pos_shape = position_ids->shape();
    infinicore::Tensor pos_ids_for_rope = position_ids;
    if (pos_shape.size() == 2) {
        pos_ids_for_rope = position_ids->narrow({{0, 0, 1}})->contiguous()->view({pos_shape[1]});
    } else if (pos_shape.size() == 1) {
        pos_ids_for_rope = position_ids->contiguous();
    } else {
        throw std::runtime_error("MiniCPM5MoeAttention: Unexpected position_ids shape");
    }

    infinicore::Tensor attn_output;
    if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        auto k_reshaped = k->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
        auto v_reshaped = v->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
        auto q_rope = infinicore::Tensor::empty({batch_size, num_attention_heads_, seq_len, head_dim_},
                                                q->dtype(), q->device())
                          ->permute({0, 2, 1, 3});
        rotary_emb_->forward(q_rope, q, pos_ids_for_rope);
        rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);
        attn_output = attn_->forward(q_rope, k_reshaped, v_reshaped);
    } else {
        auto k_reshaped = k->view({seq_len, num_key_value_heads_, head_dim_});
        auto v_reshaped = v->view({seq_len, num_key_value_heads_, head_dim_});
        if (!q->is_contiguous()) q = q->contiguous();
        if (!k_reshaped->is_contiguous()) k_reshaped = k_reshaped->contiguous();
        if (!v_reshaped->is_contiguous()) v_reshaped = v_reshaped->contiguous();
        rotary_emb_->forward(q, pos_ids_for_rope, true);
        rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);
        attn_output = attn_->forward(q, k_reshaped, v_reshaped);
    }

    if (use_gated_attention_) {
        attn_output = apply_gate_sigmoid_mul(attn_output, gate_score_cache_, gate_sigmoid_buf_);
    }
    auto o = o_proj_->forward(attn_output);
    return o;
}

void MiniCPM5MoeAttention::forward_pre_attn_piecewise(
    const infinicore::Tensor &position_ids,
    const infinicore::Tensor &hidden_states,
    global_state::PiecewiseLayerStaging &staging) const {
    auto &piecewise = global_state::get_forward_context().piecewise;
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];
    ASSERT_EQ(batch_size, 1);

    auto [qg, k, v] = qkv_proj_->forward_split(hidden_states_mutable);
    const size_t valid_len = piecewise.valid_seq_len > 0 ? piecewise.valid_seq_len : seq_len;

    infinicore::Tensor q_heads;
    if (use_gated_attention_) {
        auto qg_view = qg->view({1, seq_len, num_attention_heads_, 2 * head_dim_});
        auto q = qg_view->narrow({{3, 0, head_dim_}})->contiguous();
        auto gate = qg_view->narrow({{3, head_dim_, head_dim_}})->contiguous();
        store_gate_score(gate_score_cache_,
                         gate->view({seq_len, num_attention_heads_ * head_dim_}));
        q_heads = q->view({1, seq_len, num_attention_heads_, head_dim_});
    } else {
        q_heads = qg->view({1, seq_len, num_attention_heads_, head_dim_});
    }
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
        pos_ids_for_rope = position_ids->narrow({{0, 0, 1}})->contiguous()->view({pos_shape[1]});
    } else if (pos_shape.size() == 1) {
        pos_ids_for_rope = position_ids->contiguous();
    } else {
        throw std::runtime_error("MiniCPM5MoeAttention: Unexpected position_ids shape");
    }
    if (pos_ids_for_rope->size(0) > valid_len) {
        pos_ids_for_rope = pos_ids_for_rope->narrow({{0, 0, valid_len}})->contiguous();
    }

    auto q_rope = staging.q_rope->view({seq_len, num_attention_heads_, head_dim_})->narrow({{0, 0, valid_len}});
    auto k_rope = staging.k_rope->view({seq_len, num_key_value_heads_, head_dim_})->narrow({{0, 0, valid_len}});
    if (!q_rope->is_contiguous()) q_rope = q_rope->contiguous();
    if (!k_rope->is_contiguous()) k_rope = k_rope->contiguous();
    rotary_emb_->forward(q_rope, pos_ids_for_rope, true);
    rotary_emb_->forward(k_rope, pos_ids_for_rope, true);
}

void MiniCPM5MoeAttention::forward_eager_attn_piecewise(
    const infinicore::Tensor &,
    global_state::PiecewiseLayerStaging &staging) const {
    auto &piecewise = global_state::get_forward_context().piecewise;
    const size_t seq_len = staging.q_rope->size(1);
    const size_t valid_len = piecewise.valid_seq_len > 0 ? piecewise.valid_seq_len : seq_len;

    auto q = staging.q_rope->view({seq_len, num_attention_heads_, head_dim_})->narrow({{0, 0, valid_len}});
    auto k = staging.k_rope->view({seq_len, num_key_value_heads_, head_dim_})->narrow({{0, 0, valid_len}});
    auto v = staging.v_rope->view({seq_len, num_key_value_heads_, head_dim_})->narrow({{0, 0, valid_len}});
    if (!q->is_contiguous()) q = q->contiguous();
    if (!k->is_contiguous()) k = k->contiguous();
    if (!v->is_contiguous()) v = v->contiguous();

    auto attn_output = attn_->forward(q, k, v);
    if (use_gated_attention_ && gate_score_cache_) {
        auto gate = gate_score_cache_;
        if (gate->shape().size() == 2 && gate->shape()[0] >= valid_len) {
            gate = gate->narrow({{0, 0, valid_len}});
        }
        attn_output = apply_gate_sigmoid_mul(attn_output, gate, gate_sigmoid_buf_);
    }

    if (valid_len < seq_len) {
        staging.attn_output->narrow({{1, 0, valid_len}})->copy_from(attn_output->narrow({{1, 0, valid_len}}));
        auto attn_tail = staging.attn_output->narrow({{1, valid_len, seq_len - valid_len}});
        set_zeros(attn_tail);
    } else {
        staging.attn_output->copy_from(attn_output);
    }
}

void MiniCPM5MoeAttention::forward_post_attn_piecewise_into(
    infinicore::Tensor &hidden_states,
    global_state::PiecewiseLayerStaging &staging) const {
    forward_post_attn_piecewise_graph_into(hidden_states, staging);
    forward_post_attn_piecewise_allreduce_into(hidden_states, staging);
}

void MiniCPM5MoeAttention::forward_post_attn_piecewise_graph_into(
    infinicore::Tensor &hidden_states,
    global_state::PiecewiseLayerStaging &staging) const {
    auto &piecewise = global_state::get_forward_context().piecewise;
    const size_t seq_len = staging.attn_output->size(1);
    const size_t valid_len = piecewise.valid_seq_len > 0 ? piecewise.valid_seq_len : seq_len;
    auto hidden_narrow = hidden_states->narrow({{1, 0, valid_len}});
    auto attn_for_proj = staging.attn_output->narrow({{1, 0, valid_len}});
    auto projected = o_proj_->forward_matmul_only(attn_for_proj);
    hidden_narrow->copy_from(projected);
    if (valid_len < seq_len) {
        auto hidden_tail = hidden_states->narrow({{1, valid_len, seq_len - valid_len}});
        set_zeros(hidden_tail);
    }
}

void MiniCPM5MoeAttention::forward_post_attn_piecewise_cg_into(
    infinicore::Tensor &hidden_states,
    global_state::PiecewiseLayerStaging &staging) const {
    auto &piecewise = global_state::get_forward_context().piecewise;
    const size_t seq_len = staging.attn_output->size(1);
    const size_t valid_len = piecewise.valid_seq_len > 0 ? piecewise.valid_seq_len : seq_len;
    auto hidden_narrow = hidden_states->narrow({{1, 0, valid_len}});
    auto attn_for_proj = staging.attn_output->narrow({{1, 0, valid_len}});
    auto projected = o_proj_->forward(attn_for_proj);
    hidden_narrow->copy_from(projected);
    if (valid_len < seq_len) {
        auto hidden_tail = hidden_states->narrow({{1, valid_len, seq_len - valid_len}});
        set_zeros(hidden_tail);
    }
}

void MiniCPM5MoeAttention::forward_post_attn_piecewise_allreduce_into(
    infinicore::Tensor &hidden_states,
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

infinicore::op::inductor_segment_impl::PreAttnExternalWeightTensors
MiniCPM5MoeAttention::pre_attn_external_weights() const {
    infinicore::op::inductor_segment_impl::PreAttnExternalWeightTensors out;
    out.q_weight = qkv_proj_->q_weight();
    out.k_weight = qkv_proj_->k_weight();
    out.v_weight = qkv_proj_->v_weight();
    const auto &device = out.q_weight->device();
    const auto dtype = out.q_weight->dtype();
    out.q_norm_weight = infinicore::Tensor::empty({0}, dtype, device);
    out.k_norm_weight = infinicore::Tensor::empty({0}, dtype, device);
    return out;
}

} // namespace infinilm::models::minicpm5_moe
