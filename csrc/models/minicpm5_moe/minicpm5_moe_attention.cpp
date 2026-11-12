#include "minicpm5_moe_attention.hpp"

#include "../../global_state/piecewise_prefill_state.hpp"

#include "../../global_state/global_state.hpp"
#include "../../layers/attention/attention.hpp"
#include "../../utils.hpp"
#include "infinicore/context/context.hpp"

#include <cmath>
#include <infinicore/ops/mul.hpp>
#include <infinicore/ops/sigmoid.hpp>
#include <stdexcept>

namespace infinilm::models::minicpm5_moe {

MiniCPM5MoEAttention::MiniCPM5MoEAttention(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const infinicore::Device &device) {
    layer_idx_ = layer_idx;
    const auto &dtype{model_config->get_dtype()};
    hidden_size_ = model_config->get<size_t>("hidden_size");
    head_dim_ = model_config->get<size_t>("head_dim");
    use_gated_attention_ = model_config->get_or<bool>("use_gated_attention", true);

    size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
    size_t total_num_kv_heads = model_config->get<size_t>("num_key_value_heads");
    bool use_bias = model_config->get_or<bool>("attention_bias", false);
    bool use_output_bias = model_config->get_or<bool>("attention_output_bias", false);

    attention_backend_ = infinilm::global_state::get_infinilm_config().attention_backend;
    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
    int tp_size = infinilm::global_state::get_tensor_model_parallel_world_size();

    num_attention_heads_ = total_num_heads / tp_size;
    num_key_value_heads_ = total_num_kv_heads < static_cast<size_t>(tp_size)
                             ? 1
                             : total_num_kv_heads / tp_size;

    // Gated path doubles q_proj (per-head [q | gate]), same packing as qwen3_next.
    const size_t q_heads_for_proj = total_num_heads * (use_gated_attention_ ? 2 : 1);

    auto quantization_method = model_config->get_quantization_method();
    auto register_fn = [this](const std::string &n, infinicore::nn::Parameter p) {
        this->register_parameter(n, std::move(p));
    };
    qkv_proj_ = std::make_shared<layers::linear::QKVParallelLinear>(
        hidden_size_, head_dim_, q_heads_for_proj, total_num_kv_heads,
        "q_proj", "k_proj", "v_proj", register_fn,
        quantization_method, use_bias, dtype, device, rank_info);
    o_proj_ = this->register_module<layers::linear::RowParallelLinear>(
        "o_proj", total_num_heads * head_dim_, hidden_size_, quantization_method,
        use_output_bias, dtype, device, tp_rank, tp_size, rank_info.comm);

    rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(model_config, device);

    float scaling = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(
        num_attention_heads_, head_dim_, scaling, num_key_value_heads_, layer_idx_,
        kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_);

    infinilm::layers::attention::init_kv_cache_quant_params(
        register_fn, device, kv_cache_k_scale_, kv_cache_v_scale_);
}

infinicore::Tensor MiniCPM5MoEAttention::forward(
    const infinicore::Tensor &positions,
    const infinicore::Tensor &hidden_states) const {
    if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        return forward_static_(positions, hidden_states);
    }
    return forward_paged_(positions, hidden_states);
}

infinicore::Tensor MiniCPM5MoEAttention::forward_static_(
    const infinicore::Tensor &position_ids,
    const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    auto [q_proj_out, k, v] = qkv_proj_->forward_split(hidden_states_mutable);

    infinicore::Tensor q_reshaped;
    infinicore::Tensor gate;
    if (use_gated_attention_) {
        auto q_proj_heads = q_proj_out->as_strided(
            {batch_size * seq_len, num_attention_heads_, head_dim_ * 2},
            {q_proj_out->stride(1), static_cast<infinicore::Stride>(head_dim_ * 2), 1});
        auto q = q_proj_heads->narrow({{2, 0, head_dim_}});
        gate = q_proj_heads->narrow({{2, head_dim_, head_dim_}})
                   ->contiguous()
                   ->view({batch_size, seq_len, num_attention_heads_ * head_dim_});
        // Keep real gated strides for in-place QY RoPE; view as 4-D for static attn.
        q_reshaped = q->as_strided(
            {batch_size, seq_len, num_attention_heads_, head_dim_},
            {q_proj_out->stride(0), q_proj_out->stride(1),
             static_cast<infinicore::Stride>(head_dim_ * 2), 1});
    } else {
        q_reshaped = q_proj_out->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    }

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
        throw std::runtime_error(
            "MiniCPM5MoEAttention: Unexpected position_ids shape");
    }

    rotary_emb_->forward(q_reshaped, pos_ids_for_rope, true);
    rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);

    auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped);
    if (use_gated_attention_) {
        attn_output = infinicore::op::mul(attn_output, infinicore::op::sigmoid(gate));
    }
    return o_proj_->forward(attn_output);
}

infinicore::Tensor MiniCPM5MoEAttention::forward_paged_(
    const infinicore::Tensor &position_ids,
    const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    ASSERT_EQ(batch_size, 1);

    auto [q_proj_out, k, v] = qkv_proj_->forward_split(hidden_states_mutable);

    infinicore::Tensor q_reshaped;
    infinicore::Tensor gate;
    if (use_gated_attention_) {
        // Keep strided gated Q [seq,heads,dim] strides [8704,256,1]; QY RoPE
        // rotates in-place via rope_strided_launch (no pack/contiguous tax).
        auto q_proj_heads = q_proj_out->as_strided(
            {seq_len, num_attention_heads_, head_dim_ * 2},
            {q_proj_out->stride(1), static_cast<infinicore::Stride>(head_dim_ * 2), 1});
        auto q = q_proj_heads->narrow({{2, 0, head_dim_}});
        gate = q_proj_heads->narrow({{2, head_dim_, head_dim_}})
                   ->contiguous()
                   ->view({seq_len, num_attention_heads_ * head_dim_});
        q_reshaped = q;
    } else {
        q_reshaped = q_proj_out->view({seq_len, num_attention_heads_, head_dim_});
    }

    auto k_reshaped = k->view({seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({seq_len, num_key_value_heads_, head_dim_});

    auto pos_shape = position_ids->shape();
    infinicore::Tensor pos_ids_for_rope = position_ids;
    if (pos_shape.size() == 2) {
        auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
        pos_ids_for_rope = pos_narrowed->view({pos_shape[1]});
    } else if (pos_shape.size() == 1) {
        pos_ids_for_rope = position_ids;
    } else {
        throw std::runtime_error("MiniCPM5MoEAttention: Unexpected position_ids shape");
    }

    rotary_emb_->forward(q_reshaped, pos_ids_for_rope, true);
    rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);

    auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped);
    if (use_gated_attention_) {
        attn_output = infinicore::op::mul(
            attn_output, infinicore::op::sigmoid(gate)->view(attn_output->shape()));
    }
    return o_proj_->forward(attn_output);
}

namespace {

bool same_trailing_dims_(const infinicore::Shape &a, const infinicore::Shape &b) {
    if (a.size() != b.size() || a.empty()) {
        return false;
    }
    for (size_t i = 1; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

void ensure_same_shape_buf(infinicore::Tensor &buf, const infinicore::Tensor &proto) {
    if (buf && buf->shape() == proto->shape() && buf->dtype() == proto->dtype()
        && buf->device() == proto->device()) {
        return;
    }
    buf = infinicore::Tensor::empty(proto->shape(), proto->dtype(), proto->device());
}

void store_gate_score(infinicore::Tensor &gate_buf,
                      infinicore::Tensor &gate_write_view,
                      const infinicore::Tensor &gate) {
    // Grow-only base buffer: multi-bucket piecewise CG records copy destinations by
    // device pointer. Shrinking (exact-shape realloc) after bucket=2048 → 16 made
    // earlier graphs write into freed/undersized memory → EM collapse / garbage.
    // During CG: only copy_from into a write view prepared outside recording
    // (narrow/empty HostOps inside startGraphRecording kill MetaX decode CG).
    const bool recording = infinicore::context::isGraphRecording();
    infinicore::Tensor g = gate->is_contiguous() ? gate : gate->contiguous();
    const auto &gshape = g->shape();

    const bool need_alloc = !gate_buf || gate_buf->dtype() != g->dtype()
        || gate_buf->device() != g->device()
        || !same_trailing_dims_(gate_buf->shape(), gshape)
        || gate_buf->shape()[0] < gshape[0];

    if (need_alloc) {
        if (recording) {
            throw std::runtime_error(
                "MiniCPM5MoEAttention: gate_score_cache_ grow required during CG "
                "(dry-run must grow-only allocate via C++ QKV path)");
        }
        auto alloc_shape = gshape;
        if (gate_buf && same_trailing_dims_(gate_buf->shape(), gshape)
            && gate_buf->dtype() == g->dtype() && gate_buf->device() == g->device()) {
            alloc_shape[0] = std::max(gate_buf->shape()[0], gshape[0]);
        }
        gate_buf = infinicore::Tensor::empty(alloc_shape, g->dtype(), g->device());
        gate_write_view = infinicore::Tensor{};
    }

    const bool view_ok = gate_write_view && gate_write_view->shape() == gshape
        && gate_write_view->dtype() == g->dtype()
        && gate_write_view->device() == g->device();
    if (!view_ok) {
        if (recording) {
            throw std::runtime_error(
                "MiniCPM5MoEAttention: gate_score_write_view_ unset/mismatch during CG "
                "(dry-run must narrow write view outside recording)");
        }
        if (gate_buf->shape() == gshape) {
            gate_write_view = gate_buf;
        } else {
            gate_write_view = gate_buf->narrow({{0, 0, gshape[0]}});
        }
    }
    gate_write_view->copy_from(g);
}

infinicore::Tensor apply_gate_sigmoid_mul(const infinicore::Tensor &attn_output,
                                          const infinicore::Tensor &gate_score,
                                          infinicore::Tensor &gate_sigmoid_buf) {
    auto gate_view = gate_score;
    if (attn_output->shape().size() == 3 && gate_score->shape().size() == 2
        && attn_output->shape()[0] == 1) {
        gate_view = gate_score->view({1, gate_score->shape()[0], gate_score->shape()[1]});
    }
    auto gate_in = gate_view->contiguous();
    ensure_same_shape_buf(gate_sigmoid_buf, gate_in);
    gate_sigmoid_buf = infinicore::op::sigmoid(gate_in);
    return infinicore::op::mul(attn_output->contiguous(), gate_sigmoid_buf);
}

} // namespace

void MiniCPM5MoEAttention::forward_pre_attn_qkv_piecewise(
    const infinicore::Tensor &,
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
        // Gate extract into stable per-layer cache (device copy; dry-run allocates).
        // Gate *apply* stays with eager attn (host-break).
        auto qg_view = qg->view({1, seq_len, num_attention_heads_, 2 * head_dim_});
        auto q = qg_view->narrow({{3, 0, head_dim_}})->contiguous();
        auto gate = qg_view->narrow({{3, head_dim_, head_dim_}})->contiguous();
        store_gate_score(gate_score_cache_,
                         gate_score_write_view_,
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
}

void MiniCPM5MoEAttention::forward_pre_attn_rope_piecewise(
    const infinicore::Tensor &position_ids,
    global_state::PiecewiseLayerStaging &staging) const {
    auto &piecewise = global_state::get_forward_context().piecewise;
    const size_t seq_len = staging.q_rope->size(1);
    const size_t valid_len = piecewise.valid_seq_len > 0 ? piecewise.valid_seq_len : seq_len;
    // Same address-stable contract as Attention::forward_pre_attn_rope_piecewise:
    // no alloc/rebind of position_ids / QK under recording; past/pad via contents.
    const bool addr_stable = infinicore::context::isGraphRecording()
        || piecewise.compile_capture_active;

    auto pos_shape = position_ids->shape();
    infinicore::Tensor pos_ids_for_rope = position_ids;
    if (pos_shape.size() == 2) {
        auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
        if (addr_stable) {
            pos_ids_for_rope = pos_narrowed->view({pos_shape[1]});
        } else {
            pos_ids_for_rope = pos_narrowed->contiguous()->view({pos_shape[1]});
        }
    } else if (pos_shape.size() == 1) {
        if (!addr_stable && !position_ids->is_contiguous()) {
            pos_ids_for_rope = position_ids->contiguous();
        }
    } else {
        throw std::runtime_error("MiniCPM5MoEAttention: Unexpected position_ids shape");
    }
    if (pos_ids_for_rope->size(0) > valid_len) {
        auto narrowed = pos_ids_for_rope->narrow({{0, 0, valid_len}});
        if (addr_stable) {
            if (!narrowed->is_contiguous()) {
                throw std::runtime_error(
                    "MiniCPM5MoEAttention::forward_pre_attn_rope_piecewise: "
                    "position_ids narrow not contiguous under graph recording");
            }
            pos_ids_for_rope = narrowed;
        } else {
            pos_ids_for_rope = narrowed->contiguous();
        }
    }

    auto q_rope = staging.q_rope->view({seq_len, num_attention_heads_, head_dim_})->narrow({{0, 0, valid_len}});
    auto k_rope = staging.k_rope->view({seq_len, num_key_value_heads_, head_dim_})->narrow({{0, 0, valid_len}});
    if (!q_rope->is_contiguous()) {
        if (addr_stable) {
            throw std::runtime_error(
                "MiniCPM5MoEAttention::forward_pre_attn_rope_piecewise: "
                "q_rope not contiguous under graph recording");
        }
        q_rope = q_rope->contiguous();
    }
    if (!k_rope->is_contiguous()) {
        if (addr_stable) {
            throw std::runtime_error(
                "MiniCPM5MoEAttention::forward_pre_attn_rope_piecewise: "
                "k_rope not contiguous under graph recording");
        }
        k_rope = k_rope->contiguous();
    }
    rotary_emb_->forward(q_rope, pos_ids_for_rope, true);
    rotary_emb_->forward(k_rope, pos_ids_for_rope, true);
}

void MiniCPM5MoEAttention::forward_pre_attn_piecewise(
    const infinicore::Tensor &position_ids,
    const infinicore::Tensor &hidden_states,
    global_state::PiecewiseLayerStaging &staging) const {
    forward_pre_attn_qkv_piecewise(position_ids, hidden_states, staging);
    forward_pre_attn_rope_piecewise(position_ids, staging);
}

void MiniCPM5MoEAttention::forward_eager_attn_piecewise(
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
        // Prefer grow-only cache + valid_len prefix: write_view_ may still be the
        // last capture/eager seq (e.g. 16) after a larger bucket CG replay wrote
        // into the shared base buffer.
        auto gate = gate_score_cache_;
        if (gate->shape().size() == 2 && gate->shape()[0] >= valid_len) {
            gate = gate->narrow({{0, 0, valid_len}});
        } else if (gate_score_write_view_ && gate_score_write_view_->shape().size() == 2
                   && gate_score_write_view_->shape()[0] >= valid_len) {
            gate = gate_score_write_view_->narrow({{0, 0, valid_len}});
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

void MiniCPM5MoEAttention::forward_post_attn_piecewise_into(
    infinicore::Tensor &hidden_states,
    global_state::PiecewiseLayerStaging &staging) const {
    forward_post_attn_piecewise_graph_into(hidden_states, staging);
    forward_post_attn_piecewise_allreduce_into(hidden_states, staging);
}

void MiniCPM5MoEAttention::forward_post_attn_piecewise_graph_into(
    infinicore::Tensor &hidden_states,
    global_state::PiecewiseLayerStaging &staging) const {
    // Dest RowParallelLinear has no matmul-only/allreduce split; reuse fused path.
    forward_post_attn_piecewise_cg_into(hidden_states, staging);
}

void MiniCPM5MoEAttention::forward_post_attn_piecewise_cg_into(
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

void MiniCPM5MoEAttention::forward_post_attn_piecewise_allreduce_into(
    infinicore::Tensor &hidden_states,
    global_state::PiecewiseLayerStaging &staging) const {
    (void)hidden_states;
    (void)staging;
    // TP>1 deferred allreduce not ported on post-refactor RowParallelLinear.
}

} // namespace infinilm::models::minicpm5_moe
