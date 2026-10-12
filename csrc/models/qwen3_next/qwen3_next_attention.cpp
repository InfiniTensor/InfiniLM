#include "execution/submit.hpp"
#include "../../weights/weights.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include "qwen3_next_attention.hpp"

#include "../../global_state/forward_context.hpp"
#include "runtime/runtime.hpp"
#include "../../layers/attention/attention.hpp"
#include "../../utils.hpp"
#include <cmath>
#include <exception>
#include <stdexcept>

namespace infinilm::models::qwen3_next {

Qwen3NextAttention::Qwen3NextAttention(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx, const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording() || !model_config || rank_info.device != runtime.device()
        || rank_info.tp_size <= 0 || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size
        || (rank_info.tp_size > 1 && !rank_info.comm))
        throw std::invalid_argument("Next construction requires matching eager Runtime/rank");
    device_ = runtime.device();
    layer_idx_ = layer_idx;
    const auto dtype = model_config->get_dtype();
    hidden_size_ = model_config->get<size_t>("hidden_size");
    head_dim_ = model_config->get<size_t>("head_dim");
    const size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
    const size_t total_num_kv_heads = model_config->get<size_t>("num_key_value_heads");
    const bool use_bias = model_config->get_or<bool>("attention_bias", true);
    const bool use_output_bias = model_config->get_or<bool>("attention_output_bias", false);
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");
    const bool attn_output_gate = model_config->get_or<bool>("attn_output_gate", true);
    attention_backend_ = attention_backend;
    const size_t tp_size = static_cast<size_t>(rank_info.tp_size);
    if (!total_num_heads || !total_num_kv_heads || total_num_heads % tp_size
        || (total_num_kv_heads >= tp_size ? total_num_kv_heads % tp_size : tp_size % total_num_kv_heads))
        throw std::invalid_argument("Qwen3NextAttention: invalid TP head geometry");
    num_attention_heads_ = total_num_heads / tp_size;
    num_key_value_heads_ = total_num_kv_heads < tp_size ? 1 : total_num_kv_heads / tp_size;
    auto quantization_method = model_config->get_quantization_method();
    try {
        auto register_fn = [this](const std::string &n, infinilm::weights::Weight p) { this->bind_weight(n, std::move(p)); };
        qkv_proj_ = std::make_shared<layers::linear::QKVParallelLinear>(
            runtime, hidden_size_, head_dim_, total_num_heads * (1 + attn_output_gate), total_num_kv_heads,
            "q_proj", "k_proj", "v_proj", register_fn,
            quantization_method, use_bias, dtype, rank_info);
        o_proj_ = std::make_shared<layers::linear::RowParallelLinear>(
            runtime, total_num_heads * head_dim_, hidden_size_, quantization_method,
            use_output_bias, dtype, rank_info.tp_rank, rank_info.tp_size, rank_info.comm);
        this->bind_child("o_proj", o_proj_);
        rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(runtime, model_config, runtime.device());
        const float scaling = 1.0f / std::sqrt(static_cast<float>(head_dim_));
        attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(
            num_attention_heads_, head_dim_, scaling, num_key_value_heads_, layer_idx_,
            kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_, model_config->get_kv_quant_scheme());
        q_norm_ = std::make_shared<infinilm::layers::RMSNorm>(runtime, head_dim_, rms_norm_eps, dtype);
        this->bind_child("q_norm", q_norm_);
        k_norm_ = std::make_shared<infinilm::layers::RMSNorm>(runtime, head_dim_, rms_norm_eps, dtype);
        this->bind_child("k_norm", k_norm_);
        infinilm::layers::attention::init_kv_cache_quant_params(runtime, register_fn,
            kv_cache_k_scale_, kv_cache_v_scale_, model_config->get_kv_quant_scheme());
    } catch (...) {
        const auto failure = std::current_exception();
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}

void Qwen3NextAttention::process_weights_after_loading(infinilm::runtime::Runtime &runtime, bool pre_transpose) {
    runtime.assertOwnerThread();
    if (runtime.device() != device_)
        throw std::invalid_argument("Qwen3NextAttention: postload Runtime device mismatch");
    if (runtime.graphCaptureState() != infinilm::execution::graph::GraphManager::CaptureState::kInactive)
        throw std::logic_error("Qwen3NextAttention: postload requires inactive graph state");
    // o_proj and q/k norms are registered; only qkv_proj is hidden.
    auto old_parameters = qkv_proj_->state_dict();
    auto old_aliases = local_weights_;
    try {
        qkv_proj_->process_weights_after_loading(runtime, pre_transpose);
        runtime.syncStream();
    } catch (...) {
        const auto failure = std::current_exception();
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}

infinilm::runtime::Tensor Qwen3NextAttention::forward(const infinilm::runtime::Tensor &positions,
                                               const infinilm::runtime::Tensor &hidden_states,
                                               infinilm::global_state::ForwardContext &forward_context) const {
    if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        return forward_static_(positions, hidden_states, forward_context);
    }
    return forward_paged_(positions, hidden_states, forward_context);
}

infinilm::runtime::Tensor Qwen3NextAttention::forward_static_(const infinilm::runtime::Tensor &position_ids,
                                                       const infinilm::runtime::Tensor &hidden_states,
                                                       infinilm::global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    auto [q_proj_out, k, v] = qkv_proj_->forward_split(runtime, hidden_states_mutable);
    auto q_proj_heads = q_proj_out->as_strided(
        {batch_size * seq_len, num_attention_heads_, head_dim_ * 2},
        {q_proj_out->stride(1), static_cast<infinilm::runtime::Stride>(head_dim_ * 2), 1});
    auto q = q_proj_heads->narrow({{2, 0, head_dim_}});
    auto gate = q_proj_heads->narrow({{2, head_dim_, head_dim_}})
                    ->contiguous(runtime)
                    ->view({batch_size, seq_len, num_attention_heads_ * head_dim_});

    q = q_norm_->forward(runtime, q);
    auto k_heads = k->as_strided(
        {batch_size * seq_len, num_key_value_heads_, head_dim_},
        {k->stride(1), static_cast<infinilm::runtime::Stride>(head_dim_), 1});
    k = k_norm_->forward(runtime, k_heads);

    auto q_reshaped = q->as_strided(
        {batch_size, seq_len, num_attention_heads_, head_dim_},
        {static_cast<infinilm::runtime::Stride>(seq_len * num_attention_heads_ * head_dim_),
         static_cast<infinilm::runtime::Stride>(num_attention_heads_ * head_dim_),
         static_cast<infinilm::runtime::Stride>(head_dim_),
         1});
    auto k_reshaped = k->as_strided(
        {batch_size, seq_len, num_key_value_heads_, head_dim_},
        {static_cast<infinilm::runtime::Stride>(seq_len * num_key_value_heads_ * head_dim_),
         static_cast<infinilm::runtime::Stride>(num_key_value_heads_ * head_dim_),
         static_cast<infinilm::runtime::Stride>(head_dim_),
         1});
    auto v_reshaped = v->as_strided(
        {batch_size, seq_len, num_key_value_heads_, head_dim_},
        {v->stride(0), v->stride(1), static_cast<infinilm::runtime::Stride>(head_dim_), 1});

    auto pos_shape = position_ids->shape();
    infinilm::runtime::Tensor pos_ids_for_rope = position_ids;
    if (pos_shape.size() == 2) {
        ASSERT_EQ(pos_shape[0], batch_size);
        ASSERT_EQ(pos_shape[1], seq_len);
        pos_ids_for_rope = position_ids->contiguous(runtime);
    } else if (pos_shape.size() == 1) {
        ASSERT_EQ(batch_size, 1);
        ASSERT_EQ(pos_shape[0], seq_len);
        pos_ids_for_rope = position_ids->contiguous(runtime)->view({1, seq_len});
    } else {
        throw std::runtime_error("infinilm::models::qwen3_next::Qwen3NextAttention: Unexpected position_ids shape");
    }

    rotary_emb_->forward(runtime, q_reshaped, pos_ids_for_rope, true);
    rotary_emb_->forward(runtime, k_reshaped, pos_ids_for_rope, true);

    auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped, forward_context);
    {
        const auto ops_other = [&]() {
            auto ops_output = infinilm::runtime::Tensor::empty(runtime, gate->shape(), gate->dtype());
            INFINILM_ASSERT(runtime.device().type() != infinilm::runtime::Device::Type::kCpu
                && runtime.device().type() != infinilm::runtime::Device::Type::kMars
                && runtime.device().type() != infinilm::runtime::Device::Type::kThead);
            infinilm::execution::submit<infini::ops::Sigmoid>(
                runtime, infini::ops::Sigmoid::DefaultConfig(runtime.device().type()), gate, ops_output);
            INFINILM_TRACE_OP(Sigmoid, ops_output);
            return ops_output;
        }();
        auto ops_output = infinilm::runtime::Tensor::empty(runtime, attn_output->shape(), attn_output->dtype());
        INFINILM_ASSERT(runtime.device().type() != infinilm::runtime::Device::Type::kCpu
            && runtime.device().type() != infinilm::runtime::Device::Type::kMars
            && runtime.device().type() != infinilm::runtime::Device::Type::kThead);
        infinilm::execution::submit<infini::ops::Mul>(
            runtime, infini::ops::Mul::DefaultConfig(runtime.device().type()), attn_output, ops_other, ops_output);
        INFINILM_TRACE_OP(Mul, ops_output);
        attn_output = std::move(ops_output);
    }
    return o_proj_->forward(runtime, attn_output);
}

infinilm::runtime::Tensor Qwen3NextAttention::forward_paged_(const infinilm::runtime::Tensor &position_ids,
                                                      const infinilm::runtime::Tensor &hidden_states,
                                                      infinilm::global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    ASSERT_EQ(batch_size, 1);

    auto [q_proj_out, k, v] = qkv_proj_->forward_split(runtime, hidden_states_mutable);
    auto q_proj_heads = q_proj_out->as_strided(
        {seq_len, num_attention_heads_, head_dim_ * 2},
        {q_proj_out->stride(1), static_cast<infinilm::runtime::Stride>(head_dim_ * 2), 1});
    auto q = q_proj_heads->narrow({{2, 0, head_dim_}});
    auto gate = q_proj_heads->narrow({{2, head_dim_, head_dim_}})
                    ->contiguous(runtime)
                    ->view({seq_len, num_attention_heads_ * head_dim_});

    auto q_reshaped = q_norm_->forward(runtime, q);
    auto k_heads = k->as_strided(
        {seq_len, num_key_value_heads_, head_dim_},
        {k->stride(1), static_cast<infinilm::runtime::Stride>(head_dim_), 1});
    auto k_reshaped = k_norm_->forward(runtime, k_heads);
    auto v_reshaped = v->as_strided(
        {seq_len, num_key_value_heads_, head_dim_},
        {v->stride(1), static_cast<infinilm::runtime::Stride>(head_dim_), 1});

    auto pos_shape = position_ids->shape();
    infinilm::runtime::Tensor pos_ids_for_rope = position_ids;
    if (pos_shape.size() == 2) {
        auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
        pos_ids_for_rope = pos_narrowed->view({pos_shape[1]});
    } else if (pos_shape.size() == 1) {
        pos_ids_for_rope = position_ids;
    } else {
        throw std::runtime_error("infinilm::models::qwen3_next::Qwen3NextAttention: Unexpected position_ids shape");
    }

    rotary_emb_->forward(runtime, q_reshaped, pos_ids_for_rope, true);
    rotary_emb_->forward(runtime, k_reshaped, pos_ids_for_rope, true);

    auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped, forward_context);
    {
        const auto ops_other = [&]() {
            auto ops_output = infinilm::runtime::Tensor::empty(runtime, gate->shape(), gate->dtype());
            INFINILM_ASSERT(runtime.device().type() != infinilm::runtime::Device::Type::kCpu
                && runtime.device().type() != infinilm::runtime::Device::Type::kMars
                && runtime.device().type() != infinilm::runtime::Device::Type::kThead);
            infinilm::execution::submit<infini::ops::Sigmoid>(
                runtime, infini::ops::Sigmoid::DefaultConfig(runtime.device().type()), gate, ops_output);
            INFINILM_TRACE_OP(Sigmoid, ops_output);
            return ops_output;
        }()->view(attn_output->shape());
        auto ops_output = infinilm::runtime::Tensor::empty(runtime, attn_output->shape(), attn_output->dtype());
        INFINILM_ASSERT(runtime.device().type() != infinilm::runtime::Device::Type::kCpu
            && runtime.device().type() != infinilm::runtime::Device::Type::kMars
            && runtime.device().type() != infinilm::runtime::Device::Type::kThead);
        infinilm::execution::submit<infini::ops::Mul>(
            runtime, infini::ops::Mul::DefaultConfig(runtime.device().type()), attn_output, ops_other, ops_output);
        INFINILM_TRACE_OP(Mul, ops_output);
        attn_output = std::move(ops_output);
    }
    return o_proj_->forward(runtime, attn_output);
}

} // namespace infinilm::models::qwen3_next
