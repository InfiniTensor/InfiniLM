#include "attention.hpp"
#include "../../execution/submit.hpp"
#include "../../global_state/forward_context.hpp"
#include "../../utils.hpp"
#include "../../weights/weights.hpp"
#include "../moe/ep/ep_config.hpp"
#include "../rotary_embedding/rotary_embedding.hpp"
#include "runtime/utils.hpp"
#include <exception>
#include <limits>

namespace infinilm::layers::attention {

void detail::apply_output_gate(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor &output,
                               const infinilm::runtime::Tensor &gate, bool view_sigmoid_output) {
    const auto sigmoid = [&]() {
        auto packed_gate = gate->contiguous(runtime);
        auto ops_output = infinilm::runtime::Tensor::empty(runtime, gate->shape(), gate->dtype());
        infinilm::execution::submit<infini::ops::Sigmoid>(
            runtime, infini::ops::Sigmoid::DefaultConfig(runtime.device().type()), packed_gate, ops_output);
        INFINILM_TRACE_OP(Sigmoid, ops_output);
        return ops_output;
    };
    const auto ops_other = view_sigmoid_output ? sigmoid()->view(output->shape()) : sigmoid();
    auto ops_output = infinilm::runtime::Tensor::empty(runtime, output->shape(), output->dtype());
    infinilm::execution::submit<infini::ops::Mul>(
        runtime, infini::ops::Mul::DefaultConfig(runtime.device().type()), output, ops_other, ops_output);
    INFINILM_TRACE_OP(Mul, ops_output);
    output = std::move(ops_output);
}

Attention::Attention(infinilm::runtime::Runtime &runtime,
                     std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     size_t layer_idx, const engine::distributed::RankInfo &rank_info,
                     ::infinilm::backends::AttentionBackend attention_backend,
                     const infinilm::layers::moe::EPConfig &ep_config) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) {
        throw std::logic_error("Attention construction requires eager Runtime");
    }
    if (!model_config || rank_info.device != runtime.device() || rank_info.tp_size <= 0
        || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size
        || (rank_info.tp_size > 1 && !rank_info.comm)
        || ep_config.backend != infinilm::layers::moe::EPBackend::Disabled) {
        throw std::invalid_argument("Invalid explicit Attention construction resources");
    }
    if (attention_backend != ::infinilm::backends::AttentionBackend::STATIC_ATTN
        && attention_backend != ::infinilm::backends::AttentionBackend::PAGED_ATTN
        && attention_backend != ::infinilm::backends::AttentionBackend::FLASH_ATTN) {
        throw std::invalid_argument("Unsupported explicit Attention backend");
    }
    for (const char *key : {"hidden_size", "head_dim", "num_attention_heads", "num_key_value_heads"}) {
        if (model_config->get<int64_t>(key) <= 0) {
            throw std::invalid_argument("Attention requires positive dimensions");
        }
    }
    const size_t hidden = model_config->get<size_t>("hidden_size"), dim = model_config->get<size_t>("head_dim");
    const size_t heads = model_config->get<size_t>("num_attention_heads"), kv = model_config->get<size_t>("num_key_value_heads");
    const size_t tp = static_cast<size_t>(rank_info.tp_size);
    if (dim < 2 || heads % kv != 0 || heads % tp != 0 || (kv >= tp ? kv % tp != 0 : tp % kv != 0)) {
        throw std::invalid_argument("Attention TP/GQA/RoPE geometry is invalid");
    }
    const auto dtype = model_config->get_dtype();
    if (dtype != infinilm::runtime::DataType::kFloat32 && dtype != infinilm::runtime::DataType::kFloat16
        && dtype != infinilm::runtime::DataType::kBFloat16) {
        throw std::invalid_argument("Attention RoPE requires floating model dtype");
    }
    const size_t limit = std::numeric_limits<size_t>::max();
    const auto product = [limit](size_t a, size_t b) {
        if (b && a > limit / b) {
            throw std::length_error("Attention construction size overflow");
        }
        return a * b;
    };
    const size_t q = product(heads, dim), k = product(product(kv, dim), kv < tp ? tp / kv : 1), twice_k = product(k, 2);
    if (q > limit - twice_k) {
        throw std::length_error("Attention packed QKV size overflow");
    }
    (void)product(product(q + twice_k, hidden), infinilm::runtime::dsize(dtype));
    (void)product(product(q, hidden), infinilm::runtime::dsize(dtype));
    const auto kv_scheme = model_config->get_kv_quant_scheme();
    if (kv_scheme != infinilm::quantization::KVQuantAlgo::NONE && kv_scheme != infinilm::quantization::KVQuantAlgo::INT8) {
        throw std::invalid_argument("unsupported kv_quant_scheme");
    }
    auto quantization_method = model_config->get_quantization_method();
    construction_owner_ = &runtime;
    layer_idx_ = layer_idx;
    hidden_size_ = hidden;
    head_dim_ = dim;
    num_attention_heads_ = heads / tp;
    num_key_value_heads_ = kv < tp ? 1 : kv / tp;
    attention_backend_ = attention_backend;
    try {
        auto register_fn = [this](const std::string &name, infinilm::weights::Weight parameter) {
            this->bind_weight(name, std::move(parameter));
        };
        qkv_proj_ = std::make_shared<layers::linear::QKVParallelLinear>(
            runtime, hidden, dim, heads, kv, "q_proj", "k_proj", "v_proj", register_fn,
            quantization_method, model_config->get_or<bool>("attention_bias", true), dtype, rank_info);
        o_proj_ = std::make_shared<layers::linear::RowParallelLinear>(
            runtime, q, hidden, quantization_method, model_config->get_or<bool>("attention_output_bias", false),
            dtype, rank_info.tp_rank, rank_info.tp_size, rank_info.comm);
        this->bind_child("o_proj", o_proj_); // Member retains owner if registration throws.
        rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(runtime, model_config, runtime.device());
        // Backend takes Tensor values, so both owning scales must precede it.
        init_kv_cache_quant_params(runtime, register_fn, kv_cache_k_scale_, kv_cache_v_scale_, kv_scheme);
        attn_ = std::make_shared<AttentionLayer>(num_attention_heads_, dim,
                                                 1.0f / std::sqrt(static_cast<float>(dim)), num_key_value_heads_, layer_idx_,
                                                 kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_, kv_scheme,
                                                 model_config->get<std::string>("model_type") == "mistral"
                                                     ? model_config->get<std::vector<int64_t>>("infinilm_window_size")
                                                     : std::vector<int64_t>{-1, -1});
    } catch (...) {
        // Completed child/parameter owners remain members until after this drain.
        try {
            runtime.syncStream();
        } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor Attention::forward(
    const infinilm::runtime::Tensor &positions,
    const infinilm::runtime::Tensor &hidden_states,
    infinilm::global_state::ForwardContext &forward_context) const {
    if (construction_owner_) {
        construction_owner_->assertOwnerThread();
        if (forward_context.runtime != construction_owner_) {
            throw std::invalid_argument("Attention forward Runtime differs from construction owner");
        }
    }
    if (!forward_context.runtime) {
        throw std::invalid_argument("attention forward requires its worker Runtime");
    }
    if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        return forward_static_(positions, hidden_states, forward_context);
    }
    return forward_paged_(positions, hidden_states, forward_context);
}

infinilm::runtime::Tensor Attention::forward_static_(const infinilm::runtime::Tensor &position_ids,
                                                     const infinilm::runtime::Tensor &hidden_states,
                                                     infinilm::global_state::ForwardContext &forward_context) const {
    // hidden_states shape: [batch, seq_len, hidden_size]
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    // 1. Project Q, K, V
    auto [q, k, v] = qkv_proj_->forward_split(*forward_context.runtime, hidden_states_mutable);

    // 2. Reshape for multi-head attention
    auto q_reshaped = q->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

    // 3. Prepare position_ids for RoPE
    auto pos_shape = position_ids->shape();
    infinilm::runtime::Tensor pos_ids_for_rope = position_ids;
    if (pos_shape.size() == 2) {
        ASSERT_EQ(pos_shape[0], batch_size);
        ASSERT_EQ(pos_shape[1], seq_len);
        pos_ids_for_rope = position_ids->contiguous(*forward_context.runtime);
    } else if (pos_shape.size() == 1) {
        ASSERT_EQ(batch_size, 1);
        ASSERT_EQ(pos_shape[0], seq_len);
        pos_ids_for_rope = position_ids->contiguous(*forward_context.runtime)->view({1, seq_len});
    } else {
        throw std::runtime_error("infinilm::layers::attention::Attention: Unexpected position_ids shape");
    }

    // 4. Apply RoPE to QK
    rotary_emb_->forward(*forward_context.runtime, q_reshaped, pos_ids_for_rope, true);
    rotary_emb_->forward(*forward_context.runtime, k_reshaped, pos_ids_for_rope, true);

    // 5. Attn Backend calculate
    auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped, forward_context);

    // 7. Project output
    auto output = o_proj_->forward(*forward_context.runtime, attn_output);
    return output;
}

infinilm::runtime::Tensor Attention::forward_paged_(const infinilm::runtime::Tensor &position_ids,
                                                    const infinilm::runtime::Tensor &hidden_states,
                                                    infinilm::global_state::ForwardContext &forward_context) const {
    // hidden_states shape: [batch, seq_len, hidden_size]
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    // Only support batchsize==1, all requests should be flattened along seqlen dimension
    ASSERT_EQ(batch_size, 1);

    // 1. Project Q, K, V
    auto [q, k, v] = qkv_proj_->forward_split(*forward_context.runtime, hidden_states_mutable);

    // 2. Reshape for multi-head attention
    auto q_reshaped = q->view({seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->view({seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({seq_len, num_key_value_heads_, head_dim_});

    // 3. Prepare position_ids for RoPE
    auto pos_shape = position_ids->shape();
    infinilm::runtime::Tensor pos_ids_for_rope = position_ids;
    if (pos_shape.size() == 2) {
        auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
        pos_ids_for_rope = pos_narrowed->view({pos_shape[1]});
    } else if (pos_shape.size() == 1) {
        pos_ids_for_rope = position_ids;
    } else {
        throw std::runtime_error("Unexpected position_ids shape");
    }

    // 4. Apply RoPE to QK
    rotary_emb_->forward(*forward_context.runtime, q_reshaped, pos_ids_for_rope, true);
    rotary_emb_->forward(*forward_context.runtime, k_reshaped, pos_ids_for_rope, true);

    // 5. Attn Backend calculate
    auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped, forward_context);

    // 6. Project output
    auto output = o_proj_->forward(*forward_context.runtime, attn_output);
    return output;
}

} // namespace infinilm::layers::attention
