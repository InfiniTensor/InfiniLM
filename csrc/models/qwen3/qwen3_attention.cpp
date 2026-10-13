#include "../../weights/weights.hpp"
#include "../models_registry.hpp"
#include <exception>
#include "qwen3_attention.hpp"
#include "../../global_state/global_state.hpp"
#include "../../layers/attention/attention.hpp"
#include "../../utils.hpp"

namespace infinilm::models::qwen3 {

Qwen3Attention::Qwen3Attention(infinilm::runtime::Runtime &runtime, std::shared_ptr<config::ModelConfig> model_config,
                               size_t layer_idx, const engine::distributed::RankInfo &rank_info, backends::AttentionBackend attention_backend, const layers::moe::EPConfig &ep_config) {
    models::validate_selected_construction(model_config, runtime, rank_info, ep_config);
    try {

    layer_idx_ = layer_idx;
    hidden_size_ = model_config->get<size_t>("hidden_size");
    head_dim_ = model_config->get<size_t>("head_dim");

    const auto &dtype{model_config->get_dtype()};
    size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
    size_t total_num_kv_heads = model_config->get<size_t>("num_key_value_heads");
    bool use_bias = model_config->get_or<bool>("attention_bias", true);
    bool use_output_bias = model_config->get_or<bool>("attention_output_bias", false);
    double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    attention_backend_ = attention_backend;
    int tp_rank = rank_info.tp_rank;
    int tp_size = rank_info.tp_size;
    num_attention_heads_ = total_num_heads / tp_size;
    num_key_value_heads_ = total_num_kv_heads < static_cast<size_t>(tp_size)
                             ? 1
                             : total_num_kv_heads / tp_size;

    auto quantization_method = model_config->get_quantization_method();
    auto register_fn = [this](const std::string &n, infinilm::weights::Weight p) { this->bind_weight(n, std::move(p)); };
    qkv_proj_ = std::make_shared<layers::linear::QKVParallelLinear>(
        runtime, hidden_size_, head_dim_, total_num_heads, total_num_kv_heads,
        "q_proj", "k_proj", "v_proj", register_fn,
        quantization_method, use_bias, dtype, rank_info);
    o_proj_ = this->bind_child("o_proj", std::make_shared<layers::linear::RowParallelLinear>(runtime, total_num_heads * head_dim_, hidden_size_, quantization_method, use_output_bias, dtype, tp_rank, tp_size, rank_info.comm));

    rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(runtime, model_config, runtime.device());

    float scaling = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(num_attention_heads_, head_dim_, scaling, num_key_value_heads_, layer_idx_,
                                                                          kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_,
                                                                          model_config->get_kv_quant_scheme());

    q_norm_ = this->bind_child("q_norm", std::make_shared<std::remove_reference_t<decltype(*q_norm_ )>>(runtime, head_dim_, rms_norm_eps, dtype));
    k_norm_ = this->bind_child("k_norm", std::make_shared<std::remove_reference_t<decltype(*k_norm_ )>>(runtime, head_dim_, rms_norm_eps, dtype));

    // NONE KV scales remain the original null Parameters; no Device allocation.
        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) runtime.syncStream();
    } catch (...) {
        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) {
            try { runtime.syncStream(); } catch (...) { std::terminate(); }
        }
        throw;
    }
}

infinilm::runtime::Tensor Qwen3Attention::forward(
    const infinilm::runtime::Tensor &positions,
    const infinilm::runtime::Tensor &hidden_states,
    infinilm::global_state::ForwardContext &forward_context) const {
    if (!forward_context.runtime) {
        throw std::invalid_argument("attention forward requires its worker Runtime");
    }
    if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        return forward_static_(positions, hidden_states, forward_context);
    }
    return forward_paged_(positions, hidden_states, forward_context);
}

infinilm::runtime::Tensor Qwen3Attention::forward_static_(const infinilm::runtime::Tensor &position_ids,
                                                   const infinilm::runtime::Tensor &hidden_states,
                                                   infinilm::global_state::ForwardContext &forward_context) const {

    // hidden_states shape: [batch, seq_len, hidden_size]
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    // 1. Project Q, K, V
    auto [q, k, v] = qkv_proj_->forward_split(*forward_context.runtime, hidden_states_mutable);

    q = q_norm_->forward(*forward_context.runtime, q->view({batch_size * seq_len, num_attention_heads_, head_dim_}));
    k = k_norm_->forward(*forward_context.runtime, k->view({batch_size * seq_len, num_key_value_heads_, head_dim_}));

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
        throw std::runtime_error("infinilm::models::qwen3::Qwen3Attention: Unexpected position_ids shape");
    }

    // 4. Apply RoPE to QK
    rotary_emb_->forward(*forward_context.runtime, q_reshaped, pos_ids_for_rope, true);
    rotary_emb_->forward(*forward_context.runtime, k_reshaped, pos_ids_for_rope, true);

    // 6. Attn Backend calculate
    auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped, forward_context);

    // 7. Project output
    auto output = o_proj_->forward(*forward_context.runtime, attn_output);
    return output;
}

infinilm::runtime::Tensor Qwen3Attention::forward_paged_(const infinilm::runtime::Tensor &position_ids,
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
    q_reshaped = q_norm_->forward(*forward_context.runtime, q_reshaped);
    k_reshaped = k_norm_->forward(*forward_context.runtime, k_reshaped);

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
} // namespace infinilm::models::qwen3
