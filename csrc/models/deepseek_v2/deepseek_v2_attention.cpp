#include "deepseek_v2_attention.hpp"
#include "../../execution/tensor_expressions.hpp"
#include "../../weights/weights.hpp"
#include "runtime/runtime.hpp"
#include <exception>

#include "../../layers/attention/attention.hpp"
#include "../../layers/rotary_embedding/rotary_embedding.hpp"
#include "../../utils.hpp"
#include "deepseek_v2_utils.hpp"

#include <stdexcept>

namespace infinilm::models::deepseek_v2 {

DeepseekV2Attention::DeepseekV2Attention(infinilm::runtime::Runtime &runtime,
                                         std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                         size_t layer_idx,
                                         const engine::distributed::RankInfo &rank_info,
                                         backends::AttentionBackend attention_backend) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) {
        throw std::logic_error("DeepseekV2Attention construction requires eager Runtime");
    }
    if (!model_config || rank_info.device != runtime.device() || rank_info.tp_size <= 0
        || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size
        || (rank_info.tp_size > 1 && !rank_info.comm)) {
        throw std::invalid_argument("Invalid explicit Deepseek construction resources");
    }
    if (attention_backend != backends::AttentionBackend::STATIC_ATTN
        && attention_backend != backends::AttentionBackend::PAGED_ATTN
        && attention_backend != backends::AttentionBackend::FLASH_ATTN) {
        throw std::invalid_argument("Unsupported explicit Deepseek attention backend");
    }
    try {
        layer_idx_ = layer_idx;
        hidden_size_ = model_config->get<size_t>("hidden_size");
        qk_nope_head_dim_ = model_config->get<size_t>("qk_nope_head_dim");
        qk_rope_head_dim_ = model_config->get<size_t>("qk_rope_head_dim");
        q_head_dim_ = qk_nope_head_dim_ + qk_rope_head_dim_;
        v_head_dim_ = model_config->get<size_t>("v_head_dim");

        if (v_head_dim_ > q_head_dim_) {
            throw std::invalid_argument("DeepseekV2Attention: v_head_dim exceeds q_head_dim");
        }
        const auto &dtype{model_config->get_dtype()};
        const size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
        const size_t kv_lora_rank = model_config->get<size_t>("kv_lora_rank");
        const bool attention_bias = model_config->get_or<bool>("attention_bias", false);
        const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

        const int tp_rank = rank_info.tp_rank;
        const int tp_size = rank_info.tp_size;
        if ((total_num_heads < static_cast<size_t>(tp_size)) || (total_num_heads % static_cast<size_t>(tp_size) != 0)) {
            throw std::runtime_error("DeepseekV2Attention: num_attention_heads must be divisible by tp_size");
        }
        num_attention_heads_ = total_num_heads / static_cast<size_t>(tp_size);
        attention_backend_ = attention_backend;

        auto quantization_method = model_config->get_quantization_method();
        q_proj_ = this->bind_child("q_proj", std::make_shared<std::remove_reference_t<decltype(*q_proj_)>>(runtime, hidden_size_, total_num_heads * q_head_dim_, quantization_method, false, dtype, tp_rank, tp_size));
        kv_a_proj_with_mqa_ = this->bind_child("kv_a_proj_with_mqa", std::make_shared<std::remove_reference_t<decltype(*kv_a_proj_with_mqa_)>>(runtime, hidden_size_, kv_lora_rank + qk_rope_head_dim_, attention_bias, dtype));
        kv_a_layernorm_ = this->bind_child("kv_a_layernorm", std::make_shared<std::remove_reference_t<decltype(*kv_a_layernorm_)>>(runtime, kv_lora_rank, rms_norm_eps, dtype));
        kv_b_proj_ = this->bind_child("kv_b_proj", std::make_shared<std::remove_reference_t<decltype(*kv_b_proj_)>>(runtime, kv_lora_rank, total_num_heads * (qk_nope_head_dim_ + v_head_dim_), quantization_method, false, dtype, tp_rank, tp_size));
        o_proj_ = this->bind_child("o_proj", std::make_shared<std::remove_reference_t<decltype(*o_proj_)>>(runtime, total_num_heads * v_head_dim_, hidden_size_, quantization_method, attention_bias, dtype, tp_rank, tp_size, rank_info.comm));

        rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(runtime, model_config, runtime.device());
        softmax_scale_ = deepseek_v2_attention_softmax_scale(model_config, static_cast<float>(q_head_dim_));

        infinilm::layers::attention::init_kv_cache_quant_params(runtime, [this](const std::string &n, infinilm::weights::Weight p) { this->bind_weight(n, std::move(p)); }, kv_cache_k_scale_, kv_cache_v_scale_, model_config->get_kv_quant_scheme());
        attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(
            num_attention_heads_, q_head_dim_, softmax_scale_, num_attention_heads_, layer_idx_,
            kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_,
            model_config->get_kv_quant_scheme());
    } catch (...) {
        try {
            runtime.syncStream();
        } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor DeepseekV2Attention::position_ids_for_rope_(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &position_ids) const {
    auto pos_shape = position_ids->shape();
    if (pos_shape.size() == 2) {
        return position_ids->narrow({{0, 0, 1}})->contiguous(runtime)->view({pos_shape[1]});
    }
    if (pos_shape.size() == 1) {
        return position_ids->contiguous(runtime);
    }
    throw std::runtime_error("DeepseekV2Attention: unexpected position_ids shape");
}

infinilm::runtime::Tensor DeepseekV2Attention::trim_value_padding_(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &attn_output) const {
    const auto shape = attn_output->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    return attn_output->view({batch_size, seq_len, num_attention_heads_, q_head_dim_})
        ->narrow({{3, 0, v_head_dim_}})
        ->contiguous(runtime)
        ->view({batch_size, seq_len, num_attention_heads_ * v_head_dim_});
}

infinilm::runtime::Tensor DeepseekV2Attention::forward(const infinilm::runtime::Tensor &positions,
                                                       const infinilm::runtime::Tensor &hidden_states,
                                                       infinilm::global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        return forward_static_(positions, hidden_states, forward_context);
    }
    return forward_paged_(positions, hidden_states, forward_context);
}

infinilm::runtime::Tensor DeepseekV2Attention::forward_static_(const infinilm::runtime::Tensor &position_ids,
                                                               const infinilm::runtime::Tensor &hidden_states,
                                                               infinilm::global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    auto hidden_states_mutable = hidden_states;

    auto q = q_proj_->forward(runtime, hidden_states_mutable)->view({batch_size, seq_len, num_attention_heads_, q_head_dim_});
    auto q_nope = q->narrow({{3, 0, qk_nope_head_dim_}});
    auto q_pe = q->narrow({{3, qk_nope_head_dim_, qk_rope_head_dim_}})->contiguous(runtime);

    auto compressed = kv_a_proj_with_mqa_->forward(runtime, hidden_states_mutable);
    auto compressed_kv = compressed->narrow({{2, 0, kv_a_layernorm_->normalized_shape()}})->contiguous(runtime);
    auto k_pe = compressed->narrow({{2, kv_a_layernorm_->normalized_shape(), qk_rope_head_dim_}})->contiguous(runtime);

    auto kv_norm = kv_a_layernorm_->forward(runtime, compressed_kv);
    auto kv = kv_b_proj_->forward(runtime, kv_norm)->view({batch_size, seq_len, num_attention_heads_, qk_nope_head_dim_ + v_head_dim_});
    auto k_nope = kv->narrow({{3, 0, qk_nope_head_dim_}});
    auto value_states = kv->narrow({{3, qk_nope_head_dim_, v_head_dim_}})->contiguous(runtime);

    const auto pos_shape = position_ids->shape();
    infinilm::runtime::Tensor pos_ids;
    if (pos_shape.size() == 2) {
        ASSERT_EQ(pos_shape[0], batch_size);
        ASSERT_EQ(pos_shape[1], seq_len);
        pos_ids = position_ids->contiguous(runtime);
    } else if (pos_shape.size() == 1) {
        ASSERT_EQ(batch_size, 1);
        ASSERT_EQ(pos_shape[0], seq_len);
        pos_ids = position_ids->contiguous(runtime)->view({1, seq_len});
    } else {
        throw std::runtime_error("DeepseekV2Attention: unexpected position_ids shape");
    }
    q_pe = rotary_emb_->forward(runtime, q_pe, pos_ids, true);
    auto rotary_key_view = k_pe->view({batch_size, seq_len, 1, qk_rope_head_dim_});
    auto k_pe_broadcast = infinilm::execution::broadcast_materialized(runtime, rotary_key_view, {static_cast<int64_t>(batch_size), static_cast<int64_t>(seq_len), static_cast<int64_t>(num_attention_heads_), static_cast<int64_t>(qk_rope_head_dim_)});
    k_pe_broadcast = rotary_emb_->forward(runtime, k_pe_broadcast, pos_ids, true);

    const std::vector<infinilm::runtime::Tensor> query_parts{q_nope, q_pe};
    auto query_shape = query_parts[0]->shape();
    query_shape[3] = query_parts[0]->size(3) + query_parts[1]->size(3);
    auto query_states = infinilm::execution::cat_alloc(runtime, query_parts, 3, query_shape);
    const std::vector<infinilm::runtime::Tensor> key_parts{k_nope, k_pe_broadcast};
    auto key_shape = key_parts[0]->shape();
    key_shape[3] = key_parts[0]->size(3) + key_parts[1]->size(3);
    auto key_states = infinilm::execution::cat_alloc(runtime, key_parts, 3, key_shape);
    const std::vector<int> value_padding{0, static_cast<int>(q_head_dim_ - v_head_dim_)};
    auto value_padded = infinilm::execution::pad_constant_alloc(runtime, value_states, value_padding, 0.0);
    INFINILM_TRACE_OP(Pad, value_padded);

    auto attn_output = attn_->forward(query_states, key_states, value_padded, forward_context);
    auto trimmed_output = trim_value_padding_(runtime, attn_output);
    return o_proj_->forward(runtime, trimmed_output);
}

infinilm::runtime::Tensor DeepseekV2Attention::forward_paged_(const infinilm::runtime::Tensor &position_ids,
                                                              const infinilm::runtime::Tensor &hidden_states,
                                                              infinilm::global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    ASSERT_EQ(batch_size, 1);
    auto hidden_states_mutable = hidden_states;

    auto q = q_proj_->forward(runtime, hidden_states_mutable)->view({seq_len, num_attention_heads_, q_head_dim_});
    auto q_nope = q->narrow({{2, 0, qk_nope_head_dim_}});
    auto q_pe = q->narrow({{2, qk_nope_head_dim_, qk_rope_head_dim_}})->contiguous(runtime);

    auto compressed = kv_a_proj_with_mqa_->forward(runtime, hidden_states_mutable)->view({seq_len, kv_a_layernorm_->normalized_shape() + qk_rope_head_dim_});
    auto compressed_kv = compressed->narrow({{1, 0, kv_a_layernorm_->normalized_shape()}})->contiguous(runtime);
    auto k_pe = compressed->narrow({{1, kv_a_layernorm_->normalized_shape(), qk_rope_head_dim_}})->contiguous(runtime);

    auto kv_norm = kv_a_layernorm_->forward(runtime, compressed_kv);
    auto kv = kv_b_proj_->forward(runtime, kv_norm)->view({seq_len, num_attention_heads_, qk_nope_head_dim_ + v_head_dim_});
    auto k_nope = kv->narrow({{2, 0, qk_nope_head_dim_}});
    auto value_states = kv->narrow({{2, qk_nope_head_dim_, v_head_dim_}})->contiguous(runtime);

    auto pos_ids = position_ids_for_rope_(runtime, position_ids);
    q_pe = rotary_emb_->forward(runtime, q_pe, pos_ids, true);
    auto rotary_key_view = k_pe->view({seq_len, 1, qk_rope_head_dim_});
    auto k_pe_broadcast = infinilm::execution::broadcast_materialized(runtime, rotary_key_view, {static_cast<int64_t>(seq_len), static_cast<int64_t>(num_attention_heads_), static_cast<int64_t>(qk_rope_head_dim_)});
    k_pe_broadcast = rotary_emb_->forward(runtime, k_pe_broadcast, pos_ids, true);

    const std::vector<infinilm::runtime::Tensor> query_parts{q_nope, q_pe};
    auto query_shape = query_parts[0]->shape();
    query_shape[2] = query_parts[0]->size(2) + query_parts[1]->size(2);
    auto query_states = infinilm::execution::cat_alloc(runtime, query_parts, 2, query_shape);
    const std::vector<infinilm::runtime::Tensor> key_parts{k_nope, k_pe_broadcast};
    auto key_shape = key_parts[0]->shape();
    key_shape[2] = key_parts[0]->size(2) + key_parts[1]->size(2);
    auto key_states = infinilm::execution::cat_alloc(runtime, key_parts, 2, key_shape);
    const std::vector<int> value_padding{0, static_cast<int>(q_head_dim_ - v_head_dim_)};
    auto value_padded = infinilm::execution::pad_constant_alloc(runtime, value_states, value_padding, 0.0);
    INFINILM_TRACE_OP(Pad, value_padded);

    auto attn_output = attn_->forward(query_states, key_states, value_padded, forward_context);
    auto trimmed_output = trim_value_padding_(runtime, attn_output);
    return o_proj_->forward(runtime, trimmed_output);
}

} // namespace infinilm::models::deepseek_v2
