#include "kimi_k25_attention.hpp"
#include "../../execution/tensor_expressions.hpp"
#include "../../weights/weights.hpp"
#include "runtime/runtime.hpp"
#include <exception>

#include "../../global_state/global_state.hpp"
#include "../../layers/rotary_embedding/rotary_embedding.hpp"
#include "../deepseek_v2/deepseek_v2_utils.hpp"

#include <cmath>
#include <optional>
#include <stdexcept>

namespace infinilm::models::kimi_k25 {

KimiK25Attention::KimiK25Attention(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) {
        throw std::logic_error("KimiK25Attention construction requires eager Runtime");
    }
    if (!model_config || rank_info.device != runtime.device() || rank_info.tp_size <= 0
        || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size
        || (rank_info.tp_size > 1 && !rank_info.comm)) {
        throw std::invalid_argument("Invalid explicit KimiK25Attention construction resources");
    }
    if (attention_backend != backends::AttentionBackend::STATIC_ATTN
        && attention_backend != backends::AttentionBackend::PAGED_ATTN
        && attention_backend != backends::AttentionBackend::FLASH_ATTN) {
        throw std::invalid_argument("Invalid KimiK25Attention attention backend");
    }
    try {
        layer_idx_ = layer_idx;
        hidden_size_ = model_config->get<size_t>("hidden_size");
        q_lora_rank_ = model_config->get<size_t>("q_lora_rank");
        kv_lora_rank_ = model_config->get<size_t>("kv_lora_rank");
        qk_nope_head_dim_ = model_config->get<size_t>("qk_nope_head_dim");
        qk_rope_head_dim_ = model_config->get<size_t>("qk_rope_head_dim");
        q_head_dim_ = qk_nope_head_dim_ + qk_rope_head_dim_;
        v_head_dim_ = model_config->get<size_t>("v_head_dim");

        const auto &dtype = model_config->get_dtype();
        const size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
        const bool attention_bias = model_config->get_or<bool>("attention_bias", false);
        const double rms_norm_eps = model_config->get<double>("rms_norm_eps");
        const size_t tp_size = static_cast<size_t>(rank_info.tp_size);
        if (total_num_heads < tp_size || total_num_heads % tp_size != 0) {
            throw std::runtime_error("KimiK25Attention: num_attention_heads must be divisible by tp_size");
        }
        num_attention_heads_ = total_num_heads / tp_size;
        attention_backend_ = attention_backend;

        q_a_proj_ = std::make_shared<std::remove_reference_t<decltype(*q_a_proj_)>>(
            runtime, hidden_size_, q_lora_rank_, attention_bias, dtype);
        this->bind_child("q_a_proj", q_a_proj_);
        q_a_layernorm_ = std::make_shared<std::remove_reference_t<decltype(*q_a_layernorm_)>>(
            runtime, q_lora_rank_, rms_norm_eps, dtype);
        this->bind_child("q_a_layernorm", q_a_layernorm_);
        q_b_proj_ = std::make_shared<std::remove_reference_t<decltype(*q_b_proj_)>>(
            runtime, q_lora_rank_, total_num_heads * q_head_dim_, false, dtype, rank_info.tp_rank, rank_info.tp_size);
        this->bind_child("q_b_proj", q_b_proj_);
        kv_a_proj_with_mqa_ = std::make_shared<std::remove_reference_t<decltype(*kv_a_proj_with_mqa_)>>(
            runtime, hidden_size_, kv_lora_rank_ + qk_rope_head_dim_, attention_bias, dtype);
        this->bind_child("kv_a_proj_with_mqa", kv_a_proj_with_mqa_);
        kv_a_layernorm_ = std::make_shared<std::remove_reference_t<decltype(*kv_a_layernorm_)>>(
            runtime, kv_lora_rank_, rms_norm_eps, dtype);
        this->bind_child("kv_a_layernorm", kv_a_layernorm_);
        kv_b_proj_ = std::make_shared<std::remove_reference_t<decltype(*kv_b_proj_)>>(
            runtime, kv_lora_rank_, total_num_heads * (qk_nope_head_dim_ + v_head_dim_), false, dtype, rank_info.tp_rank, rank_info.tp_size);
        this->bind_child("kv_b_proj", kv_b_proj_);
        o_proj_ = std::make_shared<std::remove_reference_t<decltype(*o_proj_)>>(
            runtime, total_num_heads * v_head_dim_, hidden_size_, attention_bias, dtype, rank_info.tp_rank, rank_info.tp_size, rank_info.comm);
        this->bind_child("o_proj", o_proj_);

        rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(runtime, model_config, runtime.device());
        softmax_scale_ = deepseek_v2::deepseek_v2_attention_softmax_scale(
            model_config, static_cast<float>(q_head_dim_));
        infinilm::layers::attention::init_kv_cache_quant_params(
            runtime,
            [this](const std::string &name, infinilm::weights::Weight parameter) {
                this->bind_weight(name, std::move(parameter));
            },
            kv_cache_k_scale_,
            kv_cache_v_scale_,
            model_config->get_kv_quant_scheme());
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

infinilm::runtime::Tensor KimiK25Attention::position_ids_for_rope(const infinilm::runtime::Tensor &position_ids) const {
    const auto shape = position_ids->shape();
    if (shape.size() == 1) {
        return position_ids;
    }
    if (shape.size() == 2) {
        return position_ids->narrow({{0, 0, 1}})->view({shape[1]});
    }
    throw std::runtime_error("KimiK25Attention: unexpected position_ids shape");
}

infinilm::runtime::Tensor KimiK25Attention::trim_value_padding(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &attn_output) const {
    const auto shape = attn_output->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    return attn_output->view({batch_size, seq_len, num_attention_heads_, q_head_dim_})
        ->narrow({{3, 0, v_head_dim_}})
        ->contiguous(runtime)
        ->view({batch_size, seq_len, num_attention_heads_ * v_head_dim_});
}

infinilm::runtime::Tensor KimiK25Attention::forward(const infinilm::runtime::Tensor &positions,
                                                    const infinilm::runtime::Tensor &hidden_states,
                                                    global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    const auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    if (attention_backend_ != backends::AttentionBackend::STATIC_ATTN && batch_size != 1) {
        throw std::runtime_error("KimiK25Attention: paged attention expects flattened batch size 1");
    }

    auto hidden_mut = hidden_states;
    auto q_lora = q_a_proj_->forward(runtime, hidden_mut);
    q_lora = q_a_layernorm_->forward(runtime, q_lora);
    auto q = q_b_proj_->forward(runtime, q_lora);
    auto compressed = kv_a_proj_with_mqa_->forward(runtime, hidden_mut);
    auto compressed_kv = compressed->narrow({{2, 0, kv_lora_rank_}})->contiguous(runtime);
    auto k_pe = compressed->narrow({{2, kv_lora_rank_, qk_rope_head_dim_}})->contiguous(runtime);
    auto kv_norm = kv_a_layernorm_->forward(runtime, compressed_kv);
    auto kv = kv_b_proj_->forward(runtime, kv_norm);
    const auto pos_ids = position_ids_for_rope(positions);

    if (attention_backend_ == backends::AttentionBackend::STATIC_ATTN) {
        q = q->view({batch_size, seq_len, num_attention_heads_, q_head_dim_});
        auto q_nope = q->narrow({{3, 0, qk_nope_head_dim_}});
        auto q_pe = q->narrow({{3, qk_nope_head_dim_, qk_rope_head_dim_}})->contiguous(runtime);
        kv = kv->view({batch_size, seq_len, num_attention_heads_, qk_nope_head_dim_ + v_head_dim_});
        auto k_nope = kv->narrow({{3, 0, qk_nope_head_dim_}});
        auto value = kv->narrow({{3, qk_nope_head_dim_, v_head_dim_}})->contiguous(runtime);

        rotary_emb_->forward(runtime, q_pe, pos_ids, true);
        auto rotary_key_view = k_pe->view({batch_size, seq_len, 1, qk_rope_head_dim_});
        auto k_pe_heads = infinilm::execution::broadcast_materialized(runtime, rotary_key_view, {static_cast<int64_t>(batch_size), static_cast<int64_t>(seq_len), static_cast<int64_t>(num_attention_heads_), static_cast<int64_t>(qk_rope_head_dim_)});
        rotary_emb_->forward(runtime, k_pe_heads, pos_ids, true);
        const std::vector<infinilm::runtime::Tensor> query_parts{q_nope, q_pe};
        auto query_shape = query_parts[0]->shape();
        query_shape[3] = query_parts[0]->size(3) + query_parts[1]->size(3);
        auto query = infinilm::execution::cat_alloc(runtime, query_parts, 3, query_shape);
        const std::vector<infinilm::runtime::Tensor> key_parts{k_nope, k_pe_heads};
        auto key_shape = key_parts[0]->shape();
        key_shape[3] = key_parts[0]->size(3) + key_parts[1]->size(3);
        auto key = infinilm::execution::cat_alloc(runtime, key_parts, 3, key_shape);
        const std::vector<int> value_padding{0, static_cast<int>(q_head_dim_ - v_head_dim_)};
        auto value_padded = infinilm::execution::pad_constant_alloc(runtime, value, value_padding, 0.0);
        INFINILM_TRACE_OP(Pad, value_padded);
        auto output = trim_value_padding(runtime, attn_->forward(query, key, value_padded, forward_context));
        return o_proj_->forward(runtime, output);
    }

    q = q->view({seq_len, num_attention_heads_, q_head_dim_});
    auto q_nope = q->narrow({{2, 0, qk_nope_head_dim_}});
    auto q_pe = q->narrow({{2, qk_nope_head_dim_, qk_rope_head_dim_}})->contiguous(runtime);
    kv = kv->view({seq_len, num_attention_heads_, qk_nope_head_dim_ + v_head_dim_});
    auto k_nope = kv->narrow({{2, 0, qk_nope_head_dim_}});
    auto value = kv->narrow({{2, qk_nope_head_dim_, v_head_dim_}})->contiguous(runtime);

    rotary_emb_->forward(runtime, q_pe, pos_ids, true);
    auto rotary_key_view = k_pe->view({seq_len, 1, qk_rope_head_dim_});
    auto k_pe_heads = infinilm::execution::broadcast_materialized(runtime, rotary_key_view, {static_cast<int64_t>(seq_len), static_cast<int64_t>(num_attention_heads_), static_cast<int64_t>(qk_rope_head_dim_)});
    rotary_emb_->forward(runtime, k_pe_heads, pos_ids, true);
    const std::vector<infinilm::runtime::Tensor> query_parts{q_nope, q_pe};
    auto query_shape = query_parts[0]->shape();
    query_shape[2] = query_parts[0]->size(2) + query_parts[1]->size(2);
    auto query = infinilm::execution::cat_alloc(runtime, query_parts, 2, query_shape);
    const std::vector<infinilm::runtime::Tensor> key_parts{k_nope, k_pe_heads};
    auto key_shape = key_parts[0]->shape();
    key_shape[2] = key_parts[0]->size(2) + key_parts[1]->size(2);
    auto key = infinilm::execution::cat_alloc(runtime, key_parts, 2, key_shape);
    const std::vector<int> value_padding{0, static_cast<int>(q_head_dim_ - v_head_dim_)};
    auto value_padded = infinilm::execution::pad_constant_alloc(runtime, value, value_padding, 0.0);
    INFINILM_TRACE_OP(Pad, value_padded);
    auto output = trim_value_padding(runtime, attn_->forward(query, key, value_padded, forward_context));
    return o_proj_->forward(runtime, output);
}

} // namespace infinilm::models::kimi_k25
