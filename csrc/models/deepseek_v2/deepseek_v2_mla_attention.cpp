#include "deepseek_v2_mla_attention.hpp"
#include "../../layers/attention/attention_checks.hpp"
#include "../../weights/weights.hpp"
#include "../../execution/submit.hpp"
#include "../../execution/paged_cache_checks.hpp"
#include <base/reshape_and_cache_flash.h>
#include <algorithm>
#include <base/gemm.h>
#include "../../execution/tensor_expressions.hpp"
#include "runtime/runtime.hpp"
#include <exception>
#include <limits>

#include "../../layers/attention/attention.hpp"
#include "../../layers/rotary_embedding/rotary_embedding.hpp"
#include "../../utils.hpp"
#include "deepseek_v2_utils.hpp"
#ifndef INFINICORE_MHA_VARLEN_OMITTED
#include <base/flash_attn_varlen_func.h>
#endif

#include <stdexcept>

namespace infinilm::models::deepseek_v2 {

DeepseekV2MLAAttention::DeepseekV2MLAAttention(infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                               size_t layer_idx,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("DeepseekV2MLAAttention construction requires eager Runtime");
    if (!model_config || rank_info.device != runtime.device() || rank_info.tp_size <= 0
        || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size
        || (rank_info.tp_size > 1 && !rank_info.comm))
        throw std::invalid_argument("Invalid explicit Deepseek construction resources");
    if (attention_backend != backends::AttentionBackend::STATIC_ATTN
        && attention_backend != backends::AttentionBackend::PAGED_ATTN
        && attention_backend != backends::AttentionBackend::FLASH_ATTN)
        throw std::invalid_argument("Unsupported explicit Deepseek attention backend");
    try {
    layer_idx_ = layer_idx;
    hidden_size_ = model_config->get<size_t>("hidden_size");
    qk_nope_head_dim_ = model_config->get<size_t>("qk_nope_head_dim");
    qk_rope_head_dim_ = model_config->get<size_t>("qk_rope_head_dim");
    q_head_dim_ = qk_nope_head_dim_ + qk_rope_head_dim_;
    v_head_dim_ = model_config->get<size_t>("v_head_dim");
    kv_lora_rank_ = model_config->get<size_t>("kv_lora_rank");
    mla_head_dim_ = kv_lora_rank_ + qk_rope_head_dim_;

    if (model_config->get_or<size_t>("q_lora_rank", 0) != 0) {
        throw std::runtime_error("DeepseekV2MLAAttention: q_lora_rank is not supported yet");
    }

    const auto &dtype{model_config->get_dtype()};
    const size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
    const bool attention_bias = model_config->get_or<bool>("attention_bias", false);
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    const int tp_rank = rank_info.tp_rank;
    const int tp_size = rank_info.tp_size;
    if ((total_num_heads < static_cast<size_t>(tp_size)) || (total_num_heads % static_cast<size_t>(tp_size) != 0)) {
        throw std::runtime_error("DeepseekV2MLAAttention: num_attention_heads must be divisible by tp_size");
    }
    num_attention_heads_ = total_num_heads / static_cast<size_t>(tp_size);
        attention_backend_ = attention_backend;

    auto quantization_method = model_config->get_quantization_method();
        q_proj_ = this->bind_child("q_proj", std::make_shared<std::remove_reference_t<decltype(*q_proj_ )>>(runtime, hidden_size_, total_num_heads * q_head_dim_, quantization_method, false, dtype, tp_rank, tp_size));
        kv_a_proj_with_mqa_ = this->bind_child("kv_a_proj_with_mqa", std::make_shared<std::remove_reference_t<decltype(*kv_a_proj_with_mqa_ )>>(runtime, hidden_size_, kv_lora_rank_ + qk_rope_head_dim_, attention_bias, dtype));
        kv_a_layernorm_ = this->bind_child("kv_a_layernorm", std::make_shared<std::remove_reference_t<decltype(*kv_a_layernorm_ )>>(runtime, kv_lora_rank_, rms_norm_eps, dtype));
        kv_b_proj_ = this->bind_child("kv_b_proj", std::make_shared<std::remove_reference_t<decltype(*kv_b_proj_ )>>(runtime, kv_lora_rank_, total_num_heads * (qk_nope_head_dim_ + v_head_dim_), quantization_method, false, dtype, tp_rank, tp_size));
        o_proj_ = this->bind_child("o_proj", std::make_shared<std::remove_reference_t<decltype(*o_proj_ )>>(runtime, total_num_heads * v_head_dim_, hidden_size_, quantization_method, attention_bias, dtype, tp_rank, tp_size, rank_info.comm));

        rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(runtime, model_config, runtime.device());
    softmax_scale_ = deepseek_v2_attention_softmax_scale(model_config, static_cast<float>(q_head_dim_));

        infinilm::layers::attention::init_kv_cache_quant_params(runtime,
            [this](const std::string &n, infinilm::weights::Weight p) { this->bind_weight(n, std::move(p)); },
            kv_cache_k_scale_, kv_cache_v_scale_, model_config->get_kv_quant_scheme());
    latent_attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(
        num_attention_heads_, mla_head_dim_, softmax_scale_, 1, layer_idx_,
            kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_,
            model_config->get_kv_quant_scheme());
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor DeepseekV2MLAAttention::position_ids_for_rope_(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &position_ids) const {
    auto pos_shape = position_ids->shape();
    if (pos_shape.size() == 2) {
        return position_ids->narrow({{0, 0, 1}})->contiguous(runtime)->view({pos_shape[1]});
    }
    if (pos_shape.size() == 1) {
        return position_ids->contiguous(runtime);
    }
    throw std::runtime_error("DeepseekV2MLAAttention: unexpected position_ids shape");
}

infinilm::runtime::Tensor DeepseekV2MLAAttention::kv_b_weight_3d_() const {
    return kv_b_proj_->weight()->view({num_attention_heads_, qk_nope_head_dim_ + v_head_dim_, kv_lora_rank_});
}

infinilm::runtime::Tensor DeepseekV2MLAAttention::project_q_nope_to_latent_(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &q_nope) const {
    const size_t ntokens = q_nope->shape()[0];
    auto q_nope_by_head = q_nope->permute({1, 0, 2})->contiguous(runtime);
    auto w_uk_t = kv_b_weight_3d_()->narrow({{1, 0, qk_nope_head_dim_}})->contiguous(runtime);
    auto q_latent_shape = q_nope_by_head->shape();
    q_latent_shape[q_nope_by_head->ndim() - 1] = w_uk_t->size(q_nope_by_head->ndim() - 1);
    auto q_latent = infinilm::runtime::Tensor::empty(runtime, q_latent_shape, q_nope_by_head->dtype());
    infinilm::execution::submit<infini::ops::Gemm>(
        runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
        infini::ops::Gemm::DefaultConfig(runtime.device().type()), q_nope_by_head, w_uk_t,
        std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{1.0F},
        std::optional<float>{0.0F}, std::optional<int>{}, std::optional<int>{}, q_latent);
    INFINILM_TRACE_OP(Gemm, q_latent);
    return q_latent->permute({1, 0, 2})->contiguous(runtime)->view({ntokens, num_attention_heads_, kv_lora_rank_});
}

infinilm::runtime::Tensor DeepseekV2MLAAttention::project_latent_to_value_(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &attn_output,
                                                                    size_t batch_size,
                                                                    size_t seq_len) const {
    const size_t ntokens = batch_size * seq_len;
    const auto out_shape = attn_output->shape();
    const size_t out_head_dim = out_shape.back();
    infinilm::runtime::Tensor latent;
    if (out_head_dim == kv_lora_rank_) {
        latent = attn_output->view({ntokens, num_attention_heads_, kv_lora_rank_});
    } else {
        latent = attn_output->view({ntokens, num_attention_heads_, mla_head_dim_})
                     ->narrow({{2, 0, kv_lora_rank_}})
                     ->contiguous(runtime);
    }
    auto latent_by_head = latent->permute({1, 0, 2})->contiguous(runtime);
    auto w_uv = kv_b_weight_3d_()
                    ->narrow({{1, qk_nope_head_dim_, v_head_dim_}})
                    ->permute({0, 2, 1})
                    ->contiguous(runtime);
    auto value_shape = latent_by_head->shape();
    value_shape[latent_by_head->ndim() - 1] = w_uv->size(latent_by_head->ndim() - 1);
    auto value = infinilm::runtime::Tensor::empty(runtime, value_shape, latent_by_head->dtype());
    infinilm::execution::submit<infini::ops::Gemm>(
        runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
        infini::ops::Gemm::DefaultConfig(runtime.device().type()), latent_by_head, w_uv,
        std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{1.0F},
        std::optional<float>{0.0F}, std::optional<int>{}, std::optional<int>{}, value);
    INFINILM_TRACE_OP(Gemm, value);
    value = value
                     ->permute({1, 0, 2})
                     ->contiguous(runtime)
                     ->view({batch_size, seq_len, num_attention_heads_ * v_head_dim_});
    return o_proj_->forward(runtime, value);
}

infinilm::runtime::Tensor DeepseekV2MLAAttention::forward(const infinilm::runtime::Tensor &positions,
                                                   const infinilm::runtime::Tensor &hidden_states,
                                                   infinilm::global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        return forward_static_(positions, hidden_states, forward_context);
    }
    return forward_paged_(positions, hidden_states, forward_context);
}

infinilm::runtime::Tensor DeepseekV2MLAAttention::forward_static_(const infinilm::runtime::Tensor &position_ids,
                                                           const infinilm::runtime::Tensor &hidden_states,
                                                   infinilm::global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    const size_t ntokens = batch_size * seq_len;
    auto hidden_states_mutable = hidden_states;

    auto q = q_proj_->forward(runtime, hidden_states_mutable)->view({ntokens, num_attention_heads_, q_head_dim_});
    auto q_nope = q->narrow({{2, 0, qk_nope_head_dim_}})->contiguous(runtime);
    auto q_pe = q->narrow({{2, qk_nope_head_dim_, qk_rope_head_dim_}})->contiguous(runtime);

    auto compressed = kv_a_proj_with_mqa_->forward(runtime, hidden_states_mutable)->view({ntokens, kv_lora_rank_ + qk_rope_head_dim_});
    auto compressed_kv = compressed->narrow({{1, 0, kv_lora_rank_}})->contiguous(runtime);
    auto k_pe = compressed->narrow({{1, kv_lora_rank_, qk_rope_head_dim_}})->contiguous(runtime);

    auto kv_norm = kv_a_layernorm_->forward(runtime, compressed_kv);
    auto pos_shape = position_ids->shape();
    auto pos_ids = pos_shape.size() == 2 ? position_ids->contiguous(runtime)->view({ntokens}) : position_ids_for_rope_(runtime, position_ids);
    q_pe = rotary_emb_->forward(runtime, q_pe, pos_ids, true);
    auto k_pe_rope = rotary_emb_->forward(runtime, k_pe->view({ntokens, 1, qk_rope_head_dim_}), pos_ids, true);

    auto q_latent = project_q_nope_to_latent_(runtime, q_nope);
    const std::vector<infinilm::runtime::Tensor> query_parts{q_latent, q_pe};
    auto query_shape = query_parts[0]->shape();
    query_shape[2] = query_parts[0]->size(2) + query_parts[1]->size(2);
    auto packed_query_states = infinilm::execution::cat_alloc(runtime, query_parts, 2, query_shape);
    auto query_states = packed_query_states->view({batch_size, seq_len, num_attention_heads_, mla_head_dim_});
    const std::vector<infinilm::runtime::Tensor> key_parts{kv_norm->view({ntokens, 1, kv_lora_rank_}), k_pe_rope};
    auto key_shape = key_parts[0]->shape();
    key_shape[2] = key_parts[0]->size(2) + key_parts[1]->size(2);
    auto packed_key_states = infinilm::execution::cat_alloc(runtime, key_parts, 2, key_shape);
    auto key_states = packed_key_states
                          ->view({batch_size, seq_len, 1, mla_head_dim_});
    const std::vector<int> value_padding{0, static_cast<int>(qk_rope_head_dim_)};
    auto value_states = infinilm::execution::pad_constant_alloc(runtime, kv_norm->view({batch_size, seq_len, 1, kv_lora_rank_}), value_padding, 0.0);
    INFINILM_TRACE_OP(Pad, value_states);

    auto attn_output = latent_attn_->forward(query_states, key_states, value_states, forward_context);
    return project_latent_to_value_(runtime, attn_output, batch_size, seq_len);
}

infinilm::runtime::Tensor DeepseekV2MLAAttention::forward_paged_(const infinilm::runtime::Tensor &position_ids,
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
    auto q_nope = q->narrow({{2, 0, qk_nope_head_dim_}})->contiguous(runtime);
    auto q_pe = q->narrow({{2, qk_nope_head_dim_, qk_rope_head_dim_}})->contiguous(runtime);

    auto compressed = kv_a_proj_with_mqa_->forward(runtime, hidden_states_mutable)->view({seq_len, kv_lora_rank_ + qk_rope_head_dim_});
    auto compressed_kv = compressed->narrow({{1, 0, kv_lora_rank_}})->contiguous(runtime);
    auto k_pe = compressed->narrow({{1, kv_lora_rank_, qk_rope_head_dim_}})->contiguous(runtime);

    auto kv_norm = kv_a_layernorm_->forward(runtime, compressed_kv);
    auto pos_ids = position_ids_for_rope_(runtime, position_ids);
    q_pe = rotary_emb_->forward(runtime, q_pe, pos_ids, true);
    auto k_pe_rope = rotary_emb_->forward(runtime, k_pe->view({seq_len, 1, qk_rope_head_dim_}), pos_ids, true);

    auto q_latent = project_q_nope_to_latent_(runtime, q_nope);
    const std::vector<infinilm::runtime::Tensor> query_parts{q_latent, q_pe};
    auto query_shape = query_parts[0]->shape();
    query_shape[2] = query_parts[0]->size(2) + query_parts[1]->size(2);
    auto query_states = infinilm::execution::cat_alloc(runtime, query_parts, 2, query_shape);
    const std::vector<infinilm::runtime::Tensor> key_parts{kv_norm->view({seq_len, 1, kv_lora_rank_}), k_pe_rope};
    auto key_shape = key_parts[0]->shape();
    key_shape[2] = key_parts[0]->size(2) + key_parts[1]->size(2);
    auto key_states = infinilm::execution::cat_alloc(runtime, key_parts, 2, key_shape);
    auto value_states = kv_norm->view({seq_len, 1, kv_lora_rank_});

    auto &attn_metadata = forward_context.attn_metadata;
    auto &kv_cache = forward_context.kv_cache_vec[layer_idx_];

    auto block_tables = attn_metadata.block_tables;
    auto slot_mapping = attn_metadata.slot_mapping;
    auto total_sequence_lengths = attn_metadata.total_sequence_lengths;
    auto input_offsets = attn_metadata.input_offsets;
    auto cu_seqlens = attn_metadata.cu_seqlens;
    ASSERT(block_tables.has_value());
    ASSERT(slot_mapping.has_value());
    ASSERT(total_sequence_lengths.has_value());

    auto k_cache_layer = kv_cache->narrow({{3, 0, mla_head_dim_}});
    auto v_cache_layer = kv_cache->narrow({{3, mla_head_dim_, kv_lora_rank_}});
    const auto &paged_mapping = slot_mapping.value();
    const auto paged_config = infinilm::execution::paged_cache_checks::validate(
        runtime, k_cache_layer, v_cache_layer, key_states, value_states, paged_mapping);
    if (paged_config) {
        const auto paged_tokens = paged_mapping->size(0);
        const infini::ops::Tensor paged_scale{nullptr, {1}, infinilm::runtime::DataType::kFloat32, runtime.device()};
        const std::string paged_mode{"auto"};
        for (infinilm::runtime::Size paged_start = 0; paged_start < paged_tokens;) {
            const auto paged_count = std::min<infinilm::runtime::Size>(32768, paged_tokens - paged_start);
            auto paged_key = key_states->narrow({{0, paged_start, paged_count}});
            auto paged_value = value_states->narrow({{0, paged_start, paged_count}});
            auto paged_slots = paged_mapping->narrow({{0, paged_start, paged_count}});
            infinilm::execution::submit<infini::ops::ReshapeAndCacheFlash>(
                runtime, infinilm::execution::RecordingPolicy::AllowHostReplay, *paged_config,
                paged_key, paged_value, paged_slots, paged_scale, paged_scale, paged_mode, k_cache_layer, v_cache_layer);
            INFINILM_TRACE_OP(PagedCaching, k_cache_layer);
            paged_start += paged_count;
        }
    }

    auto attn_output = infinilm::runtime::Tensor::empty(runtime, {seq_len, num_attention_heads_, kv_lora_rank_}, query_states->dtype());
    const bool is_prefill = (seq_len != total_sequence_lengths.value()->shape()[0]);
    if (is_prefill) {
        ASSERT(input_offsets.has_value());
        ASSERT(cu_seqlens.has_value());
        if (attention_backend_ == ::infinilm::backends::AttentionBackend::FLASH_ATTN) {
            const size_t num_reqs = total_sequence_lengths.value()->shape()[0];
            ASSERT(num_reqs > 0);
            ASSERT_EQ(seq_len % num_reqs, 0);
            const auto max_seqlen = seq_len / num_reqs;
            if (max_seqlen > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
                throw std::invalid_argument("MLA prefill maximum length exceeds int64");
            // K/V are dense prompt tensors. Passing block_tables here makes the
            // varlen backend interpret them as a paged KV cache.
#ifndef INFINICORE_MHA_VARLEN_OMITTED
            infinilm::execution::submit<infini::ops::FlashAttnVarlenFunc>(
                runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
                infini::ops::Config::Automatic(),
                query_states, key_states, value_states, input_offsets.value(), cu_seqlens.value(),
                std::optional<infinilm::runtime::Tensor>{}, std::optional<infinilm::runtime::Tensor>{},
                static_cast<int64_t>(max_seqlen), static_cast<int64_t>(max_seqlen),
                0.0, std::optional<double>{softmax_scale_}, true, std::vector<int64_t>{-1, -1}, 0.0, false, false,
                attn_output, std::optional<infinilm::runtime::Tensor>{}, std::optional<infinilm::runtime::Tensor>{});
#else
            throw std::runtime_error("MhaVarlen is unavailable in the greedy_bootstrap build profile");
#endif
            INFINILM_TRACE_OP(MultiheadAttentionVarlen, attn_output);
        } else {
            ASSERT_EQ(attention_backend_, ::infinilm::backends::AttentionBackend::PAGED_ATTN);
            latent_attn_->prefill_paged(
                runtime,
                attn_output,
                query_states,
                k_cache_layer,
                v_cache_layer,
                block_tables.value(),
                total_sequence_lengths.value(),
                input_offsets.value(),
                softmax_scale_);
        }
    } else {
        ASSERT(query_states->ndim() == 3 && attn_output->shape() == query_states->shape());
        const auto kv_query = query_states->unsqueeze(1);
        const auto kv_output = attn_output->unsqueeze(1);
#ifdef INFINICORE_MHA_KVCACHE_OMITTED
        throw std::runtime_error("MhaKVCache is unavailable in the greedy_bootstrap build profile");
#else
        const auto kv_window = std::vector<int64_t>{-1, -1};
        const auto kv_bound = infinilm::layers::attention::detail::check_kvcache(
            runtime, kv_output, kv_query, k_cache_layer, v_cache_layer, total_sequence_lengths.value(), block_tables.value(),
            std::nullopt, size_t{0}, kv_window);
        infini::ops::FlashAttnWithKvcacheConfig kv_config(kv_bound);
        kv_config.set_implementation_index(infini::ops::Config::kAutomaticImplementation);
        const std::optional<infinilm::runtime::Tensor> kv_none;
        infinilm::execution::submit<infini::ops::FlashAttnWithKvcache>(
            runtime, infinilm::execution::RecordingPolicy::DeviceOnly, kv_config,
            kv_query, k_cache_layer, v_cache_layer, kv_none, kv_none, kv_none, kv_none,
            std::optional<infinilm::runtime::Tensor>{total_sequence_lengths.value()}, kv_none, kv_none,
            std::optional<infinilm::runtime::Tensor>{block_tables.value()}, kv_none,
            std::optional<double>{softmax_scale_}, true, kv_window, double{0.0}, true,
            int64_t{0}, false, kv_output, kv_none);
        INFINILM_TRACE_OP(MhaKVCache, kv_output);
#endif
    }
    return project_latent_to_value_(runtime, attn_output, batch_size, seq_len);
}

} // namespace infinilm::models::deepseek_v2
