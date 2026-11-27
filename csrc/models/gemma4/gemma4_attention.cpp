#include "gemma4_attention.hpp"

#include "../../execution/paged_cache_checks.hpp"
#include "../../execution/submit.hpp"
#include "../../layers/attention/attention_checks.hpp"
#include "../../utils.hpp"
#include "gemma4_ops.hpp"
#include <algorithm>
#include <base/add.h>
#include <base/copy.h>
#include <base/fill.h>
#include <base/flash_attn_varlen_func.h>
#include <base/flash_attn_with_kvcache.h>
#include <base/gemm.h>
#include <base/reshape_and_cache_flash.h>
#include <base/softmax.h>
#include <base/tril.h>
#include <base/triu.h>
#include <cstring>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace infinilm::models::gemma4 {
namespace {

size_t read_seq_len_meta(infinilm::runtime::Runtime &runtime,
                         const std::optional<size_t> &snapshot,
                         const std::optional<infinilm::runtime::Tensor> &lengths) {
    if (snapshot.has_value()) {
        return snapshot.value();
    }
    if (!lengths.has_value()) {
        throw std::invalid_argument("Gemma4TextAttention: missing sequence length metadata");
    }
    const auto &tensor = lengths.value();
    if (tensor->dtype() != infinilm::runtime::DataType::kInt32 || tensor->numel() == 0) {
        throw std::invalid_argument("Gemma4TextAttention: sequence lengths must be non-empty int32");
    }
    int32_t value = 0;
    if (tensor->device().type() == infinilm::runtime::Device::Type::kCpu) {
        std::memcpy(&value, tensor->data(), sizeof(value));
    } else if (tensor->device() == runtime.device()) {
        runtime.memcpyD2H(&value, tensor->data(), sizeof(value));
        runtime.syncStream();
    } else {
        throw std::invalid_argument("Gemma4TextAttention: sequence lengths belong to another device");
    }
    if (value < 0) {
        throw std::invalid_argument("Gemma4TextAttention: negative sequence length");
    }
    return static_cast<size_t>(value);
}

std::shared_ptr<infinilm::layers::linear::BaseLinear> make_kv_linear(
    infinilm::runtime::Runtime &runtime,
    size_t hidden_size,
    size_t total_kv_heads,
    size_t head_dim,
    const std::shared_ptr<infinilm::quantization::BaseQuantization> &quantization,
    bool bias,
    infinilm::runtime::DataType dtype,
    const engine::distributed::RankInfo &rank_info) {
    const size_t tp = static_cast<size_t>(rank_info.tp_size);
    const size_t out_features = total_kv_heads * head_dim;
    if (tp == 1 || total_kv_heads == 1) {
        return std::make_shared<infinilm::layers::linear::ReplicatedLinear>(
            runtime, hidden_size, out_features, quantization, bias, dtype);
    }
    if (total_kv_heads < tp) {
        // Replicate each KV head across (tp / total_kv_heads) ranks so each rank holds 1 KV head.
        const size_t replicated_out = tp * head_dim;
        return std::make_shared<infinilm::layers::linear::ColumnParallelLinear>(
            runtime, hidden_size, replicated_out, quantization, bias, dtype,
            rank_info.tp_rank, rank_info.tp_size, static_cast<int>(total_kv_heads));
    }
    return std::make_shared<infinilm::layers::linear::ColumnParallelLinear>(
        runtime, hidden_size, out_features, quantization, bias, dtype,
        rank_info.tp_rank, rank_info.tp_size);
}

} // namespace

Gemma4TextAttention::Gemma4TextAttention(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const Gemma4TextSpec &text_spec,
    const Gemma4LayerSpec &layer_spec,
    std::shared_ptr<Gemma4RoPE> rope,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend)
    : device_(runtime.device()),
      construction_owner_(&runtime),
      attention_backend_(attention_backend),
      layer_idx_(layer_spec.layer_idx),
      kv_source_layer_idx_(layer_spec.kv_source_layer_idx),
      is_sliding_(layer_spec.is_sliding),
      sliding_window_(text_spec.sliding_window),
      is_kv_shared_(layer_spec.is_kv_shared),
      use_k_eq_v_(layer_spec.use_k_eq_v),
      use_bidirectional_vision_attention_(text_spec.use_bidirectional_vision_attention),
      hidden_size_(text_spec.hidden_size),
      head_dim_(layer_spec.head_dim),
      rms_norm_eps_(text_spec.rms_norm_eps),
      rope_(std::move(rope)) {
    runtime.assertOwnerThread();
    const size_t tp = static_cast<size_t>(rank_info.tp_size);
    if (text_spec.num_attention_heads % tp != 0) {
        throw std::invalid_argument("Gemma4TextAttention: num_attention_heads must be divisible by tp_size");
    }
    if (layer_spec.num_kv_heads >= tp && (layer_spec.num_kv_heads % tp) != 0) {
        throw std::invalid_argument("Gemma4TextAttention: num_kv_heads must be divisible by tp_size");
    }
    if (layer_spec.num_kv_heads < tp && (tp % layer_spec.num_kv_heads) != 0) {
        throw std::invalid_argument("Gemma4TextAttention: tp_size must be divisible by num_kv_heads");
    }
    num_attention_heads_ = text_spec.num_attention_heads / tp;
    num_key_value_heads_ = (layer_spec.num_kv_heads < tp) ? 1 : (layer_spec.num_kv_heads / tp);

    const auto dtype = model_config->get_dtype();
    const auto quantization = model_config->get_quantization_method();
    const bool bias = text_spec.attention_bias;

    try {
        const size_t total_q_dim = text_spec.num_attention_heads * head_dim_;
        q_proj_ = this->bind_child(
            "q_proj",
            std::make_shared<infinilm::layers::linear::ColumnParallelLinear>(
                runtime, hidden_size_, total_q_dim, quantization, bias, dtype, rank_info.tp_rank, rank_info.tp_size));
        q_norm_ = this->bind_child(
            "q_norm",
            std::make_shared<infinilm::layers::RMSNorm>(runtime, head_dim_, rms_norm_eps_, dtype));

        if (!is_kv_shared_) {
            k_proj_ = this->bind_child(
                "k_proj",
                make_kv_linear(runtime, hidden_size_, layer_spec.num_kv_heads, head_dim_, quantization, bias, dtype, rank_info));
            k_norm_ = this->bind_child(
                "k_norm",
                std::make_shared<infinilm::layers::RMSNorm>(runtime, head_dim_, rms_norm_eps_, dtype));
            if (!use_k_eq_v_) {
                v_proj_ = this->bind_child(
                    "v_proj",
                    make_kv_linear(runtime, hidden_size_, layer_spec.num_kv_heads, head_dim_, quantization, bias, dtype, rank_info));
            }
            v_norm_ones_ = ops::make_filled_param(runtime, {head_dim_}, dtype, 1.0);
        }

        o_proj_ = this->bind_child(
            "o_proj",
            std::make_shared<infinilm::layers::linear::RowParallelLinear>(
                runtime, total_q_dim, hidden_size_, quantization, bias, dtype, rank_info.tp_rank, rank_info.tp_size, rank_info.comm));
    } catch (...) {
        if (device_.type() != infinilm::runtime::Device::Type::kCpu) {
            try {
                runtime.syncStream();
            } catch (...) {
                std::terminate();
            }
        }
        throw;
    }
}

infinilm::runtime::Tensor Gemma4TextAttention::forward(
    const infinilm::runtime::Tensor &positions,
    const infinilm::runtime::Tensor &hidden_states,
    global_state::ForwardContext &forward_context,
    const Gemma4VisionSpans *vision_spans) const {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    if (&runtime != construction_owner_) {
        throw std::invalid_argument("Gemma4TextAttention forward Runtime differs from construction owner");
    }
    if (attention_backend_ == backends::AttentionBackend::STATIC_ATTN) {
        return forward_static_(positions, hidden_states, forward_context, vision_spans);
    }
    return forward_paged_(positions, hidden_states, forward_context);
}

infinilm::runtime::Tensor Gemma4TextAttention::forward_static_(
    const infinilm::runtime::Tensor &positions,
    const infinilm::runtime::Tensor &hidden_states,
    global_state::ForwardContext &forward_context,
    const Gemma4VisionSpans *vision_spans) const {
    auto &runtime = forward_context.require_runtime();
    auto hidden_mut = hidden_states;
    const size_t batch_size = hidden_states->size(0);
    const size_t seq_len = hidden_states->size(1);

    // 1. Q projection -> q_norm -> RoPE (use 3D [tokens, heads, head_dim] + 1D positions)
    auto pos_1d = positions->contiguous(runtime)->view({positions->numel()});
    auto q = q_proj_->forward(runtime, hidden_mut);
    auto q_heads = q->view({batch_size * seq_len, num_attention_heads_, head_dim_});
    q_heads = q_norm_->forward(runtime, q_heads);
    q_heads = rope_->forward(runtime, q_heads, pos_1d, true);
    q_heads = q_heads->view({batch_size, seq_len, num_attention_heads_, head_dim_});

    const auto &attn_metadata = forward_context.attn_metadata;
    auto &kv_cache = forward_context.kv_cache_vec.at(kv_source_layer_idx_);

    const size_t cache_pos = read_seq_len_meta(
        runtime, attn_metadata.first_past_sequence_length, attn_metadata.past_sequence_lengths);
    const size_t total_seq_len = read_seq_len_meta(
        runtime, attn_metadata.first_total_sequence_length, attn_metadata.total_sequence_lengths);

    const auto &prefill = attn_metadata.static_prefill_host;
    auto cache_view = (prefill && batch_size == 1)
                        ? kv_cache->narrow({{1, prefill->physical_row, 1}})
                        : kv_cache;
    auto k_cache_layer = cache_view->narrow({{0, 0, 1}})->squeeze(0); // [bs, n_kv, max_seq, head_dim]
    auto v_cache_layer = cache_view->narrow({{0, 1, 1}})->squeeze(0); // [bs, n_kv, max_seq, head_dim]

    // 2. K, V projection -> k_norm + RoPE, v_norm(unscaled) -> update static KV cache (skipped on KV-shared layers!)
    if (!is_kv_shared_) {
        auto k_raw = k_proj_->forward(runtime, hidden_mut);
        auto v_raw = use_k_eq_v_ ? k_raw : v_proj_->forward(runtime, hidden_mut);

        auto k_heads = k_raw->view({batch_size * seq_len, num_key_value_heads_, head_dim_});
        k_heads = k_norm_->forward(runtime, k_heads);
        k_heads = rope_->forward(runtime, k_heads, pos_1d, true);
        k_heads = k_heads->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

        auto v_heads = v_raw->view({batch_size * seq_len, num_key_value_heads_, head_dim_});
        v_heads = ops::rms_norm(runtime, v_heads, v_norm_ones_, rms_norm_eps_);
        v_heads = v_heads->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

        auto k_perm = k_heads->permute({0, 2, 1, 3}); // [bs, n_kv, seq_len, head_dim]
        auto v_perm = v_heads->permute({0, 2, 1, 3});

        auto k_dst = k_cache_layer->narrow({{2, cache_pos, seq_len}});
        auto v_dst = v_cache_layer->narrow({{2, cache_pos, seq_len}});
        infinilm::execution::submit<infini::ops::Copy>(
            runtime, infini::ops::Copy::DefaultConfig(runtime.device().type()), k_perm, false, k_dst);
        infinilm::execution::submit<infini::ops::Copy>(
            runtime, infini::ops::Copy::DefaultConfig(runtime.device().type()), v_perm, false, v_dst);
    }

    // 3. Slice KV window.
    // For decode (seq_len == 1), slicing K/V to the last sliding_window tokens is exact and avoids
    // computing scores over out-of-window tokens.
    size_t kv_start = 0;
    if (seq_len == 1 && is_sliding_ && total_seq_len > sliding_window_) {
        kv_start = total_seq_len - sliding_window_;
    }
    const size_t kv_len = total_seq_len - kv_start;

    auto k_total = k_cache_layer->narrow({{2, kv_start, kv_len}}); // [bs, n_kv, kv_len, head_dim]
    auto v_total = v_cache_layer->narrow({{2, kv_start, kv_len}}); // [bs, n_kv, kv_len, head_dim]

    // 4. Native GPU Attention: Q @ K^T (scaling = 1.0!) -> causal/sliding mask -> Softmax -> P @ V.
    // Works on CUDA, MetaX C550 (zero FlashAttn dependency, zero CPU fallback), and CPU.
    // Zero-copy views for K/V (matching static_attn.cpp) and zero-copy Q/out fast path on seq_len == 1!
    const size_t ngroup = num_attention_heads_ / num_key_value_heads_;
    auto Q = (seq_len == 1)
               ? q_heads->view({batch_size * num_key_value_heads_, ngroup, head_dim_})
               : q_heads->permute({0, 2, 1, 3})
                     ->contiguous(runtime)
                     ->view({batch_size * num_key_value_heads_, ngroup * seq_len, head_dim_});
    auto K = k_total->view({batch_size * num_key_value_heads_, kv_len, head_dim_});
    auto V = v_total->view({batch_size * num_key_value_heads_, kv_len, head_dim_});
    auto K_t = K->permute({0, 2, 1}); // [bs * n_kv, head_dim, kv_len]

    auto attn_weight = infinilm::runtime::Tensor::empty(
        runtime, {batch_size * num_key_value_heads_, ngroup * seq_len, kv_len}, Q->dtype());

    const auto gemm_device = runtime.device().type();
    const auto selected = infini::ops::Gemm::DefaultConfig(gemm_device);
    infini::ops::GemmConfig gemm_config{infini::ops::GemmConfig::Fp32Precision::DefaultFast};
    gemm_config.set_implementation_index(selected.implementation_index());

    // Gemma 4 attention scaling is 1.0 (because q_norm, k_norm, v_norm normalize Q/K/V).
    infinilm::execution::submit<infini::ops::Gemm>(
        runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
        gemm_config, Q, K_t,
        std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{1.0F},
        std::optional<float>{0.0F}, std::optional<int>{}, std::optional<int>{}, attn_weight);
    INFINILM_TRACE_OP(Gemm, attn_weight);

    auto attn_scores = attn_weight->view({batch_size * num_attention_heads_, seq_len, kv_len});
    if (seq_len > 1) {
        const auto rows = static_cast<int64_t>(seq_len);
        const auto cols = static_cast<int64_t>(kv_len);
        const auto dev = runtime.device().type();
        constexpr auto policy = infinilm::execution::RecordingPolicy::DeviceOnly;
        const double neg_inf = -std::numeric_limits<double>::infinity();

        if (use_bidirectional_vision_attention_ && vision_spans != nullptr && !vision_spans->empty()) {
            // Exact HF create_masks_for_vision_model: causal + bidirectional within each vision block,
            // intersected with sliding window when is_sliding_.
            const int64_t past_len = cols - rows;
            std::vector<int32_t> block_id(kv_len, -1);
            for (size_t b = 0; b < vision_spans->size(); ++b) {
                const auto [start, len] = (*vision_spans)[b];
                const size_t k_start = static_cast<size_t>(past_len) + start;
                const size_t k_end = std::min<size_t>(kv_len, k_start + len);
                for (size_t pos = k_start; pos < k_end; ++pos) {
                    block_id[pos] = static_cast<int32_t>(b);
                }
            }
            const float neg_inf_f = -std::numeric_limits<float>::infinity();
            std::vector<float> mask_host(seq_len * kv_len, neg_inf_f);
            for (size_t i = 0; i < seq_len; ++i) {
                const int64_t q_pos = past_len + static_cast<int64_t>(i);
                const int32_t q_blk = (q_pos >= 0 && static_cast<size_t>(q_pos) < kv_len)
                                        ? block_id[static_cast<size_t>(q_pos)]
                                        : -1;
                for (size_t j = 0; j < kv_len; ++j) {
                    const int64_t k_pos = static_cast<int64_t>(j);
                    const bool causal_ok = (k_pos <= q_pos);
                    const bool bidir_ok = (q_blk >= 0 && block_id[j] == q_blk);
                    bool allowed = causal_ok || bidir_ok;
                    if (allowed && is_sliding_) {
                        const int64_t diff = q_pos - k_pos;
                        if (diff >= static_cast<int64_t>(sliding_window_)
                            || -diff >= static_cast<int64_t>(sliding_window_)) {
                            allowed = false;
                        }
                    }
                    if (allowed) {
                        mask_host[i * kv_len + j] = 0.0F;
                    }
                }
            }
            const auto cpu = infinilm::runtime::Device(infinilm::runtime::Device::Type::kCpu, 0);
            auto mask_f32 = infinilm::runtime::Tensor::empty(runtime, {seq_len, kv_len}, infinilm::runtime::DataType::kFloat32);
            mask_f32->copy_from(runtime, infinilm::runtime::Tensor::from_blob(
                                             mask_host.data(), {seq_len, kv_len}, infinilm::runtime::DataType::kFloat32, cpu));
            auto mask_dev = (attn_scores->dtype() == infinilm::runtime::DataType::kFloat32)
                              ? mask_f32
                              : ops::cast(runtime, mask_f32, attn_scores->dtype());
            infinilm::execution::submit<infini::ops::Add>(
                runtime, policy, infini::ops::Add::DefaultConfig(dev), attn_scores, mask_dev, attn_scores);
        } else {
            // Upper causal mask: forbid j > (cols - rows) + i  <=>  j - i >= cols - rows + 1
            auto upper_mask = infinilm::runtime::Tensor::empty(runtime, {seq_len, kv_len}, attn_scores->dtype());
            infinilm::execution::submit<infini::ops::Fill>(
                runtime, policy, infini::ops::Fill::DefaultConfig(dev), upper_mask, neg_inf, upper_mask);
            infinilm::execution::submit<infini::ops::Triu>(
                runtime, policy, infini::ops::Triu::DefaultConfig(dev), upper_mask, cols - rows + 1, upper_mask);
            infinilm::execution::submit<infini::ops::Tril>(
                runtime, policy, infini::ops::Tril::DefaultConfig(dev), attn_scores, cols - rows, attn_scores);
            infinilm::execution::submit<infini::ops::Add>(
                runtime, policy, infini::ops::Add::DefaultConfig(dev), attn_scores, upper_mask, attn_scores);

            // Lower sliding-window mask: forbid j <= (cols - rows) + i - sliding_window
            if (is_sliding_ && kv_len > sliding_window_) {
                const int64_t lower_diag = cols - rows - static_cast<int64_t>(sliding_window_);
                auto lower_mask = infinilm::runtime::Tensor::empty(runtime, {seq_len, kv_len}, attn_scores->dtype());
                infinilm::execution::submit<infini::ops::Fill>(
                    runtime, policy, infini::ops::Fill::DefaultConfig(dev), lower_mask, neg_inf, lower_mask);
                infinilm::execution::submit<infini::ops::Tril>(
                    runtime, policy, infini::ops::Tril::DefaultConfig(dev), lower_mask, lower_diag, lower_mask);
                infinilm::execution::submit<infini::ops::Triu>(
                    runtime, policy, infini::ops::Triu::DefaultConfig(dev), attn_scores, lower_diag + 1, attn_scores);
                infinilm::execution::submit<infini::ops::Add>(
                    runtime, policy, infini::ops::Add::DefaultConfig(dev), attn_scores, lower_mask, attn_scores);
            }
        }
    }

    infinilm::execution::submit<infini::ops::Softmax>(
        runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
        infini::ops::Softmax::DefaultConfig(runtime.device().type()),
        attn_scores, int64_t{-1}, std::optional<infini::ops::DataType>{}, attn_scores);
    INFINILM_TRACE_OP(CausalSoftmax, attn_scores);

    auto out = infinilm::runtime::Tensor::empty(
        runtime, {batch_size * num_key_value_heads_, ngroup * seq_len, head_dim_}, attn_weight->dtype());
    infinilm::execution::submit<infini::ops::Gemm>(
        runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
        gemm_config, attn_weight, V,
        std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{1.0F},
        std::optional<float>{0.0F}, std::optional<int>{}, std::optional<int>{}, out);
    INFINILM_TRACE_OP(Gemm, out);

    auto attn_out = (seq_len == 1)
                      ? out->view({batch_size, 1, num_attention_heads_ * head_dim_})
                      : out->view({batch_size, num_attention_heads_, seq_len, head_dim_})
                            ->permute({0, 2, 1, 3})
                            ->contiguous(runtime)
                            ->view({batch_size, seq_len, num_attention_heads_ * head_dim_});
    return o_proj_->forward(runtime, attn_out);
}

infinilm::runtime::Tensor Gemma4TextAttention::forward_paged_(
    const infinilm::runtime::Tensor &positions,
    const infinilm::runtime::Tensor &hidden_states,
    global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    auto hidden_mut = hidden_states;
    const size_t seq_len = hidden_states->size(1);

    auto q = q_proj_->forward(runtime, hidden_mut);
    auto q_heads = q->view({seq_len, num_attention_heads_, head_dim_});
    q_heads = q_norm_->forward(runtime, q_heads);
    q_heads = rope_->forward(runtime, q_heads, positions, true);

    const auto &attn_metadata = forward_context.attn_metadata;
    auto &kv_cache = forward_context.kv_cache_vec.at(kv_source_layer_idx_);
    auto k_cache_layer = kv_cache->narrow({{0, 0, 1}})->squeeze(0);
    auto v_cache_layer = kv_cache->narrow({{0, 1, 1}})->squeeze(0);

    if (!is_kv_shared_) {
        auto k_raw = k_proj_->forward(runtime, hidden_mut);
        auto v_raw = use_k_eq_v_ ? k_raw : v_proj_->forward(runtime, hidden_mut);

        auto k_heads = k_raw->view({seq_len, num_key_value_heads_, head_dim_});
        k_heads = k_norm_->forward(runtime, k_heads);
        k_heads = rope_->forward(runtime, k_heads, positions, true);

        auto v_heads = v_raw->view({seq_len, num_key_value_heads_, head_dim_});
        v_heads = ops::rms_norm(runtime, v_heads, v_norm_ones_, rms_norm_eps_);

        const auto &slot_mapping = attn_metadata.slot_mapping.value();
        const auto paged_config = infinilm::execution::paged_cache_checks::validate(
            runtime, k_cache_layer, v_cache_layer, k_heads, v_heads, slot_mapping);
        if (paged_config) {
            const auto paged_tokens = slot_mapping->size(0);
            const infini::ops::Tensor paged_scale{nullptr, {1}, infinilm::runtime::DataType::kFloat32, runtime.device()};
            const std::string paged_mode{"auto"};
            for (infinilm::runtime::Size start = 0; start < paged_tokens;) {
                const auto count = std::min<infinilm::runtime::Size>(32768, paged_tokens - start);
                infinilm::execution::submit<infini::ops::ReshapeAndCacheFlash>(
                    runtime, infinilm::execution::RecordingPolicy::AllowHostReplay, *paged_config,
                    k_heads->narrow({{0, start, count}}),
                    v_heads->narrow({{0, start, count}}),
                    slot_mapping->narrow({{0, start, count}}),
                    paged_scale, paged_scale, paged_mode, k_cache_layer, v_cache_layer);
                start += count;
            }
        }
    }

    const std::vector<int64_t> window_size = is_sliding_
                                               ? std::vector<int64_t>{static_cast<int64_t>(sliding_window_) - 1, 0}
                                               : std::vector<int64_t>{-1, -1};
    const auto &total_seq_lens = attn_metadata.total_sequence_lengths.value();
    const auto &block_tables = attn_metadata.block_tables.value();
    const bool is_prefill = (seq_len != total_seq_lens->size(0));
    auto attn_out = infinilm::runtime::Tensor::empty(runtime, {seq_len, num_attention_heads_, head_dim_}, q_heads->dtype());

    if (is_prefill) {
        const auto max_seqlen_k = block_tables->size(1) * k_cache_layer->size(1);
        infinilm::execution::submit<infini::ops::FlashAttnVarlenFunc>(
            runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
            infini::ops::Config::Automatic(),
            q_heads, k_cache_layer, v_cache_layer,
            attn_metadata.input_offsets.value(), attn_metadata.cu_seqlens.value(),
            std::optional<infinilm::runtime::Tensor>{}, std::optional<infinilm::runtime::Tensor>{block_tables},
            static_cast<int64_t>(seq_len), static_cast<int64_t>(max_seqlen_k),
            0.0, std::optional<double>{1.0}, true, window_size, 0.0, false, false,
            attn_out, std::optional<infinilm::runtime::Tensor>{}, std::optional<infinilm::runtime::Tensor>{});
    } else {
        const auto kv_query = q_heads->unsqueeze(1);
        const auto kv_output = attn_out->unsqueeze(1);
        const auto kv_bound = infinilm::layers::attention::detail::check_kvcache(
            runtime, kv_output, kv_query, k_cache_layer, v_cache_layer, total_seq_lens, block_tables,
            std::nullopt, attn_metadata.logical_kv_upper_bound);
        infini::ops::FlashAttnWithKvcacheConfig kv_config(kv_bound);
        kv_config.set_implementation_index(infini::ops::Config::kAutomaticImplementation);
        const std::optional<infinilm::runtime::Tensor> kv_none;
        infinilm::execution::submit<infini::ops::FlashAttnWithKvcache>(
            runtime, infinilm::execution::RecordingPolicy::DeviceOnly, kv_config,
            kv_query, k_cache_layer, v_cache_layer, kv_none, kv_none, kv_none, kv_none,
            std::optional<infinilm::runtime::Tensor>{total_seq_lens}, kv_none, kv_none,
            std::optional<infinilm::runtime::Tensor>{block_tables}, kv_none,
            std::optional<double>{1.0}, true, window_size, double{0.0}, true,
            int64_t{0}, false, kv_output, kv_none);
    }

    auto out_3d = attn_out->view({1, seq_len, num_attention_heads_ * head_dim_});
    return o_proj_->forward(runtime, out_3d);
}

} // namespace infinilm::models::gemma4
