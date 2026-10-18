#include "../../../execution/submit.hpp"
#include "../../../execution/paged_cache_checks.hpp"
#include <base/reshape_and_cache_flash.h>
#include <algorithm>
#include <base/gemm.h>
#include "static_attn.hpp"
#include "../../../utils.hpp"
#include "attention_layer.hpp"
#include <base/flash_attn_varlen_func.h>
#include "runtime/runtime.hpp"
#include "../attention_checks.hpp"
#include <cstring>
#include <stdexcept>
#include <limits>

namespace infinilm::layers::attention::backends {
namespace {

size_t sequence_length_from_metadata(
    infinilm::runtime::Runtime &runtime,
    const std::optional<size_t> &snapshot,
    const std::optional<infinilm::runtime::Tensor> &lengths) {
    if (snapshot.has_value()) {
        return snapshot.value();
    }

    ASSERT(lengths.has_value());
    const auto &tensor = lengths.value();
    if (tensor->dtype() != infinilm::runtime::DataType::kInt32 || tensor->numel() == 0) {
        throw std::invalid_argument("Static Attention requires nonempty int32 sequence lengths");
    }
    int32_t length = 0;
    if (tensor->device().type() == infinilm::runtime::Device::Type::kCpu) {
        std::memcpy(&length, tensor->data(), sizeof(length));
    } else if (tensor->device() == runtime.device()) {
        runtime.memcpyD2H(&length, tensor->data(), sizeof(length));
    } else {
        throw std::invalid_argument("Static Attention sequence lengths belong to another device");
    }
    if (length < 0) {
        throw std::invalid_argument("Static Attention sequence lengths must be nonnegative");
    }
    return static_cast<size_t>(length);
}

} // namespace

StaticAttentionImpl::StaticAttentionImpl(size_t num_heads,
                                         size_t head_size,
                                         float scale,
                                         size_t num_kv_heads,
                                         size_t layer_idx,
                                         infinilm::quantization::KVQuantAlgo kv_quant_scheme)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      layer_idx_(layer_idx),
      head_dim_(head_size), kv_quant_scheme_(kv_quant_scheme) {}

infinilm::runtime::Tensor StaticAttentionImpl::forward(const AttentionLayer &layer,
                                                infinilm::runtime::Tensor &q_rope,
                                                infinilm::runtime::Tensor &k_reshaped,
                                                infinilm::runtime::Tensor &v_reshaped,
                                                infinilm::runtime::Tensor &kv_cache,
                                                const infinilm::global_state::AttentionMetadata &attn_metadata,
                                                infinilm::runtime::Runtime &runtime) const {
    runtime.assertOwnerThread();
    if (runtime.device() != q_rope->device() || runtime.device() != kv_cache->device()) {
        throw std::invalid_argument("Static Attention Runtime and tensor devices differ");
    }

    auto k_scale = layer.get_k_scale();
    auto v_scale = layer.get_v_scale();
    if (infinilm::quantization::KVQuantAlgo::NONE != this->kv_quant_scheme_) {
        infinilm::KVQuantUtils::quantize(
            runtime, k_reshaped, v_reshaped,
            this->kv_quant_scheme_,
            k_scale,
            v_scale);
    }

    auto q_reshaped = q_rope->permute({0, 2, 1, 3});     // [bs, n_q_head, seq_len, head_dim]
    auto k_permuted = k_reshaped->permute({0, 2, 1, 3}); // [bs, n_kv_head, seq_len, head_dim]
    auto v_permuted = v_reshaped->permute({0, 2, 1, 3}); // [bs, n_kv_head, seq_len, head_dim]

    //  Prepare Attn
    auto shape = q_reshaped->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[2];
    size_t value_head_dim = v_reshaped->size(3);

    const auto &prefill = attn_metadata.static_prefill_host;
    const auto &window_size = layer.window_size();
    const bool finite_window = window_size[0] >= 0;
    bool varlen_candidate = finite_window;
    varlen_candidate = varlen_candidate || (prefill && seq_len > 1
        && kv_quant_scheme_ == infinilm::quantization::KVQuantAlgo::NONE
        && (q_rope->dtype() == infinilm::runtime::DataType::kFloat16
            || q_rope->dtype() == infinilm::runtime::DataType::kBFloat16));
    if (finite_window && seq_len > 1 && !prefill)
        throw std::invalid_argument("Static finite window requires validated prefill row/past/total metadata");
    if (prefill) {
        if (seq_len <= 1 || batch_size != prefill->batch_size
            || seq_len != prefill->query_length
            || kv_cache->ndim() != 5 || kv_cache->size(1) != prefill->cache_capacity
            || kv_cache->size(3) != prefill->page_size
            || prefill->physical_row >= prefill->cache_capacity
            || (batch_size != 1 && batch_size != prefill->cache_capacity)
            || prefill->past > prefill->page_size
            || seq_len > prefill->page_size - prefill->past
            || prefill->total != prefill->past + seq_len
            || (batch_size != 1 && prefill->physical_row != 0)
            || !attn_metadata.first_past_sequence_length
            || !attn_metadata.first_total_sequence_length
            || *attn_metadata.first_past_sequence_length != prefill->past
            || *attn_metadata.first_total_sequence_length != prefill->total) {
            throw std::invalid_argument("Static prefill host description does not match query or cache");
        }
    } else if (seq_len > 1 && batch_size == 1 && kv_cache->size(1) > 1) {
        throw std::invalid_argument("Static prefill into a larger cache requires a validated physical row");
    }

    if (seq_len == 1) {
        if (kv_quant_scheme_ == infinilm::quantization::KVQuantAlgo::NONE
            && use_cache_decode_(q_reshaped, k_permuted, v_permuted, kv_cache, attn_metadata, runtime, window_size)) {
            return forward_decode_(q_reshaped, k_permuted, v_permuted, kv_cache, attn_metadata, runtime, window_size);
        }
        if (finite_window)
            throw std::invalid_argument("Static finite window decode has no supported cache provider; global fallback is forbidden");
        const auto &host = attn_metadata.static_decode_host;
        if (batch_size != kv_cache->size(1)
            || (host && (!host->identity_rows || !host->uniform_lengths))) {
            throw std::invalid_argument(
                "Static generic decode cannot ignore physical rows or nonuniform lengths");
        }
    }
    if (runtime.isGraphRecording()) {
        throw std::runtime_error("Static Attention has no prepared cache decode path for recording");
    }

    const size_t cache_pos = sequence_length_from_metadata(
        runtime,
        attn_metadata.first_past_sequence_length,
        attn_metadata.past_sequence_lengths);
    const size_t total_seq_len = sequence_length_from_metadata(
        runtime,
        attn_metadata.first_total_sequence_length,
        attn_metadata.total_sequence_lengths);
    if (varlen_candidate && (batch_size == 0 || total_seq_len == 0
        || seq_len > static_cast<size_t>(std::numeric_limits<int32_t>::max()) / batch_size
        || total_seq_len > static_cast<size_t>(std::numeric_limits<int32_t>::max()) / batch_size))
        throw std::invalid_argument("Static finite window cumulative lengths exceed int32");

    // update static kv cache
    // k_total:  [bs, n_kv_head, max_seq_len, head_dim]
    // v_total : [bs, n_kv_head, max_seq_len, head_dim]
    auto cache_for_update = prefill && batch_size == 1
        ? kv_cache->narrow({{1, prefill->physical_row, 1}}) : kv_cache;
    auto [k_total, v_total] = do_kv_cache_update(
        layer, k_permuted, v_permuted, cache_for_update, cache_pos, runtime);

        if (infinilm::quantization::KVQuantAlgo::NONE != this->kv_quant_scheme_) {
            infinilm::KVQuantUtils::dequantize(
            runtime, k_total, v_total,
                this->kv_quant_scheme_,
                k_scale,
                v_scale,
                q_reshaped);
        }

    k_total = k_total->narrow({{2, 0, total_seq_len}}); // [bs, n_kv_head, total_seq_len, head_dim]
    v_total = v_total->narrow({{2, 0, total_seq_len}}); // [bs, n_kv_head, total_seq_len, head_dim]

    if (varlen_candidate) {
        // Flat, nonpaged Varlen: complete prefixes remain stored; only visibility changes.
        // GPU temporaries use the Runtime's existing same-stream allocator contract.
        // Only missing CPU offsets require synchronous H2D before host owner release.
        infinilm::runtime::Tensor Q, K, V, out;
        auto cu_q = attn_metadata.input_offsets;
        auto cu_k = attn_metadata.cu_seqlens;
        std::vector<int32_t> q_offsets, k_offsets;
        Q = q_rope->contiguous(runtime);
        Q = Q->view({batch_size * seq_len, num_heads_, head_dim_});
        K = k_total->permute({0, 2, 1, 3})->contiguous(runtime);
        K = K->view({batch_size * total_seq_len, num_kv_heads_, head_dim_});
        V = v_total->permute({0, 2, 1, 3})->contiguous(runtime);
        V = V->view({batch_size * total_seq_len, num_kv_heads_, value_head_dim});
        if (!cu_q) {
            q_offsets.resize(batch_size + 1);
            for (size_t i = 0; i <= batch_size; ++i)
                q_offsets[i] = static_cast<int32_t>(i * seq_len);
            cu_q = infinilm::runtime::Tensor::empty(runtime, {batch_size + 1}, infinilm::runtime::DataType::kInt32);
            runtime.memcpyH2D((*cu_q)->data(), q_offsets.data(), q_offsets.size() * sizeof(int32_t), false);
        }
        if (!cu_k) {
            k_offsets.resize(batch_size + 1);
            for (size_t i = 0; i <= batch_size; ++i)
                k_offsets[i] = static_cast<int32_t>(i * total_seq_len);
            cu_k = infinilm::runtime::Tensor::empty(runtime, {batch_size + 1}, infinilm::runtime::DataType::kInt32);
            runtime.memcpyH2D((*cu_k)->data(), k_offsets.data(), k_offsets.size() * sizeof(int32_t), false);
        }
        auto out_shape = Q->shape();
        out_shape.back() = value_head_dim;
        out = infinilm::runtime::Tensor::empty(runtime, out_shape, Q->dtype());
        const auto config = infini::ops::Config::Automatic();
        auto arguments = std::make_tuple(
            Q, K, V, *cu_q, *cu_k,
            std::optional<infinilm::runtime::Tensor>{}, std::optional<infinilm::runtime::Tensor>{},
            static_cast<int64_t>(seq_len), static_cast<int64_t>(total_seq_len),
            0.0, std::optional<double>{scale_}, true, window_size, 0.0, false, false,
            out, std::optional<infinilm::runtime::Tensor>{}, std::optional<infinilm::runtime::Tensor>{});
        const bool supported = finite_window || std::apply([&](const auto &...args) {
            return infini::ops::FlashAttnVarlenFunc::CanMake(
                config, infinilm::execution::detail::view(args)...);
        }, arguments);
        if (supported) {
            std::apply([&](const auto &...args) {
                infinilm::execution::submit<infini::ops::FlashAttnVarlenFunc>(
                    runtime, infinilm::execution::RecordingPolicy::DeviceOnly, config, args...);
            }, arguments);
            INFINILM_TRACE_OP(MultiheadAttentionVarlen, out);
            return out->view({batch_size, seq_len, num_heads_ * value_head_dim});
        }
    }

    // Compute attention.
        size_t ngroup = num_heads_ / num_kv_heads_;
    auto Q = q_reshaped->contiguous(runtime)->view({batch_size * num_kv_heads_, ngroup * seq_len, head_dim_});
        auto K = k_total->view({batch_size * num_kv_heads_, total_seq_len, head_dim_});
        auto V = v_total->view({batch_size * num_kv_heads_, total_seq_len, value_head_dim});

    auto K_transposed = K->permute({0, 2, 1}); // [bs * n_kv_head, head_dim, total_seq_len]

    // Keep the legacy default on providers that have not qualified full-FP32
    // GEMM; an explicit FullFloat32 caller never receives this fallback.
    const auto gemm_device = runtime.device().type();
    const auto selected = infini::ops::Gemm::DefaultConfig(gemm_device);
    const bool full_fp32 = q_reshaped->dtype() == infinilm::runtime::DataType::kFloat32
        && !infini::ops::Gemm::active_implementation_indices(gemm_device).empty()
        && infini::ops::GemmConfig::SupportsFp32Precision(gemm_device, selected.implementation_index(),
            infini::ops::GemmConfig::Fp32Precision::FullFp32);
    infini::ops::GemmConfig gemm_config{full_fp32
        ? infini::ops::GemmConfig::Fp32Precision::FullFp32
        : infini::ops::GemmConfig::Fp32Precision::DefaultFast};
    gemm_config.set_implementation_index(selected.implementation_index());
    auto attn_weight_shape = Q->shape();
    attn_weight_shape[Q->ndim() - 1] = K_transposed->size(Q->ndim() - 1);
    auto attn_weight = infinilm::runtime::Tensor::empty(runtime, attn_weight_shape, Q->dtype());
    infinilm::execution::submit<infini::ops::Gemm>(
        runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
        gemm_config, Q, K_transposed,
        std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{scale_},
        std::optional<float>{0.0F}, std::optional<int>{}, std::optional<int>{}, attn_weight);
    INFINILM_TRACE_OP(Gemm, attn_weight); // [bs * n_kv_head, ng * seq_len, total_seq_len]

        auto attn_weight_softmax = attn_weight->view({batch_size * num_heads_, seq_len, total_seq_len});
    {
        if (seq_len > total_seq_len)
            throw std::invalid_argument("causal_softmax has unsupported shape or dtype");
        std::optional<infinilm::runtime::Tensor> causal_mask;
        if (seq_len != 1)
            causal_mask = infinilm::runtime::Tensor::empty(
                runtime, {seq_len, total_seq_len}, attn_weight_softmax->dtype());
        const auto causal_device = runtime.device().type();
        constexpr auto causal_policy = infinilm::execution::RecordingPolicy::DeviceOnly;
        if (causal_mask) {
            const auto &mask = *causal_mask;
            const auto rows = static_cast<int64_t>(seq_len);
            const auto columns = static_cast<int64_t>(total_seq_len);
            infinilm::execution::submit<infini::ops::Fill>(
                runtime, causal_policy, infini::ops::Fill::DefaultConfig(causal_device),
                mask, -std::numeric_limits<double>::infinity(), mask);
            infinilm::execution::submit<infini::ops::Triu>(
                runtime, causal_policy, infini::ops::Triu::DefaultConfig(causal_device),
                mask, columns - rows + 1, mask);
            infinilm::execution::submit<infini::ops::Tril>(
                runtime, causal_policy, infini::ops::Tril::DefaultConfig(causal_device),
                attn_weight_softmax, columns - rows, attn_weight_softmax);
            infinilm::execution::submit<infini::ops::Add>(
                runtime, causal_policy, infini::ops::Add::DefaultConfig(causal_device),
                attn_weight_softmax, mask, attn_weight_softmax);
        }
        infinilm::execution::submit<infini::ops::Softmax>(
            runtime, causal_policy, infini::ops::Softmax::DefaultConfig(causal_device),
            attn_weight_softmax, int64_t{-1}, std::optional<infini::ops::DataType>{},
            attn_weight_softmax);
        INFINILM_TRACE_OP(CausalSoftmax, attn_weight_softmax);
    }

    auto out_shape_1 = attn_weight->shape();
    out_shape_1[attn_weight->ndim() - 1] = V->size(attn_weight->ndim() - 1);
    auto out = infinilm::runtime::Tensor::empty(runtime, out_shape_1, attn_weight->dtype());
    infinilm::execution::submit<infini::ops::Gemm>(
        runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
        gemm_config, attn_weight, V,
        std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{1.0F},
        std::optional<float>{0.0F}, std::optional<int>{}, std::optional<int>{}, out);
    INFINILM_TRACE_OP(Gemm, out); // [bs * n_kv_head, ng * seq_len, value_head_dim]

    return out->view({batch_size, num_heads_, seq_len, value_head_dim})
                          ->permute({0, 2, 1, 3})
        ->contiguous(runtime)
        ->view({batch_size, seq_len, num_heads_ * value_head_dim}); // [bs, seq_len, n_q_head * value_head_dim]
    }

bool StaticAttentionImpl::use_cache_decode_(
    const infinilm::runtime::Tensor &query, const infinilm::runtime::Tensor &key,
    const infinilm::runtime::Tensor &value, const infinilm::runtime::Tensor &kv_cache,
    const infinilm::global_state::AttentionMetadata &metadata,
    infinilm::runtime::Runtime &runtime, const std::vector<int64_t> &window_size) const {
    if (!metadata.block_tables || !metadata.slot_mapping || !metadata.total_sequence_lengths) {
        return false;
    }
    const std::array<const infinilm::runtime::Tensor *, 7> tensors{
        &query, &key, &value, &kv_cache, &*metadata.total_sequence_lengths,
        &*metadata.block_tables, &*metadata.slot_mapping};
    for (const auto *tensor : tensors) {
        if (!*tensor || (*tensor)->device() != runtime.device()) {
            throw std::invalid_argument("Static decode inputs and metadata must belong to its Runtime");
        }
    }
    const auto &lengths = *metadata.total_sequence_lengths;
    const auto &blocks = *metadata.block_tables;
    const auto &slots = *metadata.slot_mapping;
    if (lengths->dtype() != infinilm::runtime::DataType::kInt32 || lengths->ndim() != 1
        || lengths->size(0) != query->size(0) || !lengths->is_contiguous()
        || blocks->dtype() != infinilm::runtime::DataType::kInt32 || blocks->ndim() != 2
        || blocks->size(0) != query->size(0) || blocks->size(1) != 1 || !blocks->is_contiguous()
        || slots->dtype() != infinilm::runtime::DataType::kInt64 || slots->ndim() != 1
        || slots->size(0) != query->size(0) || !slots->is_contiguous()) {
        throw std::invalid_argument("Invalid static decode metadata layout");
    }
    bool matches = decode_choice_ && decode_choice_->runtime == &runtime
        && decode_choice_->window_size == window_size;
    for (size_t i = 0; matches && i < tensors.size(); ++i) {
        const auto &tensor = *tensors[i];
        const auto &layout = decode_choice_->layouts[i];
        matches = layout.shape == tensor->shape() && layout.strides == tensor->strides()
            && layout.dtype == tensor->dtype();
    }
    if (matches) {
        return decode_choice_->fused;
    }
    if (runtime.isGraphRecording()) {
        throw std::logic_error("Static Attention decode layout must be prepared before recording");
    }
    const auto layout = [](const infinilm::runtime::Tensor &t) {
        return infinilm::runtime::TensorMetaData{t->shape(), t->strides(), t->dtype()};
    };
    DecodeChoice choice{{layout(query), layout(key), layout(value), layout(kv_cache),
                         layout(lengths), layout(blocks), layout(slots)},
                        &runtime, window_size, false};
    auto k_cache = kv_cache->narrow({{0, 0, 1}})->squeeze(0)->permute({0, 2, 1, 3});
    auto v_cache = kv_cache->narrow({{0, 1, 1}})->squeeze(0)->permute({0, 2, 1, 3});
    auto k = key->permute({0, 2, 1, 3})->squeeze(1);
    auto v = value->permute({0, 2, 1, 3})->squeeze(1);
    infinilm::runtime::Tensor q, out;
    // Query both halves before writing cache. A constructor, allocation or
    // execution failure is never retried with another algorithm.
    q = query->contiguous(runtime);
    q = q->view({query->size(0), 1, num_heads_, head_dim_});
    out = infinilm::runtime::Tensor::empty(runtime, q->shape(), q->dtype());
    bool attention_supported = false;
    const auto kv_bound = window_size[0] >= 0 ? metadata.logical_kv_upper_bound : 0;
    if (kv_bound <= static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        infini::ops::FlashAttnWithKvcacheConfig kv_config(static_cast<int64_t>(kv_bound));
        kv_config.set_implementation_index(infini::ops::Config::kAutomaticImplementation);
        const std::optional<infini::ops::Tensor> kv_none;
        attention_supported = infini::ops::FlashAttnWithKvcache::CanMake(
            kv_config, infinilm::execution::detail::view(q),
            infinilm::execution::detail::view(k_cache), infinilm::execution::detail::view(v_cache),
            kv_none, kv_none, kv_none, kv_none,
            std::optional<infini::ops::Tensor>{infinilm::execution::detail::view(lengths)},
            kv_none, kv_none,
            std::optional<infini::ops::Tensor>{infinilm::execution::detail::view(blocks)},
            kv_none, std::optional<double>{scale_}, true, window_size, double{0.0}, true,
            int64_t{0}, false, infinilm::execution::detail::view(out), kv_none);
    }
    choice.fused = attention_supported
        && [&]() {
            if (infini::ops::ReshapeAndCacheFlash::active_implementation_indices(runtime.device().type()).empty()) return false;
            const auto config = infini::ops::ReshapeAndCacheFlash::DefaultConfig(runtime.device().type());
            const infini::ops::Tensor unused{nullptr, {1}, infinilm::runtime::DataType::kFloat32, runtime.device()};
            const std::string mode{"auto"};
            return infini::ops::ReshapeAndCacheFlash::CanMake(
                config, infinilm::execution::detail::view(k), infinilm::execution::detail::view(v),
                infinilm::execution::detail::view(*metadata.slot_mapping), unused, unused, mode,
                infinilm::execution::detail::view(k_cache), infinilm::execution::detail::view(v_cache));
        }();
    decode_choice_ = std::move(choice);
    return decode_choice_->fused;
}

infinilm::runtime::Tensor StaticAttentionImpl::forward_decode_(
    const infinilm::runtime::Tensor &query,
    const infinilm::runtime::Tensor &key,
    const infinilm::runtime::Tensor &value,
    infinilm::runtime::Tensor &kv_cache,
    const infinilm::global_state::AttentionMetadata &attn_metadata,
    infinilm::runtime::Runtime &runtime, const std::vector<int64_t> &window_size) const {
    ASSERT_EQ(query->size(2), 1);
    ASSERT(attn_metadata.block_tables.has_value());
    ASSERT(attn_metadata.past_sequence_lengths.has_value());
    ASSERT(attn_metadata.total_sequence_lengths.has_value());
    ASSERT(attn_metadata.slot_mapping.has_value());

    auto k_cache = kv_cache->narrow({{0, 0, 1}})->squeeze(0)->permute({0, 2, 1, 3});
    auto v_cache = kv_cache->narrow({{0, 1, 1}})->squeeze(0)->permute({0, 2, 1, 3});
    auto key_decode = key->permute({0, 2, 1, 3})->squeeze(1);
    auto value_decode = value->permute({0, 2, 1, 3})->squeeze(1);
    infinilm::runtime::Tensor q_decode, output;
    const auto &paged_mapping = attn_metadata.slot_mapping.value();
    const auto paged_config = infinilm::execution::paged_cache_checks::validate(
        runtime, k_cache, v_cache, key_decode, value_decode, paged_mapping);
    if (paged_config) {
        const auto paged_tokens = paged_mapping->size(0);
        const infini::ops::Tensor paged_scale{nullptr, {1}, infinilm::runtime::DataType::kFloat32, runtime.device()};
        const std::string paged_mode{"auto"};
        for (infinilm::runtime::Size paged_start = 0; paged_start < paged_tokens;) {
            const auto paged_count = std::min<infinilm::runtime::Size>(32768, paged_tokens - paged_start);
            auto paged_key = key_decode->narrow({{0, paged_start, paged_count}});
            auto paged_value = value_decode->narrow({{0, paged_start, paged_count}});
            auto paged_slots = paged_mapping->narrow({{0, paged_start, paged_count}});
            infinilm::execution::submit<infini::ops::ReshapeAndCacheFlash>(
                runtime, infinilm::execution::RecordingPolicy::AllowHostReplay, *paged_config,
                paged_key, paged_value, paged_slots, paged_scale, paged_scale, paged_mode, k_cache, v_cache);
            INFINILM_TRACE_OP(PagedCaching, k_cache);
            paged_start += paged_count;
        }
    }

    q_decode = query->contiguous(runtime);
    q_decode = q_decode->view({query->size(0), query->size(1), query->size(3)});
    output = infinilm::runtime::Tensor::empty(runtime, q_decode->shape(), q_decode->dtype());
    ASSERT(q_decode->ndim() == 3 && output->shape() == q_decode->shape());
    const auto kv_query = q_decode->unsqueeze(1);
    const auto kv_output = output->unsqueeze(1);
    const auto kv_window = window_size;
    const auto kv_bound = infinilm::layers::attention::detail::check_kvcache(
        runtime, kv_output, kv_query, k_cache, v_cache, attn_metadata.total_sequence_lengths.value(), attn_metadata.block_tables.value(),
        std::nullopt, window_size[0] >= 0 ? attn_metadata.logical_kv_upper_bound : 0);
    infini::ops::FlashAttnWithKvcacheConfig kv_config(kv_bound);
    kv_config.set_implementation_index(infini::ops::Config::kAutomaticImplementation);
    const std::optional<infinilm::runtime::Tensor> kv_none;
    infinilm::execution::submit<infini::ops::FlashAttnWithKvcache>(
        runtime, infinilm::execution::RecordingPolicy::DeviceOnly, kv_config,
        kv_query, k_cache, v_cache, kv_none, kv_none, kv_none, kv_none,
        std::optional<infinilm::runtime::Tensor>{attn_metadata.total_sequence_lengths.value()}, kv_none, kv_none,
        std::optional<infinilm::runtime::Tensor>{attn_metadata.block_tables.value()}, kv_none,
        std::optional<double>{scale_}, true, kv_window, double{0.0}, true,
        int64_t{0}, false, kv_output, kv_none);
    INFINILM_TRACE_OP(MhaKVCache, kv_output);

    return output->view({query->size(0), 1, num_heads_ * head_dim_});
}

std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor> StaticAttentionImpl::do_kv_cache_update(const AttentionLayer &layer,
                                                                                           const infinilm::runtime::Tensor key,
                                                                                           const infinilm::runtime::Tensor value,
                                                                                           infinilm::runtime::Tensor &kv_cache,
                                                                                           size_t cache_pos,
                                                                                           infinilm::runtime::Runtime &runtime) const {

    auto batch_size = key->size(0);
    auto update_len = key->size(2);
    auto k_cache_layer = kv_cache->narrow({{0, 0, 1}})->squeeze(0);
    auto v_cache_layer = kv_cache->narrow({{0, 1, 1}})->squeeze(0);
    v_cache_layer = v_cache_layer->narrow({{3, 0, value->size(3)}});

    size_t max_batch_size = k_cache_layer->size(0);
    size_t max_seq_len = k_cache_layer->size(2);
    ASSERT_EQ(batch_size, max_batch_size);

    auto result_len = cache_pos + update_len;
    ASSERT(result_len <= max_seq_len);

    auto k_cache_update = k_cache_layer->narrow({{2, cache_pos, update_len}});
    auto v_cache_update = v_cache_layer->narrow({{2, cache_pos, update_len}});

    infinilm::execution::submit<infini::ops::Copy>(
        runtime, infini::ops::Copy::DefaultConfig(runtime.device().type()),
        key, false, k_cache_update);
    infinilm::execution::submit<infini::ops::Copy>(
        runtime, infini::ops::Copy::DefaultConfig(runtime.device().type()),
        value, false, v_cache_update);

    return {k_cache_layer, v_cache_layer};
}

} // namespace infinilm::layers::attention::backends
