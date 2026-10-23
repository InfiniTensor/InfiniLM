#pragma once

#include "../../global_state/global_state.hpp"
#include "../../models/infinilm_model.hpp"

#include "infinicore/context/context.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace infinilm::engine::attn_metadata_utils {

/// Refresh host max_query_len / max_kv_len from length tensors (once per forward).
/// Uses Tensor::to(cpu()) — same safe path as compute_prefill_len — instead of raw
/// memcpyD2H on possibly-narrowed / graph-backed views (hcErrorInvalidValue).
inline void refresh_max_seqlens(infinilm::global_state::AttentionMetadata &meta) {
    meta.max_query_length = 0;
    meta.max_sequence_length = 0;

    if (meta.input_offsets.has_value()) {
        const auto &t = meta.input_offsets.value();
        const size_t n = t->size(0);
        if (n >= 2) {
            auto cpu = t->to(infinicore::Device{infinicore::Device::Type::kCpu});
            const auto *offs = reinterpret_cast<const int32_t *>(cpu->data());
            for (size_t i = 0; i + 1 < n; ++i) {
                meta.max_query_length = std::max(meta.max_query_length, static_cast<size_t>(offs[i + 1] - offs[i]));
            }
        }
    }

    if (meta.total_sequence_lengths.has_value()) {
        const auto &t = meta.total_sequence_lengths.value();
        const size_t n = t->size(0);
        if (n > 0) {
            auto cpu = t->to(infinicore::Device{infinicore::Device::Type::kCpu});
            const auto *totals = reinterpret_cast<const int32_t *>(cpu->data());
            for (size_t i = 0; i < n; ++i) {
                meta.max_sequence_length = std::max(meta.max_sequence_length, static_cast<size_t>(totals[i]));
            }
        }
    }

    // Prefer cu_seqlens diffs when present (authoritative KV spans for FA varlen).
    if (meta.cu_seqlens.has_value()) {
        const auto &t = meta.cu_seqlens.value();
        const size_t n = t->size(0);
        if (n >= 2) {
            auto cpu = t->to(infinicore::Device{infinicore::Device::Type::kCpu});
            const auto *cu = reinterpret_cast<const int32_t *>(cpu->data());
            for (size_t i = 0; i + 1 < n; ++i) {
                meta.max_sequence_length = std::max(meta.max_sequence_length, static_cast<size_t>(cu[i + 1] - cu[i]));
            }
        }
    }

    // Clamp to block-table capacity so FA never indexes past table columns when
    // max_position_embeddings (e.g. 131072) exceeds num_blocks * block_size.
    if (meta.block_tables.has_value() && meta.block_tables.value()->ndim() >= 2) {
        const size_t bt_cols = meta.block_tables.value()->size(1);
        constexpr size_t kMaxPage = 256;
        const size_t table_cap_hi = bt_cols * kMaxPage;
        if (meta.max_sequence_length > table_cap_hi) {
            meta.max_sequence_length = table_cap_hi;
        }
    }

    if (meta.max_query_length == 0 && meta.slot_mapping.has_value()) {
        meta.max_query_length = meta.slot_mapping.value()->shape()[0];
    }
    if (meta.max_sequence_length == 0) {
        meta.max_sequence_length = meta.max_query_length;
    }
    if (meta.max_query_length == 0) {
        meta.max_query_length = meta.max_sequence_length;
    }
}

inline void set_attn_metadata(const InfinilmModel::Input &input) {
    auto &meta = infinilm::global_state::get_forward_context().attn_metadata;
    meta = {
        input.past_sequence_lengths,
        input.total_sequence_lengths,
        input.input_offsets,
        input.cu_seqlens,
        input.block_tables,
        input.slot_mapping,
    };
    refresh_max_seqlens(meta);
}

/// Narrow compiled graph buffers to runtime active shapes (RC-2 analog for replay).
inline void set_attn_metadata_for_varlen_batch(const InfinilmModel::Input &compiled,
                                               const InfinilmModel::Input &runtime) {
    const size_t runtime_n_req = runtime.block_tables.value()->size(0);
    const size_t block_per_req = runtime.block_tables.value()->size(1);
    const size_t offset_len = runtime.input_offsets.value()->size(0);
    const size_t cu_len = runtime.cu_seqlens.value()->size(0);
    const size_t slot_len = runtime.slot_mapping.value()->shape()[0];

    auto &meta = infinilm::global_state::get_forward_context().attn_metadata;
    meta.past_sequence_lengths = compiled.past_sequence_lengths.has_value()
                                     ? std::optional<infinicore::Tensor>(
                                           compiled.past_sequence_lengths.value()->narrow({{0, 0, runtime_n_req}}))
                                     : std::nullopt;
    meta.total_sequence_lengths = compiled.total_sequence_lengths.value()->narrow({{0, 0, runtime_n_req}});
    meta.input_offsets = compiled.input_offsets.value()->narrow({{0, 0, offset_len}});
    meta.cu_seqlens = compiled.cu_seqlens.value()->narrow({{0, 0, cu_len}});
    meta.block_tables = compiled.block_tables.value()->narrow({{0, 0, runtime_n_req}, {1, 0, block_per_req}});
    // paged_caching uses slot_mapping.shape[0] as num_tokens (see paged_caching/info.h).
    meta.slot_mapping = compiled.slot_mapping.value()->narrow({{0, 0, slot_len}});
    // Host max seqlens must come from RUNTIME length tensors: narrowed compiled/graph
    // views fail raw D2H (and may fail to(cpu) on graph storage). FA still uses
    // narrowed compiled meta (CG-safe addresses); runtime lengths were already
    // copy_from'd into those buffers in copy_runtime_into_bucket_. Do NOT swap meta
    // to runtime tensor objects under pad-up — that breaks CG address binding (SIGSEGV).
    //
    // Clamp capacity from *compiled* block_tables (full num_blocks width), not the
    // runtime table's shorter padded width. Large-past MIXED (past≫capture, e.g.
    // ~18k) was clamping max_kv_len to runtime_bt_cols*256 and under-covering the
    // cu_seqlens KV span → FA varlen OOB / silent death (211225 class).
    {
        auto io = meta.input_offsets;
        auto tot = meta.total_sequence_lengths;
        auto cu = meta.cu_seqlens;
        auto bt = meta.block_tables;
        meta.input_offsets = runtime.input_offsets;
        meta.total_sequence_lengths = runtime.total_sequence_lengths;
        meta.cu_seqlens = runtime.cu_seqlens;
        meta.block_tables = bt; // compiled width for table_cap clamp
        refresh_max_seqlens(meta);
        const int mq = meta.max_query_length;
        const int mk = meta.max_sequence_length;
        meta.input_offsets = io;
        meta.total_sequence_lengths = tot;
        meta.cu_seqlens = cu;
        meta.block_tables = bt;
        meta.max_query_length = mq;
        meta.max_sequence_length = mk;
    }
}

/// Decode replay: same narrow helper; decode batches use fixed width == num_requests.
inline void set_attn_metadata_for_decode_batch(const InfinilmModel::Input &compiled,
                                               const InfinilmModel::Input &runtime) {
    set_attn_metadata_for_varlen_batch(compiled, runtime);
}

} // namespace infinilm::engine::attn_metadata_utils
