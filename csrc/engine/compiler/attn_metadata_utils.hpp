#pragma once

#include "../../global_state/global_state.hpp"
#include "../../models/infinilm_model.hpp"

#include "infinicore/context/context.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>

namespace infinilm::engine::attn_metadata_utils {

// #region agent log
inline void dbg_log_d39f05_(const char *hypothesisId,
                            const char *location,
                            const char *message,
                            const std::string &data_json,
                            const char *runId = "paged-bisect") {
    try {
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
        std::ostringstream line;
        line << "{\"sessionId\":\"8b13ee\",\"runId\":\"" << runId << "\",\"hypothesisId\":\""
             << hypothesisId << "\",\"location\":\"" << location << "\",\"message\":\"" << message
             << "\",\"data\":" << data_json << ",\"timestamp\":" << ms << "}\n";
        const std::string s = line.str();
        // Mirror to stderr so serve.log captures evidence if file I/O fails.
        std::fprintf(stderr, "[8b13ee] %s", s.c_str());
        std::fflush(stderr);
        std::ofstream ofs("/opt/offline/infinilm-metax-20260622/.cursor/debug-8b13ee.log",
                          std::ios::app);
        if (ofs) {
            ofs << s;
            ofs.flush();
        }
    } catch (...) {
    }
}
// #endregion

/// Refresh host max_query_len / max_kv_len from length tensors (once per forward).
/// Uses Tensor::to(cpu()) — same safe path as compute_prefill_len — instead of raw
/// memcpyD2H on possibly-narrowed / graph-backed views (hcErrorInvalidValue).
inline void refresh_max_seqlens(infinilm::global_state::AttentionMetadata &meta) {
    meta.max_query_len = 0;
    meta.max_kv_len = 0;

    if (meta.input_offsets.has_value()) {
        const auto &t = meta.input_offsets.value();
        const size_t n = t->size(0);
        // #region agent log
        {
            std::ostringstream dj;
            dj << "{\"tensor\":\"input_offsets\",\"n\":" << n
               << ",\"contiguous\":" << (t->is_contiguous() ? "true" : "false")
               << ",\"device\":\"" << t->device().toString() << "\""
               << ",\"data_null\":" << (t->data() == nullptr ? "true" : "false")
               << ",\"nbytes\":" << (n * sizeof(int32_t)) << "}";
            dbg_log_d39f05_("A,B,C,D,E", "attn_metadata_utils.hpp:refresh:before_offsets",
                            "about_to_to_cpu", dj.str());
        }
        // #endregion
        if (n >= 2) {
            try {
                auto cpu = t->to(infinicore::Device::cpu());
                const auto *offs = reinterpret_cast<const int32_t *>(cpu->data());
                for (size_t i = 0; i + 1 < n; ++i) {
                    meta.max_query_len = std::max(meta.max_query_len, offs[i + 1] - offs[i]);
                }
                // #region agent log
                dbg_log_d39f05_("B", "attn_metadata_utils.hpp:refresh:after_offsets", "to_cpu_ok",
                                "{\"tensor\":\"input_offsets\"}");
                // #endregion
            } catch (const std::exception &ex) {
                // #region agent log
                {
                    std::ostringstream dj;
                    dj << "{\"tensor\":\"input_offsets\",\"what\":\"" << ex.what() << "\"}";
                    dbg_log_d39f05_("A,B,C", "attn_metadata_utils.hpp:refresh:offsets_fail",
                                    "to_cpu_exception", dj.str());
                }
                // #endregion
                throw;
            }
        }
    }

    if (meta.total_sequence_lengths.has_value()) {
        const auto &t = meta.total_sequence_lengths.value();
        const size_t n = t->size(0);
        // #region agent log
        {
            std::ostringstream dj;
            dj << "{\"tensor\":\"total_sequence_lengths\",\"n\":" << n
               << ",\"contiguous\":" << (t->is_contiguous() ? "true" : "false")
               << ",\"device\":\"" << t->device().toString() << "\""
               << ",\"data_null\":" << (t->data() == nullptr ? "true" : "false") << "}";
            dbg_log_d39f05_("A,C,D", "attn_metadata_utils.hpp:refresh:before_totals",
                            "about_to_to_cpu", dj.str());
        }
        // #endregion
        if (n > 0) {
            try {
                auto cpu = t->to(infinicore::Device::cpu());
                const auto *totals = reinterpret_cast<const int32_t *>(cpu->data());
                for (size_t i = 0; i < n; ++i) {
                    meta.max_kv_len = std::max(meta.max_kv_len, totals[i]);
                }
            } catch (const std::exception &ex) {
                // #region agent log
                {
                    std::ostringstream dj;
                    dj << "{\"tensor\":\"total_sequence_lengths\",\"what\":\"" << ex.what() << "\"}";
                    dbg_log_d39f05_("A,C", "attn_metadata_utils.hpp:refresh:totals_fail",
                                    "to_cpu_exception", dj.str());
                }
                // #endregion
                throw;
            }
        }
    }

    // Prefer cu_seqlens diffs when present (authoritative KV spans for FA varlen).
    if (meta.cu_seqlens.has_value()) {
        const auto &t = meta.cu_seqlens.value();
        const size_t n = t->size(0);
        // #region agent log
        {
            std::ostringstream dj;
            dj << "{\"tensor\":\"cu_seqlens\",\"n\":" << n
               << ",\"contiguous\":" << (t->is_contiguous() ? "true" : "false")
               << ",\"device\":\"" << t->device().toString() << "\""
               << ",\"data_null\":" << (t->data() == nullptr ? "true" : "false") << "}";
            dbg_log_d39f05_("A,C,D", "attn_metadata_utils.hpp:refresh:before_cu",
                            "about_to_to_cpu", dj.str());
        }
        // #endregion
        if (n >= 2) {
            try {
                auto cpu = t->to(infinicore::Device::cpu());
                const auto *cu = reinterpret_cast<const int32_t *>(cpu->data());
                for (size_t i = 0; i + 1 < n; ++i) {
                    meta.max_kv_len = std::max(meta.max_kv_len, cu[i + 1] - cu[i]);
                }
            } catch (const std::exception &ex) {
                // #region agent log
                {
                    std::ostringstream dj;
                    dj << "{\"tensor\":\"cu_seqlens\",\"what\":\"" << ex.what() << "\"}";
                    dbg_log_d39f05_("A,C", "attn_metadata_utils.hpp:refresh:cu_fail",
                                    "to_cpu_exception", dj.str());
                }
                // #endregion
                throw;
            }
        }
    }

    // Clamp to block-table capacity so FA never indexes past table columns when
    // max_position_embeddings (e.g. 131072) exceeds num_blocks * block_size.
    if (meta.block_tables.has_value() && meta.block_tables.value()->ndim() >= 2) {
        const size_t bt_cols = meta.block_tables.value()->size(1);
        constexpr size_t kMaxPage = 256;
        const int table_cap_hi = static_cast<int>(bt_cols * kMaxPage);
        if (meta.max_kv_len > table_cap_hi) {
            meta.max_kv_len = table_cap_hi;
        }
    }

    if (meta.max_query_len <= 0 && meta.slot_mapping.has_value()) {
        meta.max_query_len = static_cast<int>(meta.slot_mapping.value()->shape()[0]);
    }
    if (meta.max_kv_len <= 0) {
        meta.max_kv_len = meta.max_query_len;
    }
    if (meta.max_query_len <= 0) {
        meta.max_query_len = meta.max_kv_len;
    }
    // #region agent log
    {
        std::ostringstream dj;
        dj << "{\"max_query_len\":" << meta.max_query_len << ",\"max_kv_len\":" << meta.max_kv_len
           << "}";
        dbg_log_d39f05_("A", "attn_metadata_utils.hpp:refresh:done", "refresh_ok", dj.str());
    }
    // #endregion
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
    // #region agent log
    dbg_log_d39f05_("C", "attn_metadata_utils.hpp:set_attn_metadata", "entry",
                    "{\"path\":\"eager_or_runtime\"}");
    // #endregion
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
    // #region agent log
    {
        std::ostringstream dj;
        dj << "{\"path\":\"varlen_batch_narrow\",\"n_req\":" << runtime_n_req
           << ",\"offset_len\":" << offset_len << ",\"cu_len\":" << cu_len
           << ",\"slot_len\":" << slot_len << "}";
        dbg_log_d39f05_("A,C", "attn_metadata_utils.hpp:set_attn_metadata_for_varlen_batch",
                        "before_refresh_from_runtime", dj.str());
    }
    // #endregion
    // Host max seqlens must come from RUNTIME tensors: narrowed compiled/graph views
    // fail raw D2H (and may fail to(cpu) on graph storage). FA still uses narrowed
    // compiled meta (CG-safe addresses); runtime lengths were already copy_from'd
    // into those buffers in copy_runtime_into_bucket_. Do NOT swap meta to runtime
    // tensor objects under pad-up — that breaks CG address binding (SIGSEGV).
    {
        const size_t compiled_slot =
            compiled.slot_mapping.has_value() ? compiled.slot_mapping.value()->shape()[0] : slot_len;
        const bool pad_up = slot_len < compiled_slot;
        auto io = meta.input_offsets;
        auto tot = meta.total_sequence_lengths;
        auto cu = meta.cu_seqlens;
        auto bt = meta.block_tables;
        meta.input_offsets = runtime.input_offsets;
        meta.total_sequence_lengths = runtime.total_sequence_lengths;
        meta.cu_seqlens = runtime.cu_seqlens;
        meta.block_tables = runtime.block_tables;
        refresh_max_seqlens(meta);
        const int mq = meta.max_query_len;
        const int mk = meta.max_kv_len;
        meta.input_offsets = io;
        meta.total_sequence_lengths = tot;
        meta.cu_seqlens = cu;
        meta.block_tables = bt;
        meta.max_query_len = mq;
        meta.max_kv_len = mk;
        // #region agent log
        {
            std::ostringstream dj;
            dj << "{\"pad_up\":" << (pad_up ? "true" : "false") << ",\"slot_len\":" << slot_len
               << ",\"compiled_slot\":" << compiled_slot << ",\"max_query_len\":" << meta.max_query_len
               << ",\"max_kv_len\":" << meta.max_kv_len << "}";
            dbg_log_d39f05_("A,C", "attn_metadata_utils.hpp:set_attn_metadata_for_varlen_batch",
                            "pad_up_runtime_lens", dj.str());
        }
        // #endregion
    }
}

/// Decode replay: same narrow helper; decode batches use fixed width == num_requests.
inline void set_attn_metadata_for_decode_batch(const InfinilmModel::Input &compiled,
                                               const InfinilmModel::Input &runtime) {
    set_attn_metadata_for_varlen_batch(compiled, runtime);
}

} // namespace infinilm::engine::attn_metadata_utils
