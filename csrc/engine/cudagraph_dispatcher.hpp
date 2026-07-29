#pragma once

/// vLLM-shaped CUDA-graph runtime dispatch (thin InfiniLM mirror).
///
/// Selects FULL / PIECEWISE / NONE from a BatchDescriptor derived from input
/// shape (not scheduler PREFILL/DECODE/MIXED). Under ``full_and_piecewise``:
///   - uniform decode batches in ``INFINI_DECODE_CG_BATCHES`` → FULL
///     (exact hit, or pad-up to next FULL key unless ``INFINI_DECODE_CG_PAD_UP=0``)
///   - ragged / mixed / multi-req prefill (``!uniform_decode``): pad-up
///     ``num_tokens`` to the next ``INFINI_NATIVE_CG_CAPTURE_BUCKETS`` entry
///     (vLLM ``bs_to_padded_graph_size`` / ``mixed_mode=PIECEWISE``) → PIECEWISE
///     with ``key.num_tokens = padded``; eager only when
///     ``num_tokens > max_capture_size``
///   - else → NONE (eager)
/// ``eager`` policy → always NONE.

#include "compiled_prefill_flags.hpp"
#include "compiler/piecewise_bucket_policy.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace infinilm::engine {

enum class CudaGraphRuntimeMode {
    None = 0,
    Piecewise = 1,
    Full = 2,
};

inline const char *cudagraph_runtime_mode_cstr(CudaGraphRuntimeMode mode) {
    switch (mode) {
    case CudaGraphRuntimeMode::Full:
        return "FULL";
    case CudaGraphRuntimeMode::Piecewise:
        return "PIECEWISE";
    case CudaGraphRuntimeMode::None:
    default:
        return "NONE";
    }
}

struct BatchDescriptor {
    size_t num_tokens{0};
    size_t num_reqs{0};
    /// True when block_tables.batch == input_ids.width and every row schedules
    /// exactly one new token (decode-shaped).
    bool uniform_decode{false};
};

inline bool operator==(const BatchDescriptor &a, const BatchDescriptor &b) {
    return a.num_tokens == b.num_tokens && a.num_reqs == b.num_reqs
           && a.uniform_decode == b.uniform_decode;
}

namespace detail {

inline std::vector<size_t> parse_csv_sizes_(const char *raw) {
    std::vector<size_t> out;
    if (raw == nullptr || raw[0] == '\0') {
        return out;
    }
    std::string spec(raw);
    size_t start = 0;
    while (start < spec.size()) {
        const size_t comma = spec.find(',', start);
        const std::string token =
            spec.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!token.empty()) {
            out.push_back(static_cast<size_t>(std::stoul(token)));
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

} // namespace detail

class CudagraphDispatcher {
public:
    CudagraphDispatcher() = default;

    /// Load FULL / PIECEWISE key sets from env for the active cudagraph policy.
    void initialize_from_env() {
        full_keys_.clear();
        piecewise_keys_.clear();
        bs_to_padded_.clear();
        bs_to_padded_decode_.clear();
        max_capture_ = 0;
        max_decode_bs_ = 0;
        const char *policy = cudagraph_policy();
        if (std::strcmp(policy, "eager") == 0) {
            // No capture keys — dispatch always NONE.
            return;
        }
        if (std::strcmp(policy, "full_and_piecewise") != 0 && policy[0] != '\0') {
            // Unknown policy string already normalized to "" by cudagraph_policy().
            return;
        }
        // full_and_piecewise or legacy (empty policy with companion envs).
        if (std::strcmp(policy, "full_and_piecewise") == 0
            || std::strcmp(policy, "") == 0) {
            for (size_t b : detail::parse_csv_sizes_(std::getenv("INFINI_DECODE_CG_BATCHES"))) {
                full_keys_.insert(b);
            }
            for (size_t b :
                 detail::parse_csv_sizes_(std::getenv("INFINI_NATIVE_CG_CAPTURE_BUCKETS"))) {
                piecewise_keys_.insert(b);
            }
            rebuild_pad_table_();
            rebuild_decode_pad_table_();
        }
    }

    void add_full_key(size_t num_tokens) {
        full_keys_.insert(num_tokens);
        rebuild_decode_pad_table_();
    }
    void add_piecewise_key(size_t num_tokens) {
        piecewise_keys_.insert(num_tokens);
        rebuild_pad_table_();
    }

    const std::set<size_t> &full_keys() const { return full_keys_; }
    const std::set<size_t> &piecewise_keys() const { return piecewise_keys_; }

    /// Priority FULL > PIECEWISE > NONE. Returns (mode, padded key descriptor).
    /// Prefill pad-up matches vLLM ``bs_to_padded_graph_size``: in-range
    /// ``num_tokens`` maps to the next capture bucket; key.num_tokens is padded.
    /// Decode pad-up (default on under full_and_piecewise): exact miss maps to
    /// the next ``INFINI_DECODE_CG_BATCHES`` entry; kill-switch
    /// ``INFINI_DECODE_CG_PAD_UP=0``.
    std::pair<CudaGraphRuntimeMode, BatchDescriptor> dispatch(const BatchDescriptor &desc) const {
        if (std::strcmp(cudagraph_policy(), "eager") == 0) {
            return {CudaGraphRuntimeMode::None, desc};
        }
        if (desc.uniform_decode && !full_keys_.empty()) {
            if (full_keys_.count(desc.num_tokens) > 0) {
                BatchDescriptor key = desc;
                key.num_reqs = desc.num_tokens; // uniform decode: 1 token / req
                return {CudaGraphRuntimeMode::Full, key};
            }
            if (decode_cg_pad_up_enabled_() && desc.num_tokens <= max_decode_bs_) {
                const size_t padded = padded_bucket_for_seq_len(
                    desc.num_tokens, bs_to_padded_decode_, /*fallback=*/0);
                if (padded > 0 && full_keys_.count(padded) > 0) {
                    BatchDescriptor key = desc;
                    key.num_tokens = padded;
                    key.num_reqs = padded;
                    return {CudaGraphRuntimeMode::Full, key};
                }
            }
        }
        // Ragged / mixed / homogeneous prefill: vLLM-style pad-up of num_tokens
        // (no num_reqs==1 gate; capture must allow runtime_n_req ≤ max_capture_req).
        if (!desc.uniform_decode && !piecewise_keys_.empty()) {
            if (desc.num_tokens > max_capture_) {
                return {CudaGraphRuntimeMode::None, desc};
            }
            const size_t padded =
                padded_bucket_for_seq_len(desc.num_tokens, bs_to_padded_, /*fallback=*/0);
            if (padded > 0 && piecewise_keys_.count(padded) > 0) {
                BatchDescriptor key = desc;
                key.num_tokens = padded;
                return {CudaGraphRuntimeMode::Piecewise, key};
            }
        }
        return {CudaGraphRuntimeMode::None, desc};
    }

    /// Classify why ``dispatch`` returned NONE (for profile / hang_trace histograms).
    /// ``is_mixed`` is the scheduler MIXED / ragged flag from RankWorker.
    const char *none_reason(const BatchDescriptor &desc, bool is_mixed) const {
        if (std::strcmp(cudagraph_policy(), "eager") == 0) {
            return "eager_policy";
        }
        if (desc.uniform_decode) {
            if (!full_keys_.empty() && desc.num_tokens > max_decode_bs_) {
                return "decode_bs_over_max";
            }
            return "decode_bs_miss";
        }
        if (!piecewise_keys_.empty() && desc.num_tokens > max_capture_) {
            return "over_max";
        }
        // Rare: no keys / pad table miss. Keep legacy labels for hist continuity.
        if (is_mixed) {
            return "mixed";
        }
        if (desc.num_reqs > 1) {
            return "multi_req_prefill";
        }
        return "bucket_miss";
    }

private:
    /// Decode batch pad-up default on; ``INFINI_DECODE_CG_PAD_UP=0`` disables.
    static bool decode_cg_pad_up_enabled_() {
        const char *v = std::getenv("INFINI_DECODE_CG_PAD_UP");
        if (v != nullptr && v[0] == '0' && v[1] == '\0') {
            return false;
        }
        return true;
    }

    void rebuild_pad_table_() {
        bs_to_padded_.clear();
        max_capture_ = 0;
        if (piecewise_keys_.empty()) {
            return;
        }
        std::vector<size_t> caps(piecewise_keys_.begin(), piecewise_keys_.end());
        bs_to_padded_ = build_bs_to_padded_bucket(caps);
        max_capture_ = *piecewise_keys_.rbegin();
    }

    void rebuild_decode_pad_table_() {
        bs_to_padded_decode_.clear();
        max_decode_bs_ = 0;
        if (full_keys_.empty()) {
            return;
        }
        std::vector<size_t> caps(full_keys_.begin(), full_keys_.end());
        bs_to_padded_decode_ = build_bs_to_padded_bucket(caps);
        max_decode_bs_ = *full_keys_.rbegin();
    }

    std::set<size_t> full_keys_;
    std::set<size_t> piecewise_keys_;
    std::vector<size_t> bs_to_padded_;
    std::vector<size_t> bs_to_padded_decode_;
    size_t max_capture_{0};
    size_t max_decode_bs_{0};
};

} // namespace infinilm::engine
