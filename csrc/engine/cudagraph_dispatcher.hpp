#pragma once

/// vLLM-shaped CUDA-graph runtime dispatch (thin InfiniLM mirror).
///
/// Selects FULL / PIECEWISE / NONE from a BatchDescriptor derived from input
/// shape (not scheduler PREFILL/DECODE/MIXED). MIXED → NONE until ragged
/// mixed piecewise exists. Under ``full_and_piecewise``:
///   - uniform decode batches in ``INFINI_DECODE_CG_BATCHES`` → FULL
///   - homogeneous single-req prefill: pad-up ``num_tokens`` to the next
///     ``INFINI_NATIVE_CG_CAPTURE_BUCKETS`` entry (vLLM ``bs_to_padded_graph_size``)
///     → PIECEWISE with ``key.num_tokens = padded``; eager only when
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
        max_capture_ = 0;
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
        }
    }

    void add_full_key(size_t num_tokens) { full_keys_.insert(num_tokens); }
    void add_piecewise_key(size_t num_tokens) {
        piecewise_keys_.insert(num_tokens);
        rebuild_pad_table_();
    }

    const std::set<size_t> &full_keys() const { return full_keys_; }
    const std::set<size_t> &piecewise_keys() const { return piecewise_keys_; }

    /// Priority FULL > PIECEWISE > NONE. Returns (mode, padded key descriptor).
    /// Prefill pad-up matches vLLM ``bs_to_padded_graph_size``: in-range
    /// ``num_tokens`` maps to the next capture bucket; key.num_tokens is padded.
    std::pair<CudaGraphRuntimeMode, BatchDescriptor> dispatch(const BatchDescriptor &desc) const {
        if (std::strcmp(cudagraph_policy(), "eager") == 0) {
            return {CudaGraphRuntimeMode::None, desc};
        }
        if (desc.uniform_decode && full_keys_.count(desc.num_tokens) > 0) {
            BatchDescriptor key = desc;
            key.num_reqs = desc.num_tokens; // uniform decode: 1 token / req
            return {CudaGraphRuntimeMode::Full, key};
        }
        // Homogeneous single-req prefill with vLLM-style pad-up (MIXED / multi-req → NONE).
        if (!desc.uniform_decode && desc.num_reqs == 1 && !piecewise_keys_.empty()) {
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
        if (is_mixed) {
            return "mixed";
        }
        if (!desc.uniform_decode && desc.num_reqs > 1) {
            return "multi_req_prefill";
        }
        if (desc.uniform_decode) {
            return "decode_bs_miss";
        }
        if (!piecewise_keys_.empty() && desc.num_tokens > max_capture_) {
            return "over_max";
        }
        return "bucket_miss";
    }

private:
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

    std::set<size_t> full_keys_;
    std::set<size_t> piecewise_keys_;
    std::vector<size_t> bs_to_padded_;
    size_t max_capture_{0};
};

} // namespace infinilm::engine
