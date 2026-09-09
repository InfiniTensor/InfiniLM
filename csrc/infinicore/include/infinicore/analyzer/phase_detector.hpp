#pragma once

#include "op_trace.hpp"
#include "optimization_intent.hpp"

#include <algorithm>
#include <cstddef>

namespace infinicore::analyzer {

/// PhaseDetector — detects the current task phase from the
/// recent op trace window using rule-based pattern matching.
///
/// Design choice (MVP): Fixed rule matching based on op type
/// composition in a sliding window. Will evolve to support
/// offline-generated phase templates in future iterations.
class PhaseDetector {
public:
    /// Configuration for phase detection thresholds.
    struct Config {
        size_t window_size;
        float attention_threshold;
        float gemm_mlp_threshold;
        float kv_cache_threshold;
        uint32_t decode_seq_len_max;
        uint32_t prefill_seq_len_min;

        Config()
            : window_size(16),
              attention_threshold(0.3f),
              gemm_mlp_threshold(0.3f),
              kv_cache_threshold(0.4f),
              decode_seq_len_max(4),
              prefill_seq_len_min(32) {}
    };

    explicit PhaseDetector(Config config = {}) : config_(config) {}

    /// Detect the current phase from a window of recent op traces.
    PhaseType detect(const std::vector<OpTraceEntry> &window) const {
        if (window.empty()) {
            return PhaseType::UNKNOWN;
        }

        // Count op categories in the window
        size_t attention_count = 0;
        size_t gemm_mlp_count = 0;
        size_t kv_cache_count = 0;
        size_t activation_count = 0;
        size_t total = window.size();

        // Track shape info for prefill/decode inference
        uint32_t max_seq_len = 0;
        uint32_t min_seq_len = UINT32_MAX;
        bool has_attention_shape = false;

        for (auto &entry : window) {
            if (isAttentionOp(entry.op_type)) {
                attention_count++;
                // For attention ops, shape[1] or shape[2] typically indicates seq_len
                if (entry.ndim >= 2) {
                    // Heuristic: for attention-like ops, look at the sequence dimension
                    // Typically shape = [batch, seq_len, ...] or [batch, heads, seq_len, ...]
                    uint32_t seq_dim = (entry.ndim >= 3) ? entry.shape[2] : entry.shape[1];
                    max_seq_len = std::max(max_seq_len, seq_dim);
                    min_seq_len = std::min(min_seq_len, seq_dim);
                    has_attention_shape = true;
                }
            } else if (isGemmMlpOp(entry.op_type)) {
                gemm_mlp_count++;
            } else if (isKvCacheOp(entry.op_type)) {
                kv_cache_count++;
            } else if (isActivationOp(entry.op_type)) {
                activation_count++;
            }
        }

        float attention_ratio = static_cast<float>(attention_count) / total;
        float gemm_mlp_ratio = static_cast<float>(gemm_mlp_count + activation_count) / total;
        float kv_cache_ratio = static_cast<float>(kv_cache_count) / total;

        // --- Phase classification ---

        // KV cache phase (high KV cache op ratio)
        if (kv_cache_ratio >= config_.kv_cache_threshold) {
            return PhaseType::KV_CACHE;
        }

        // --- Prefill vs Decode inference from shape ---
        // Self-inferred from sequence length, no external flags needed.
        if (has_attention_shape) {
            if (max_seq_len <= config_.decode_seq_len_max) {
                return PhaseType::DECODE;
            }
            if (min_seq_len >= config_.prefill_seq_len_min) {
                return PhaseType::PREFILL;
            }
        }

        // Attention-dense phase
        if (attention_ratio >= config_.attention_threshold && attention_ratio >= gemm_mlp_ratio) {
            return PhaseType::ATTENTION_DENSE;
        }

        // GEMM/MLP-dense phase (include activation ops as co-indicators)
        if (gemm_mlp_ratio >= config_.gemm_mlp_threshold && gemm_mlp_ratio >= attention_ratio) {
            return PhaseType::GEMM_MLP_DENSE;
        }

        return PhaseType::UNKNOWN;
    }

    /// Convenience: detect from the global trace ring.
    PhaseType detectFromTrace(const OpTraceRing &trace) const {
        auto window = trace.getRecentEntries(config_.window_size);
        return detect(window);
    }

    const Config &config() const { return config_; }
    void setConfig(Config config) { config_ = config; }

private:
    Config config_;
};

} // namespace infinicore::analyzer
