#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace infinilm::engine {

constexpr size_t kPiecewisePowerLadderFloor = 512;
constexpr size_t kPiecewisePowerLadderCap = 8192;
constexpr size_t kPiecewiseOverflowTailBucket = 8448;

inline size_t compile_max_seq_from_env(size_t default_max = kPiecewiseOverflowTailBucket) {
    const char *raw = std::getenv("INFINI_COMPILE_MAX_SEQ");
    if (raw == nullptr || raw[0] == '\0') {
        return default_max;
    }
    return static_cast<size_t>(std::stoul(raw));
}

inline bool vllm_capture_ladder_enabled() {
    const char *raw = std::getenv("INFINI_VLLM_CAPTURE_LADDER");
    if (raw == nullptr || raw[0] == '\0') {
        return false;
    }
    const std::string v(raw);
    return v != "0" && v != "false" && v != "no" && v != "off";
}

inline size_t prefill_chunk_size_from_env(size_t default_size = 512) {
    const char *raw = std::getenv("INFINI_PREFILL_CHUNK_SIZE");
    if (raw == nullptr || raw[0] == '\0') {
        return default_size;
    }
    return static_cast<size_t>(std::stoul(raw));
}

inline std::vector<size_t> vllm_piecewise_capture_sizes(size_t chunk_cap) {
    std::vector<size_t> buckets;
    const size_t cap = std::min(chunk_cap, kPiecewisePowerLadderFloor);
    for (size_t s = 1; s <= cap; s *= 2) {
        buckets.push_back(s);
        if (s > SIZE_MAX / 2) {
            break;
        }
    }
    return buckets;
}

inline std::vector<size_t> merge_bucket_ladders(const std::vector<size_t> &a,
                                                const std::vector<size_t> &b) {
    std::vector<size_t> merged = a;
    merged.insert(merged.end(), b.begin(), b.end());
    std::sort(merged.begin(), merged.end());
    merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
    return merged;
}

inline std::vector<size_t> piecewise_compile_buckets(size_t max_seq_len) {
    std::vector<size_t> buckets;
    for (size_t s = kPiecewisePowerLadderFloor; s <= std::min(max_seq_len, kPiecewisePowerLadderCap); s *= 2) {
        buckets.push_back(s);
    }
    if (max_seq_len > kPiecewisePowerLadderCap) {
        buckets.push_back(max_seq_len);
    } else if (max_seq_len > 0 && (buckets.empty() || buckets.back() != max_seq_len)) {
        buckets.push_back(max_seq_len);
    }
    if (max_seq_len >= kPiecewisePowerLadderCap && max_seq_len < kPiecewiseOverflowTailBucket) {
        buckets.push_back(kPiecewiseOverflowTailBucket);
    }
    std::sort(buckets.begin(), buckets.end());
    buckets.erase(std::unique(buckets.begin(), buckets.end()), buckets.end());
    return buckets;
}

inline std::vector<size_t> piecewise_compile_buckets_vllm(size_t max_seq_len, size_t chunk_cap) {
    return merge_bucket_ladders(
        vllm_piecewise_capture_sizes(std::min(chunk_cap, kPiecewisePowerLadderFloor)),
        piecewise_compile_buckets(max_seq_len));
}

inline std::vector<size_t> piecewise_capture_buckets_vllm(size_t max_seq_len, size_t chunk_cap) {
    auto buckets = piecewise_compile_buckets_vllm(max_seq_len, chunk_cap);
    buckets.erase(
        std::remove_if(buckets.begin(), buckets.end(),
                       [](size_t b) { return b > kPiecewisePowerLadderCap; }),
        buckets.end());
    return buckets;
}

inline std::vector<size_t> piecewise_capture_buckets(size_t max_seq_len) {
    auto buckets = piecewise_compile_buckets(max_seq_len);
    // Pad ladder may include 8448 overflow tail; graphs capture through 8192 only.
    buckets.erase(
        std::remove_if(buckets.begin(), buckets.end(),
                       [](size_t b) { return b > kPiecewisePowerLadderCap; }),
        buckets.end());
    return buckets;
}

inline std::vector<size_t> build_bs_to_padded_bucket(const std::vector<size_t> &capture_sizes) {
    if (capture_sizes.empty()) {
        return {0};
    }
    std::vector<size_t> sizes = capture_sizes;
    std::sort(sizes.begin(), sizes.end(), std::greater<size_t>());
    const size_t max_capture = sizes.front();
    std::vector<size_t> table(max_capture + 1, 0);
    for (size_t i = 0; i < sizes.size(); ++i) {
        const size_t end = sizes[i];
        const size_t start = (i + 1 < sizes.size()) ? sizes[i + 1] : 0;
        for (size_t bs = start; bs < end; ++bs) {
            table[bs] = (bs == start) ? start : end;
        }
    }
    table[max_capture] = max_capture;
    return table;
}

inline size_t padded_bucket_for_seq_len(size_t seq_len,
                                        const std::vector<size_t> &bs_to_padded,
                                        size_t fallback) {
    if (seq_len < bs_to_padded.size() && bs_to_padded[seq_len] > 0) {
        return bs_to_padded[seq_len];
    }
    return fallback;
}

inline size_t graph_replay_bucket_for_padded(size_t padded_bucket) {
    return padded_bucket;
}

} // namespace infinilm::engine
