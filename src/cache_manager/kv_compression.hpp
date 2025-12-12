#pragma once

#include "../cache.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Compression configuration set at model creation.
struct CompressionConfig {
    bool enable = false;
    uint32_t compression_factor = 1; // e.g., 4 or 5
    uint32_t min_seq_len = 0;        // threshold to trigger compression
    std::string weight_path;         // path to binary weights
    uint32_t image_kv_len = 0;       // optional prefix length (tokens) treated as image KV
    // Future: per-layer override, algorithm type, dtype override.
};

// Metadata and storage for compressed KV per device.
struct CompressedKV {
    struct LayerKV {
        std::shared_ptr<Tensor> k_comp; // compressed key
        std::shared_ptr<Tensor> v_comp; // compressed value
        uint32_t orig_seq_len = 0;
        uint32_t comp_seq_len = 0;
        std::shared_ptr<Tensor> indices; // optional: mapping indices (int32)
        std::shared_ptr<Tensor> scales;  // optional: scaling factors
    };

    // Device-local layout: [layer] order matches original KV.
    std::vector<LayerKV> layers;
};

// Compressor interface (declaration only; implementation to be added in a new .cpp).
class Compressor {
public:
    explicit Compressor(const CompressionConfig &cfg);

    // Load weights from binary file; returns false on failure.
    bool loadWeights();

    // Compress device-local KV; returns compressed structure or nullptr on failure.
    std::unique_ptr<CompressedKV> compress(const KVCache &kv, uint32_t seq_len);

    // Decompress into temporary buffers for attention use; returns false on failure.
    bool decompress(const CompressedKV &ckv,
                    std::vector<std::shared_ptr<Tensor>> &k_out,
                    std::vector<std::shared_ptr<Tensor>> &v_out);

private:
    struct LinearWeight {
        std::shared_ptr<Tensor> weight;
        std::shared_ptr<Tensor> bias;
    };

    CompressionConfig config_;
    // Store per-layer weights; concrete types depend on algorithm design.
    std::vector<std::shared_ptr<Tensor>> weights_;
    // Offsets into weights_ for each layer (start index of that layer).
    std::vector<uint32_t> prefix_offsets_;
    // Structured access: [layer][block] -> (weight, bias)
    std::vector<std::vector<LinearWeight>> layered_weights_;

    // Helper to fetch weight tensor by (layer, index offset within layer).
    std::shared_ptr<Tensor> getWeight(uint32_t layer, uint32_t idx) const;

    // Mapping utilities: each layer is expected to have weights ordered by prefix.
    // Order: compress_tk[3], compress_tv[3], compress_ik[3], compress_iv[3], attention[*] (if present).
    static constexpr uint32_t kWeightsPerPrefix = 3;
    static constexpr uint32_t kPrefixCount = 4; // compress_tk, compress_tv, compress_ik, compress_iv

    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
    getLinearWithBias(uint32_t layer, uint32_t prefix_idx, uint32_t slot) const;
};
