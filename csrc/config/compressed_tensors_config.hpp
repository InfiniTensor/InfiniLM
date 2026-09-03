#pragma once

#include "nlohmann/json.hpp"

#include <optional>
#include <string>
#include <vector>

namespace infinilm::config {

// Values mirror the public `compressed-tensors` configuration vocabulary. They
// describe checkpoint metadata, not a concrete InfiniLM kernel implementation.
enum class QuantizationValueType {
    INT,
    FLOAT,
};

enum class QuantizationStrategy {
    TENSOR,
    CHANNEL,
    GROUP,
    BLOCK,
    TOKEN,
    TENSOR_GROUP,
    ATTN_HEAD,
};

// The `compressed-tensors` format accepts `false` (static), `true` (fully
// dynamic), and `"local"` (only local quantization parameters are dynamic).
enum class QuantizationDynamicMode {
    STATIC,
    DYNAMIC,
    LOCAL,
};

enum class QuantizationStatus {
    INITIALIZED,
    CALIBRATION,
    FROZEN,
    COMPRESSED,
    DECOMPRESSED,
};

struct QuantizationArgs {
    int num_bits = 8;
    QuantizationValueType type = QuantizationValueType::INT;
    bool symmetric = true;
    QuantizationStrategy strategy = QuantizationStrategy::TENSOR;
    QuantizationDynamicMode dynamic = QuantizationDynamicMode::STATIC;
};

struct QuantizationGroup {
    std::string name;
    std::vector<std::string> targets;
    std::optional<QuantizationArgs> weights;
    std::optional<QuantizationArgs> input_activations;
    std::optional<QuantizationArgs> output_activations;
    std::optional<std::string> format;
};

struct CompressedTensorsConfig {
    static CompressedTensorsConfig from_json(const nlohmann::json &config);

    std::string quant_method = "compressed-tensors";
    std::string format = "fakequant";
    QuantizationStatus quantization_status = QuantizationStatus::INITIALIZED;
    std::vector<QuantizationGroup> config_groups;
    std::vector<std::string> ignore;
    std::optional<QuantizationArgs> kv_cache_scheme;
    std::optional<double> global_compression_ratio;

    // Keep the source object for diagnostics and forward-compatible inspection
    // of metadata that does not affect the currently supported inference path.
    nlohmann::json raw_config;
};

} // namespace infinilm::config
