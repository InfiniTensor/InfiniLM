#pragma once

#include "../backends/attention_backends.hpp"
#include "model_config.hpp"
#include "runtime/device.hpp"
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace infinilm::config {

inline bool supports_dense_memory_ops(const ModelConfig &model, runtime::Device::Type device) {
    return device == runtime::Device::Type::kNvidia
        && model.get_dtype() == runtime::DataType::kBFloat16
        && model.get_quant_scheme() == quantization::QuantScheme::NONE;
}

inline bool supports_chunked_prefill(
    const ModelConfig &model, runtime::Device::Type device, int tp_size, int pp_size,
    backends::AttentionBackend backend, bool segmentable_decoder) {
    return segmentable_decoder && supports_dense_memory_ops(model, device)
        && tp_size > 0 && pp_size == 1 && backend == backends::AttentionBackend::PAGED_ATTN
        && model.get_kv_quant_scheme() == quantization::KVQuantAlgo::NONE;
}

inline bool supports_tp_weight_sharding(
    const ModelConfig &model, runtime::Device::Type device, int tp_size, int pp_size) {
    return supports_dense_memory_ops(model, device) && tp_size > 1 && pp_size == 1;
}

inline size_t prefill_chunk_size(const ModelConfig &model) {
    const auto &json = model.get_config_json();
    const auto value = json.find("infinilm_prefill_chunk_size");
    if (value == json.end()) {
        return 4096;
    }
    if ((!value->is_number_integer() && !value->is_number_unsigned())
        || *value < 0 || *value > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument(
            "`infinilm_prefill_chunk_size` must be an integer between 0 and INT32_MAX.");
    }
    return value->get<size_t>();
}

inline bool tp_vocab_sharding(const ModelConfig &model) {
    const auto &json = model.get_config_json();
    const auto value = json.find("infinilm_tp_vocab_sharding");
    if (value == json.end()) {
        return true;
    }
    if (!value->is_boolean()) {
        throw std::invalid_argument("`infinilm_tp_vocab_sharding` must be a boolean.");
    }
    return value->get<bool>();
}

inline bool tp_embedding_sharding(const ModelConfig &model) {
    const auto &json = model.get_config_json();
    const auto value = json.find("infinilm_tp_embedding_sharding");
    if (value == json.end()) {
        // Preserve the original combined opt-out when no embedding override is set.
        return tp_vocab_sharding(model);
    }
    if (!value->is_boolean()) {
        throw std::invalid_argument("`infinilm_tp_embedding_sharding` must be a boolean.");
    }
    return value->get<bool>();
}

} // namespace infinilm::config
