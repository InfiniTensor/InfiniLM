#pragma once

#include "model_config.hpp"
#include "runtime/device.hpp"
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace infinilm::config {

inline bool supports_qwen3_memory_optimization(
    const ModelConfig &model, runtime::Device::Type device, int tp_size, int pp_size) {
    return model.get_or<std::string>("model_type", "") == "qwen3"
        && device == runtime::Device::Type::kNvidia
        && model.get_dtype() == runtime::DataType::kBFloat16
        && pp_size == 1 && (tp_size == 1 || tp_size == 2)
        && model.get_quant_scheme() == quantization::QuantScheme::NONE
        && model.get_kv_quant_scheme() == quantization::KVQuantAlgo::NONE;
}

inline size_t qwen3_prefill_chunk_size(const ModelConfig &model) {
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

inline bool qwen3_tp_vocab_sharding(const ModelConfig &model) {
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

} // namespace infinilm::config
