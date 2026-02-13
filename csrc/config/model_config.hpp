#pragma once

#include "infinicore/nn/rope.hpp"
#include "infinicore/ops.hpp"
#include "quant_config.hpp"
#include <fstream>
#include <string>

namespace infinilm::config {
class ModelConfig {
    // Model config is implemented using nlohmann/json and is primarily used for advanced configuration
    // beyond the standard model config. It is initialized via ModelConfig(const std::string& path)
    // and passed through the InferEngine during inference.
public:
    ModelConfig() = default;
    // Not Implemented
    // ModelConfig(const nlohmann::json &json) : config_json(json) {};
    ModelConfig(const std::string &path);

    // Template Function to get a value by key with type safety
    template <typename T>
    T get(const std::string &key) const {
        if (!config_json.contains(key)) {
            throw std::out_of_range("Key '" + key + "' not found in config.");
        }
        try {
            return config_json.at(key).get<T>();
        } catch (const nlohmann::json::type_error &e) {
            throw std::runtime_error("Type conversion failed for key '" + key + "': " + std::string(e.what()));
        }
    }

    template <typename T>
    T get_or(const std::string &key, const T &default_value) const {
        if (!config_json.contains(key) || config_json.at(key).is_null()) {
            return default_value;
        }
        try {
            return config_json.at(key).get<T>();
        } catch (const nlohmann::json::type_error &) {
            // If type conversion fails, return default value
            return default_value;
        }
    }
    size_t get_kv_dim() const {
        return get<size_t>("hidden_size") * get<size_t>("num_key_value_heads") / get<size_t>("num_attention_heads");
    }
    size_t get_head_dim() const {
        if (config_json.contains("head_dim")) {
            return get<size_t>("head_dim");
        }
        return get<size_t>("hidden_size") / get<size_t>("num_attention_heads");
    }

    QuantConfig get_quant_config() const {
        return quant_config;
    }

    std::shared_ptr<infinicore::quantization::BaseQuantization> get_quantization_method() const {
        return quant_config.get_quantization_method();
    }

    infinicore::DataType get_dtype() const;
    infinicore::quantization::QuantScheme get_quant_scheme() const;
    std::shared_ptr<infinicore::nn::RoPE::ScalingConfig> get_rope_scaling() const;

private:
    nlohmann::json config_json;
    QuantConfig quant_config;
};
} // namespace infinilm::config
