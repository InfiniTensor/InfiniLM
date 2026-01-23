#pragma once

// #include "infinicore/nn/quantization.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/ops.hpp"
#include "quant_config.hpp"
#include <fstream>
#include <string>

namespace infinilm::config::global_config {
struct GlobalConfig {
    // Global config is implemented using nlohmann/json and is primarily used for advanced configuration
    // beyond the standard model config. It is initialized via GlobalConfig(const std::string& path)
    // and passed through the InferEngine during inference.
public:
    GlobalConfig() = default;
    GlobalConfig(const nlohmann::json &json) : config_json(json) {};
    GlobalConfig(const std::string &path);

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
        return get<size_t>("hidden_size") / get<size_t>("num_attention_heads");
    }

    infinicore::DataType get_dtype() const;
    infinicore::nn::QuantScheme get_quant_scheme() const;
    std::shared_ptr<infinicore::nn::RoPE::ScalingConfig> get_rope_scaling() const;

private:
    nlohmann::json config_json;
    quantization::QuantConfig quant_config;
};
} // namespace infinilm::config::global_config
