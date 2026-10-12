#pragma once

#include "../utils.hpp"
#include "../layers/rotary_embedding/rope.hpp"
#include "quant_config.hpp"
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace infinilm::config {
class ModelConfig {
    // Model config is implemented using nlohmann/json and is primarily used for advanced configuration
    // beyond the standard model config. It is initialized via ModelConfig(const std::string& path)
    // and passed through the InferEngine during inference.
public:
    ModelConfig() = default;
    ModelConfig(const nlohmann::json &json);
    ModelConfig(const std::string &path);

    nlohmann::json &get_config_json() {
        return config_json;
    }
    const nlohmann::json &get_config_json() const {
        return config_json;
    }

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
        return get_or<T>({std::string_view(key)}, default_value);
    }

    template <typename T>
    T get_or(std::initializer_list<std::string_view> keys, const T &default_value) const {
        for (std::string_view key : keys) {
            if (key.empty()) {
                continue;
            }
            auto it = config_json.find(std::string(key));
            if (it == config_json.end() || it->is_null()) {
                continue;
            }
            try {
                return it->get<T>();
            } catch (const nlohmann::json::type_error &) {
                return default_value;
            }
        }
        return default_value;
    }

    template <typename T>
    T get_or_alias(const std::string &key, const std::string &alias, const T &default_value) const {
        return get_or<T>({key, alias}, default_value);
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

    // Compute the actual rotary dimension based on partial rotation factor
    size_t get_rotary_dim() const;

    QuantConfig get_quant_config() const {
        return quant_config;
    }

    std::shared_ptr<infinilm::quantization::BaseQuantization> get_quantization_method() const {
        return quant_config.get_quantization_method();
    }

    infinilm::runtime::DataType get_dtype() const;
    infinilm::quantization::QuantScheme get_quant_scheme() const;

    void set_kv_quant_scheme(infinilm::runtime::DataType kv_cache_dtype) {
        this->quant_config.set_kv_quant_scheme(kv_cache_dtype);
    }
    infinilm::quantization::KVQuantAlgo get_kv_quant_scheme() const {
        return quant_config.get_kv_quant_scheme();
    }
    infinilm::runtime::DataType get_kv_cache_dtype() const {
        if (this->quant_config.get_kv_cache_dtype().has_value()) {
            return this->quant_config.get_kv_cache_dtype().value();
        } else {
            return this->get_dtype();
        }
    }

    // Get reference to JSON value (non-const)
    nlohmann::json &get_ref(const std::string &key) {
        if (!config_json.contains(key)) {
            throw std::out_of_range("Key '" + key + "' not found in config.");
        }
        return config_json.at(key);
    }

    // Get const reference to JSON value
    const nlohmann::json &get_ref(const std::string &key) const {
        if (!config_json.contains(key)) {
            throw std::out_of_range("Key '" + key + "' not found in config.");
        }
        return config_json.at(key);
    }

    // Stream output operator
    friend std::ostream &operator<<(std::ostream &os, const ModelConfig &config);

    infinilm::layers::RoPE::Algo get_rope_algo() const {
        return rope_algo_;
    }

    void set_rope_algo(infinilm::layers::RoPE::Algo algo) {
        rope_algo_ = algo;
    }

    // RoPE tables belong to this model configuration's execution instance,
    // not to a process- or thread-global cache shared by unrelated models.
    std::shared_ptr<infinilm::layers::RoPE> get_or_create_rope(
        const std::string &key,
        const std::function<std::shared_ptr<infinilm::layers::RoPE>()> &create) const;

private:
    struct RopeCache {
        std::mutex mutex;
        std::unordered_map<std::string, std::shared_ptr<infinilm::layers::RoPE>> entries;
    };
    nlohmann::json config_json;
    QuantConfig quant_config;

    infinilm::layers::RoPE::Algo rope_algo_ = infinilm::layers::RoPE::Algo::GPT_NEOX;
    std::shared_ptr<RopeCache> rope_cache_ = std::make_shared<RopeCache>();
};
} // namespace infinilm::config
