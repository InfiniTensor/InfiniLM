#pragma once
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

    infinicore::nn::QuantScheme get_quant_scheme() const;

private:
    nlohmann::json config_json;
    quantization::QuantConfig quant_config;
};
} // namespace infinilm::config::global_config
