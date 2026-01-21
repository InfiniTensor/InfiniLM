#pragma once

#include "quant_config.hpp"
#include <fstream>
#include <optional>
#include <string>

namespace infinilm::config::global_config {
struct GlobalConfig {
    // Quantization configuration
public:
    GlobalConfig() = default;
    GlobalConfig(const nlohmann::json &json) : config_json(json) {};
    GlobalConfig(const std::string &path);

    infinicore::nn::QuantScheme get_quant_scheme() const {
        if (quant_config.get_quant_scheme() != infinicore::nn::QuantScheme::NONE) {
            return quant_config.get_quant_scheme();
        } else {
            return infinicore::nn::QuantScheme::NONE;
        }
    }

private:
    nlohmann::json config_json;
    quantization::QuantConfig quant_config;
};
} // namespace infinilm::config::global_config