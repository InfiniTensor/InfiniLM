#pragma once

#include "nlohmann/json.hpp"

namespace infinilm::config::quantization {

struct QuantConfig {
    nlohmann::json quantization_config;

    QuantConfig() = default;
    QuantConfig(const nlohmann::json &json) : quantization_config(json) {};
    nlohmann::json to_json() const {
        return quantization_config;
    }
};

} // namespace infinilm::config::quantization