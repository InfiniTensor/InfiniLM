#pragma once

#include "quant_config.hpp"
#include <fstream>
#include <optional>
#include <string>

namespace infinilm::config::global_config {
struct GlobalConfig {
    // Quantization configuration
public:
    infinilm::config::quantization::QuantConfig get_quant_config_json() const {
        return infinilm::config::quantization::QuantConfig(config_json.value("quantization_config", nlohmann::json::object())).to_json();
    }

    GlobalConfig() = default;
    GlobalConfig(const nlohmann::json &json) : config_json(json) {};
    GlobalConfig(const std::string &path) {
        std::ifstream file(path);
        if (file.is_open()) {
            file >> config_json;
            file.close();
        } else {
            throw std::runtime_error("Could not open config file: " + path);
        }
    }

private:
    nlohmann::json config_json;
};
} // namespace infinilm::config::global_config