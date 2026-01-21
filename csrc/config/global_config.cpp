#include "global_config.hpp"
#include <iostream>
namespace infinilm::config::global_config {
GlobalConfig::GlobalConfig(const std::string &path) {
    std::ifstream file(path);
    if (file.is_open()) {
        file >> config_json;
        file.close();
    } else {
        throw std::runtime_error("Could not open config file: " + path);
    }
    this->quant_config = quantization::QuantConfig(config_json["quantization_config"]);
}
} // namespace infinilm::config::global_config