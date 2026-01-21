#include "global_config.hpp"

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

infinicore::nn::QuantScheme
GlobalConfig::get_quant_scheme() const {
    if (quant_config.get_quant_scheme() != infinicore::nn::QuantScheme::NONE) {
        return quant_config.get_quant_scheme();
    } else {
        return infinicore::nn::QuantScheme::NONE;
    }
}
} // namespace infinilm::config::global_config
