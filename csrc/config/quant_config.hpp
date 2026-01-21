#pragma once
#include "../quantization/quantization.hpp"
#include "nlohmann/json.hpp"

namespace infinilm::config::quantization {

class QuantConfig {
    // QuantConfig is used to store and parse the "quantization" field from config.json.
    // This is currently a basic version and will be extended in the future.
public:
    QuantConfig() = default;
    QuantConfig(const nlohmann::json &json);

    infinicore::nn::QuantScheme get_quant_scheme() const {
        if (quantization_method != nullptr) {
            return quantization_method->get_quant_scheme();
        } else {
            return infinicore::nn::QuantScheme::NONE;
        }
    }

private:
    nlohmann::json quantization_config;
    std::shared_ptr<infinilm::quantization::BaseQuantization> get_quantization_method() const;
    std::shared_ptr<infinilm::quantization::BaseQuantization> quantization_method;
};

} // namespace infinilm::config::quantization
