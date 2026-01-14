#pragma once
// #include "../quantization/quantization.hpp"
#include "infinicore/quantization.hpp"
#include "nlohmann/json.hpp"

namespace infinilm::config {

class QuantConfig {
    // QuantConfig is used to store and parse the "quantization" field from config.json.
    // This is currently a basic version and will be extended in the future.
public:
    QuantConfig() = default;
    QuantConfig(const nlohmann::json &json);

    std::shared_ptr<infinicore::quantization::BaseQuantization> get_quantization_method() const;

    infinicore::quantization::QuantScheme get_quant_scheme() const {
        if (quantization_method != nullptr) {
            return quantization_method->get_quant_scheme();
        } else {
            return infinicore::quantization::QuantScheme::NONE;
        }
    }

private:
    nlohmann::json quantization_config;
    std::shared_ptr<infinicore::quantization::BaseQuantization> quantization_method;
};

} // namespace infinilm::config
