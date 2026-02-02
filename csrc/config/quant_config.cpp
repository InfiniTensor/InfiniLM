#include "quant_config.hpp"

namespace infinilm::config {
QuantConfig::QuantConfig(const nlohmann::json &json) : quantization_config(json) {
    this->quantization_method = get_quantization_method();
}

std::shared_ptr<infinicore::quantization::BaseQuantization>
QuantConfig::get_quantization_method() const {
    if (quantization_config.is_null()) {
        // return nullptr;
        return std::make_shared<infinicore::quantization::NoneQuantization>(quantization_config); // Default case if no matching scheme
    }

    // Determine the quantization scheme from the JSON config
    if (quantization_config["quant_method"] == "compressed-tensors") {
        return std::make_shared<infinicore::quantization::CompressedTensors>(quantization_config);
    } else if (quantization_config["quant_method"] == "awq") {
        return std::make_shared<infinicore::quantization::AWQ>(quantization_config);
    } else {
        return std::make_shared<infinicore::quantization::NoneQuantization>(quantization_config);
    }
    // Add other schemes as needed

    return std::make_shared<infinicore::quantization::NoneQuantization>(quantization_config); // Default case if no matching scheme
}
} // namespace infinilm::config
