#include "quant_config.hpp"

namespace infinilm::config {
QuantConfig::QuantConfig(const nlohmann::json &json) : quantization_config(json) {
    this->quantization_method = get_quantization_method();
}

std::shared_ptr<infinilm::quantization::BaseQuantization>
QuantConfig::get_quantization_method() const {
    if (quantization_config.is_null()) {
        return std::make_shared<infinilm::quantization::NoneQuantization>(quantization_config); // Default case if no matching scheme
    }

    // Determine the quantization scheme from the JSON config
    if (quantization_config["quant_method"] == "compressed-tensors") {
        return std::make_shared<infinilm::quantization::CompressedTensors>(quantization_config);
    } else if (quantization_config["quant_method"] == "awq") {
        return std::make_shared<infinilm::quantization::AWQ>(quantization_config);
    } else if (quantization_config["quant_method"] == "gptq") {
        return std::make_shared<infinilm::quantization::GPTQ>(quantization_config);
    } else {
        return std::make_shared<infinilm::quantization::NoneQuantization>(quantization_config);
    }
    // Add other schemes as needed

    return std::make_shared<infinilm::quantization::NoneQuantization>(quantization_config); // Default case if no matching scheme
}
} // namespace infinilm::config
