#include "quant_config.hpp"
#include <stdexcept>

namespace infinilm::config {
QuantConfig::QuantConfig(const nlohmann::json &json) : quantization_config(json) {
    this->quantization_method = get_quantization_method();
}

std::shared_ptr<infinilm::quantization::BaseQuantization>
QuantConfig::get_quantization_method() const {
    if (quantization_config.is_null()) {
        return std::make_shared<infinilm::quantization::NoneQuantization>(quantization_config); // Default case if no matching scheme
    }

    const std::string quant_method = quantization_config.value("quant_method", "");

    if (quant_method == "compressed-tensors") {
        return std::make_shared<infinilm::quantization::CompressedTensors>(quantization_config);
    } else if (quant_method == "awq") {
        return std::make_shared<infinilm::quantization::AWQ>(quantization_config);
    } else if (quant_method == "gptq") {
        return std::make_shared<infinilm::quantization::GPTQ>(quantization_config);
    } else if (quant_method == "quark") {
        return std::make_shared<infinilm::quantization::MXFP4>(quantization_config);
    } else {
        return std::make_shared<infinilm::quantization::NoneQuantization>(quantization_config);
    }
}
} // namespace infinilm::config
