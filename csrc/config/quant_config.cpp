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
        throw std::runtime_error(
            "`compressed-tensors` quantization is unsupported because InfiniLM lacks the end-to-end activation quantization and scaled-GEMM integration.");
    } else if (quant_method == "awq") {
        throw std::runtime_error(
            "AWQ quantization is unsupported because InfiniLM has no InfiniOps-backed dense AWQ GEMM path.");
    } else if (quant_method == "gptq") {
        throw std::runtime_error(
            "GPTQ quantization is unsupported because InfiniLM has no InfiniOps-backed dense GPTQ/Marlin GEMM path.");
    } else if (quant_method == "quark") {
        throw std::runtime_error(
            "MXFP4 quantization is unsupported because InfiniLM has no InfiniOps-backed MXFP4 execution path.");
    } else {
        return std::make_shared<infinilm::quantization::NoneQuantization>(quantization_config);
    }
}
} // namespace infinilm::config
