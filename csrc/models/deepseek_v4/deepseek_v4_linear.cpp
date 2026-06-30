#include "deepseek_v4_linear.hpp"

#include "../../layers/quantization/none_quantization.hpp"

#include <string>

namespace infinilm::models::deepseek_v4 {

bool use_deepseek_v4_w8a8_linear(const std::shared_ptr<infinilm::config::ModelConfig> &model_config) {
    const auto &config_json = model_config->get_config_json();
    const nlohmann::json *quant_config = nullptr;
    if (config_json.contains("quantization_config")) {
        quant_config = &config_json.at("quantization_config");
    } else if (config_json.contains("compression_config")) {
        quant_config = &config_json.at("compression_config");
    }

    if (quant_config == nullptr || quant_config->is_null()) {
        return false;
    }

    const std::string quant_method = quant_config->value("quant_method", "");
    return quant_method == "compressed-tensors" || quant_method == "compressed_tensors";
}

std::shared_ptr<infinilm::quantization::BaseQuantization> deepseek_v4_linear_quantization(
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
    bool use_quantization) {
    if (use_quantization && use_deepseek_v4_w8a8_linear(model_config)) {
        return model_config->get_quantization_method();
    }
    return std::make_shared<infinilm::quantization::NoneQuantization>(nullptr);
}

} // namespace infinilm::models::deepseek_v4
