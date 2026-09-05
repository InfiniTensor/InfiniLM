#include "quant_config.hpp"

#include <stdexcept>

namespace infinilm::config {
namespace {

bool is_unquantized(const QuantizationGroup &group) {
    return !group.weights.has_value()
        && !group.input_activations.has_value()
        && !group.output_activations.has_value();
}

bool is_supported_w8a8(const QuantizationGroup &group) {
    if (!group.weights.has_value()
        || !group.input_activations.has_value()
        || group.output_activations.has_value()) {
        return false;
    }

    const auto &weights = *group.weights;
    const auto &inputs = *group.input_activations;
    return weights.num_bits == 8
        && weights.type == QuantizationValueType::INT
        && weights.symmetric
        && weights.strategy == QuantizationStrategy::CHANNEL
        && weights.dynamic == QuantizationDynamicMode::STATIC
        && inputs.num_bits == 8
        && inputs.type == QuantizationValueType::INT
        && inputs.symmetric
        && inputs.strategy == QuantizationStrategy::TOKEN
        && inputs.dynamic == QuantizationDynamicMode::DYNAMIC;
}

} // namespace

QuantConfig::QuantConfig(const nlohmann::json &json) : quantization_config(json) {
    if (!quantization_config.is_null()
        && quantization_config.value("quant_method", "") == "compressed-tensors") {
        compressed_tensors_config_ = CompressedTensorsConfig::from_json(quantization_config);
    }
    quantization_method = get_quantization_method();
}

std::shared_ptr<infinilm::quantization::BaseQuantization>
QuantConfig::get_quantization_method() const {
    if (quantization_config.is_null()) {
        return std::make_shared<infinilm::quantization::NoneQuantization>(quantization_config); // Default case if no matching scheme
    }

    const std::string quant_method = quantization_config.value("quant_method", "");

    // Determine the quantization scheme from the JSON config
    if (quant_method == "compressed-tensors") {
        return std::make_shared<infinilm::quantization::CompressedTensors>(quantization_config);
    } else if (quant_method == "awq") {
        return std::make_shared<infinilm::quantization::AWQ>(quantization_config);
    } else if (quant_method == "gptq") {
        return std::make_shared<infinilm::quantization::GPTQ>(quantization_config);
    } else if (quant_method == "quark") {
        return std::make_shared<infinilm::quantization::MXFP4>(quantization_config);
    }
    // Add other schemes as needed

    return std::make_shared<infinilm::quantization::NoneQuantization>(quantization_config); // Default case if no matching scheme
}

std::shared_ptr<infinilm::quantization::BaseQuantization>
QuantConfig::get_quantization_method(
    std::string_view module_name,
    std::string_view module_type) const {
    if (!compressed_tensors_config_.has_value()) {
        return get_quantization_method();
    }

    const auto &config = *compressed_tensors_config_;
    const auto *group = config.resolve_group(module_name, module_type);
    if (group == nullptr || is_unquantized(*group)) {
        return std::make_shared<infinilm::quantization::NoneQuantization>(
            quantization_config);
    }

    const auto &format = group->format.value_or(config.format);
    if (format != "int-quantized") {
        throw std::invalid_argument(
            "unsupported `compressed-tensors` format `" + format
            + "` for module `" + std::string(module_name) + "`");
    }
    if (!is_supported_w8a8(*group)) {
        throw std::invalid_argument(
            "unsupported `compressed-tensors` scheme in group `" + group->name
            + "` for module `" + std::string(module_name)
            + "`; expected static symmetric INT8 channel weights and dynamic "
              "symmetric INT8 per-token input activations");
    }

    return std::make_shared<infinilm::quantization::CompressedTensors>(
        quantization_config);
}
} // namespace infinilm::config
