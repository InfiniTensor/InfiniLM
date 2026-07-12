#include "quant_config.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>

namespace infinilm::config {
namespace {

std::string lower_string(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string env_string(const char *name) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return {};
    }
    return lower_string(value);
}

bool truthy_env(const char *name) {
    auto value = env_string(name);
    return value == "1" || value == "true" || value == "on" || value == "yes";
}

bool is_w16a16_marlin_method(const std::string &method) {
    return method == "w16a16_marlin" || method == "hygon_w16a16_marlin";
}

bool is_w8a8_marlin_method(const std::string &method) {
    return method == "slimquant_marlin" || method == "slimquant_compressed_tensors_marlin" ||
           method == "w8a8_marlin" || method == "hygon_w8a8_marlin";
}

bool is_unquantized_config(const nlohmann::json &quantization_config) {
    if (quantization_config.is_null()) {
        return true;
    }
    if (!quantization_config.is_object()) {
        return false;
    }
    auto it = quantization_config.find("quant_method");
    if (it == quantization_config.end() || it->is_null()) {
        return true;
    }
    if (!it->is_string()) {
        return false;
    }
    auto method = lower_string(it->get<std::string>());
    return method.empty() || method == "none" || method == "dense";
}

std::string explicit_moe_weight_method(const nlohmann::json &quantization_config) {
    if (!quantization_config.is_object()) {
        return {};
    }
    for (const char *key : {"moe_weight_method", "weight_method", "moe_kernel_method"}) {
        auto it = quantization_config.find(key);
        if (it != quantization_config.end() && it->is_string()) {
            return lower_string(it->get<std::string>());
        }
    }
    auto it = quantization_config.find("quant_method");
    if (it != quantization_config.end() && it->is_string()) {
        auto method = lower_string(it->get<std::string>());
        if (is_w16a16_marlin_method(method) || is_w8a8_marlin_method(method)) {
            return method;
        }
    }
    return {};
}

} // namespace
QuantConfig::QuantConfig(const nlohmann::json &json) : quantization_config(json) {
    this->quantization_method = get_quantization_method();
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
    } else if (quantization_config["quant_method"] == "w16a16_marlin" ||
               quantization_config["quant_method"] == "hygon_w16a16_marlin") {
        return std::make_shared<infinilm::quantization::NoneQuantization>(quantization_config);
    } else {
        return std::make_shared<infinilm::quantization::NoneQuantization>(quantization_config);
    }
    // Add other schemes as needed

    return std::make_shared<infinilm::quantization::NoneQuantization>(quantization_config); // Default case if no matching scheme
}

std::string QuantConfig::get_moe_weight_method() const {
    return get_moe_weight_method(infinicore::Device(infinicore::Device::Type::CPU, 0));
}

std::string QuantConfig::get_moe_weight_method(const infinicore::Device &device) const {
    auto env_method = env_string("INFINILM_MOE_WEIGHT_METHOD");
    if (!env_method.empty()) {
        return env_method;
    }
    if (truthy_env("INFINILM_HYGON_MOE_W16A16_MARLIN")) {
        return "w16a16_marlin";
    }
    auto configured_method = explicit_moe_weight_method(quantization_config);
    if (!configured_method.empty()) {
        return configured_method;
    }
    if (quantization_method != nullptr) {
        auto method = quantization_method->get_moe_weight_method(device);
        if (method != "dense") {
            return method;
        }
    }
    if (device.getType() == infinicore::Device::Type::HYGON && is_unquantized_config(quantization_config)) {
        return "hygon_w16a16_marlin";
    }
    return "dense";
}

bool QuantConfig::is_moe_w16a16_marlin_enabled() const {
    return is_w16a16_marlin_method(get_moe_weight_method());
}

bool QuantConfig::is_moe_w16a16_marlin_enabled(const infinicore::Device &device) const {
    return is_w16a16_marlin_method(get_moe_weight_method(device));
}

bool QuantConfig::is_moe_w8a8_marlin_enabled(const infinicore::Device &device) const {
    return is_w8a8_marlin_method(get_moe_weight_method(device));
}

} // namespace infinilm::config
