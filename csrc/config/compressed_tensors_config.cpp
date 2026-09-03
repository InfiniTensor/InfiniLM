#include "compressed_tensors_config.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace infinilm::config {
namespace {

using json = nlohmann::json;

[[noreturn]] void config_error(const std::string &path, const std::string &message) {
    throw std::invalid_argument(path + ": " + message);
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string uppercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return value;
}

std::string require_string(const json &value, const std::string &path) {
    if (!value.is_string()) {
        config_error(path, "expected a string");
    }
    return value.get<std::string>();
}

int require_int(const json &value, const std::string &path) {
    if (!value.is_number_integer()) {
        config_error(path, "expected an integer");
    }
    const auto parsed = value.get<long long>();
    if (parsed < std::numeric_limits<int>::min()
        || parsed > std::numeric_limits<int>::max()) {
        config_error(path, "integer is outside the supported range");
    }
    return static_cast<int>(parsed);
}

std::vector<std::string> parse_string_list(const json &value, const std::string &path) {
    if (!value.is_array()) {
        config_error(path, "expected an array of strings");
    }

    std::vector<std::string> result;
    result.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        result.push_back(require_string(
            value.at(i), path + "[" + std::to_string(i) + "]"));
    }
    return result;
}

QuantizationValueType parse_value_type(const json &value, const std::string &path) {
    const auto type = lowercase(require_string(value, path));
    if (type == "int") {
        return QuantizationValueType::INT;
    }
    if (type == "float") {
        return QuantizationValueType::FLOAT;
    }
    config_error(path, "expected \"int\" or \"float\"");
}

QuantizationStrategy parse_strategy(const json &value, const std::string &path) {
    const auto strategy = lowercase(require_string(value, path));
    if (strategy == "tensor") {
        return QuantizationStrategy::TENSOR;
    }
    if (strategy == "channel") {
        return QuantizationStrategy::CHANNEL;
    }
    if (strategy == "group") {
        return QuantizationStrategy::GROUP;
    }
    if (strategy == "block") {
        return QuantizationStrategy::BLOCK;
    }
    if (strategy == "token") {
        return QuantizationStrategy::TOKEN;
    }
    if (strategy == "tensor_group") {
        return QuantizationStrategy::TENSOR_GROUP;
    }
    if (strategy == "attn_head") {
        return QuantizationStrategy::ATTN_HEAD;
    }
    config_error(path, "unknown quantization strategy \"" + strategy + "\"");
}

QuantizationDynamicMode parse_dynamic(const json &value, const std::string &path) {
    if (value.is_boolean()) {
        return value.get<bool>()
                 ? QuantizationDynamicMode::DYNAMIC
                 : QuantizationDynamicMode::STATIC;
    }
    if (value.is_string() && lowercase(value.get<std::string>()) == "local") {
        return QuantizationDynamicMode::LOCAL;
    }
    config_error(path, "expected false, true, or \"local\"");
}

QuantizationArgs parse_quantization_args(const json &value, const std::string &path) {
    if (!value.is_object()) {
        config_error(path, "expected an object");
    }

    QuantizationArgs args;
    if (value.contains("num_bits") && !value.at("num_bits").is_null()) {
        args.num_bits = require_int(value.at("num_bits"), path + ".num_bits");
        if (args.num_bits <= 0) {
            config_error(path + ".num_bits", "must be positive");
        }
    }
    if (value.contains("type") && !value.at("type").is_null()) {
        args.type = parse_value_type(value.at("type"), path + ".type");
    }
    if (value.contains("symmetric") && !value.at("symmetric").is_null()) {
        if (!value.at("symmetric").is_boolean()) {
            config_error(path + ".symmetric", "expected a boolean");
        }
        args.symmetric = value.at("symmetric").get<bool>();
    }
    if (value.contains("strategy") && !value.at("strategy").is_null()) {
        args.strategy = parse_strategy(value.at("strategy"), path + ".strategy");
    }
    if (value.contains("dynamic") && !value.at("dynamic").is_null()) {
        args.dynamic = parse_dynamic(value.at("dynamic"), path + ".dynamic");
    }

    // Fields outside InfiniLM's `W8A8` execution scope remain available through
    // `CompressedTensorsConfig::raw_config`. The scheme resolver will reject an
    // unsupported combination instead of silently selecting a kernel.
    return args;
}

QuantizationGroup make_w8a8_preset(
    const std::string &name,
    const json &targets,
    const std::string &path) {
    QuantizationGroup group;
    group.name = name;
    group.targets = parse_string_list(targets, path);

    QuantizationArgs weights;
    weights.type = QuantizationValueType::INT;
    weights.strategy = QuantizationStrategy::CHANNEL;
    group.weights = weights;

    QuantizationArgs inputs;
    inputs.type = QuantizationValueType::INT;
    inputs.strategy = QuantizationStrategy::TOKEN;
    inputs.dynamic = QuantizationDynamicMode::DYNAMIC;
    group.input_activations = inputs;
    return group;
}

QuantizationGroup parse_group(
    const std::string &name,
    const json &value,
    const std::string &path) {
    if (value.is_array()) {
        const auto preset = uppercase(name);
        if (preset == "W8A8" || preset == "INT8") {
            return make_w8a8_preset(name, value, path);
        }
        if (preset == "UNQUANTIZED") {
            QuantizationGroup group;
            group.name = name;
            group.targets = parse_string_list(value, path);
            return group;
        }
        config_error(
            path,
            "unsupported preset \"" + name + "\"; use an explicit group object");
    }
    if (!value.is_object()) {
        config_error(path, "expected a group object or preset target list");
    }
    if (!value.contains("targets")) {
        config_error(path + ".targets", "missing required field");
    }

    QuantizationGroup group;
    group.name = name;
    group.targets = parse_string_list(value.at("targets"), path + ".targets");

    if (value.contains("weights") && !value.at("weights").is_null()) {
        group.weights = parse_quantization_args(value.at("weights"), path + ".weights");
    }
    if (value.contains("input_activations")
        && !value.at("input_activations").is_null()) {
        group.input_activations = parse_quantization_args(
            value.at("input_activations"), path + ".input_activations");
    }
    if (value.contains("output_activations")
        && !value.at("output_activations").is_null()) {
        group.output_activations = parse_quantization_args(
            value.at("output_activations"), path + ".output_activations");
    }
    if (value.contains("format") && !value.at("format").is_null()) {
        group.format = require_string(value.at("format"), path + ".format");
    }
    return group;
}

QuantizationStatus parse_status(const json &value, const std::string &path) {
    const auto status = lowercase(require_string(value, path));
    if (status == "initialized") {
        return QuantizationStatus::INITIALIZED;
    }
    if (status == "calibration") {
        return QuantizationStatus::CALIBRATION;
    }
    if (status == "frozen") {
        return QuantizationStatus::FROZEN;
    }
    if (status == "compressed") {
        return QuantizationStatus::COMPRESSED;
    }
    if (status == "decompressed") {
        return QuantizationStatus::DECOMPRESSED;
    }
    config_error(path, "unknown quantization status \"" + status + "\"");
}

} // namespace

CompressedTensorsConfig CompressedTensorsConfig::from_json(const json &config) {
    constexpr std::string_view root = "quantization_config";
    if (!config.is_object()) {
        config_error(std::string(root), "expected an object");
    }

    CompressedTensorsConfig parsed;
    parsed.raw_config = config;

    if (config.contains("quant_method") && !config.at("quant_method").is_null()) {
        parsed.quant_method = require_string(
            config.at("quant_method"), std::string(root) + ".quant_method");
    }
    if (parsed.quant_method != "compressed-tensors") {
        config_error(
            std::string(root) + ".quant_method",
            "expected \"compressed-tensors\"");
    }
    if (config.contains("format") && !config.at("format").is_null()) {
        parsed.format = require_string(
            config.at("format"), std::string(root) + ".format");
    }
    if (config.contains("quantization_status")
        && !config.at("quantization_status").is_null()) {
        parsed.quantization_status = parse_status(
            config.at("quantization_status"),
            std::string(root) + ".quantization_status");
    }
    if (config.contains("ignore") && !config.at("ignore").is_null()) {
        parsed.ignore = parse_string_list(
            config.at("ignore"), std::string(root) + ".ignore");
    }
    if (config.contains("kv_cache_scheme")
        && !config.at("kv_cache_scheme").is_null()) {
        parsed.kv_cache_scheme = parse_quantization_args(
            config.at("kv_cache_scheme"),
            std::string(root) + ".kv_cache_scheme");
    }
    if (config.contains("global_compression_ratio")
        && !config.at("global_compression_ratio").is_null()) {
        const auto &ratio = config.at("global_compression_ratio");
        if (!ratio.is_number()) {
            config_error(
                std::string(root) + ".global_compression_ratio",
                "expected a number");
        }
        parsed.global_compression_ratio = ratio.get<double>();
    }

    if (!config.contains("config_groups")) {
        config_error(std::string(root) + ".config_groups", "missing required field");
    }
    const auto &groups = config.at("config_groups");
    if (!groups.is_object()) {
        config_error(std::string(root) + ".config_groups", "expected an object");
    }
    parsed.config_groups.reserve(groups.size());
    for (const auto &[name, group] : groups.items()) {
        parsed.config_groups.push_back(parse_group(
            name,
            group,
            std::string(root) + ".config_groups." + name));
    }
    return parsed;
}

} // namespace infinilm::config
